"""Check real query accounting, isolation, fail-fast validation and local artifacts."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.forrester import ForresterTask, derive_seed, generate_manifest
from experiments.evaluate import content_hash, run_experiment, run_single_task, validate_config
from experiments.metrics import estimate_reference_range
from experiments.run_forrester import main
from models.random_search import RandomSearch


def test_total_budget_counting_and_incremental_observations():
    task = ForresterTask("counter", 1, 0, 0)
    reference = estimate_reference_range(task)

    class CountingTask:
        task_id = "counter"
        points_evaluated = 0

        def evaluate(self, X):
            self.points_evaluated += len(X)
            return task.evaluate(X)

    class Spy(RandomSearch):
        sizes = []

        def suggest(self, n=1):
            self.sizes.append(len(self.history.y))
            return super().suggest(n)

    counter = CountingTask()
    model = Spy([[0, 1]], seed=44)
    result = run_single_task(model, counter, 8, 2, 0, reference=reference)
    assert counter.points_evaluated == 8
    assert model.sizes == [2, 3, 4, 5, 6, 7]
    assert [r["phase"] for r in result.observations] == ["initial"] * 2 + ["sequential"] * 6
    assert [r["query_index"] for r in result.observations] == list(range(8))
    assert result.optimizer_seed == 44
    assert np.all(np.diff([r["best_so_far"] for r in result.observations]) <= 0)
    assert all(r["simple_regret"] >= 0 for r in result.observations)


@pytest.mark.parametrize("proposal", [np.array([[2.0]]), np.array([0.5]), np.array([[np.nan]]), np.array([[0.1], [0.2]])])
def test_invalid_optimizer_proposal_fails_before_objective(proposal):
    class Bad(RandomSearch):
        def suggest(self, n=1):
            return proposal
    task = ForresterTask("invalid", 1, 0, 0)
    reference = estimate_reference_range(task)
    # If an invalid proposal reaches objective, that is a failed test.
    class NeverEvaluate:
        task_id = "invalid"
        def evaluate(self, X):
            pytest.fail("Invalid proposal reached the objective")
    with pytest.raises(ValueError):
        run_single_task(Bad([[0, 1]]), NeverEvaluate(), 1, 0, 0, reference=reference)


def test_history_mutation_during_suggest_is_detected():
    class Leaky(RandomSearch):
        def suggest(self, n=1):
            self.observe([[0.1]], [-999.0])
            return np.array([[0.2]])
    with pytest.raises(ValueError, match="changed, leaked"):
        run_single_task(Leaky([[0, 1]]), ForresterTask("target", 1, 0, 0), 2, 0, 0)


def test_wrong_reset_and_reordered_history_detected():
    class NoReset(RandomSearch):
        def reset(self):
            pass
    model = NoReset([[0, 1]])
    model.observe([[0.1]], [1])
    with pytest.raises(ValueError, match="empty after reset"):
        run_single_task(model, ForresterTask("target", 1, 0, 0), 1, 0, 0)

    class Reordered(RandomSearch):
        def observe(self, X, y):
            super().observe(X, y)
            self.history.X.reverse()
            self.history.y.reverse()
    with pytest.raises(ValueError, match="reordered"):
        run_single_task(Reordered([[0, 1]], seed=44), ForresterTask("target", 1, 0, 0), 3, 2, 0)


def test_source_only_meta_training_and_fresh_objects(small_config):
    created = []
    class Spy(RandomSearch):
        def __init__(self, bounds, seed):
            # Constructor receives no task descriptors, objective or oracle.
            super().__init__(bounds, seed)
            self.sources = None
            created.append(self)
        def meta_train(self, tasks):
            self.sources = tasks
            assert len(tasks) == 2
            assert all(t.X.shape == (3, 1) for t in tasks)
            assert all(np.allclose(t.objective(t.X), t.y) for t in tasks)
    small_config["evaluation"]["meta_train"] = True
    outcome = run_experiment(small_config, optimizer_factory=Spy)
    assert len(created) == 4
    assert len({id(m) for m in created}) == 4
    assert all(len(m.history.y) == 4 and m.sources is not None for m in created)
    manifest = json.loads((Path(outcome["output_directory"]) / "task_manifest.json").read_text())
    assert {t['task_id'] for t in manifest['meta_tasks']}.isdisjoint(t['task_id'] for t in manifest['target_tasks'])


def test_artifacts_and_completion_from_real_local_runs(small_config):
    outcome = run_experiment(small_config)
    root = Path(outcome["output_directory"])
    observations = list(csv.DictReader((root / "observations.csv").open()))
    runs = list(csv.DictReader((root / "runs.csv").open()))
    assert len(observations) == 16 and len(runs) == 4
    assert {"phase", "query_index", "x", "y", "seed", "task_id", "normalized_regret"} <= observations[0].keys()
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["n_runs"] == 4
    assert summary["table1_reproduction"] is False
    assert summary["exploratory_cvar"]["table1_equivalence"] == "UNVERIFIED"
    assert summary["metrics"]["final_normalized_regret"]["n"] == 4
    assert not (root / "wandb_run_ids.json").exists()
    for filename in ["environment.json", "config_resolved.yaml", "task_manifest.json", "completion.json",
                     "regret_curve.png", "best_so_far_curve.png"]:
        assert (root / filename).stat().st_size > 0
    assert (root / "regret_curve.png").read_bytes().startswith(b"\x89PNG")
    for run in runs:
        assert (root / "runs" / run["run_id"] / "regret_curve.png").is_file()


def test_optimizer_error_never_marked_as_success(small_config):
    class Bad(RandomSearch):
        def suggest(self, n=1):
            raise RuntimeError("synthetic model failure")
    small_config["evaluation"]["n_init"] = 0
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        run_experiment(small_config, optimizer_factory=Bad)
    failures = list(Path(small_config["output_dir"]).glob("*/runs/*.failed.json"))
    assert len(failures) == 1 and json.loads(failures[0].read_text())["status"] == "failed"
    assert not list(Path(small_config["output_dir"]).glob("*/summary.json"))


def test_manifest_rejects_config_mismatch(small_config, tmp_path):
    wrong = generate_manifest(2, 1, 3, 44, 45)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(wrong))
    small_config["benchmark"]["manifest_path"] = str(path)
    with pytest.raises(ValueError, match="do not match config"):
        run_experiment(small_config)


@pytest.mark.parametrize("path,value", [(('evaluation','budget_mode'), 'additional'),
    (('evaluation','seeds'), []), (('evaluation','seeds'), [0,0]), (('evaluation','seeds'), [-1]),
    (('evaluation','meta_train'), 1), (('benchmark','num_target_tasks'), 0), (('model',), 'pfn')])
def test_invalid_config(small_config, path, value):
    target = small_config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(small_config)


def test_cli_refuses_accidental_full_execution():
    with pytest.raises(SystemExit) as exc:
        main(["--config", "configs/forrester_paper_candidate.yaml", "--mode", "smoke"])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        main(["--config", "configs/forrester_smoke.yaml", "--mode", "full"])
    assert exc.value.code == 1


@pytest.mark.parametrize("n_init", [0, 4])
def test_zero_and_all_initialization_budgets(n_init):
    result = run_single_task(RandomSearch([[0, 1]], seed=44), ForresterTask("initial", 1, 0, 0), 4, n_init, 0)
    assert len(result.observations) == 4
    assert sum(r["phase"] == "initial" for r in result.observations) == n_init


def test_supplied_manifest_is_paired_and_input_config_not_mutated(small_config, tmp_path):
    manifest = generate_manifest(2, 2, 3, 44, 45)
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(manifest))
    small_config["benchmark"]["manifest_path"] = str(path)
    before = copy.deepcopy(small_config)
    result = run_experiment(small_config)
    assert small_config == before
    assert result["summary"]["manifest_sha256"] == content_hash(manifest)


def test_phase2_factory_and_model_options_use_same_evaluator(small_config):
    class FutureAdapter(RandomSearch):
        def __init__(self, bounds, seed, scale):
            assert scale == 2
            super().__init__(bounds, seed)
    small_config["model"] = "test_adapter"
    small_config["model_options"] = {"scale": 2}
    result = run_experiment(small_config, optimizer_factory=FutureAdapter)
    assert result["summary"]["algorithm"] == "test_adapter"


@pytest.mark.parametrize("section,field,value", [("tracking", "api_key", "do-not-store-credentials"),
    ("evaluation", "buget", 4), ("benchmark", "num_tasks", 2)])
def test_config_rejects_unrecognized_or_sensitive_keys(small_config, section, field, value):
    small_config[section][field] = value
    with pytest.raises(ValueError):
        validate_config(small_config)
