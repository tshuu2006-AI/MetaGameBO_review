"""Shared evaluator integration: pairing, cold start, repeat/resume and CLI gates."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.forrester import ForresterTask
from experiments import evaluate, run_forrester
from experiments.evaluate import content_hash, run_experiment, run_single_task, validate_config
from experiments.metrics import estimate_reference_range
from experiments.run_forrester import load_config, main
from models.lfbo import LFBO
from models.random_search import RandomSearch


@pytest.fixture
def lfbo_config(small_config):
    small_config.update(model="lfbo", model_options={"n_estimators": 5, "n_candidates": 32})
    return small_config


@pytest.mark.parametrize("n_init", [0, 2, 4])
def test_lfbo_budget_cold_start_and_paired_initialization(n_init):
    task = ForresterTask("paired-lfbo", 1, 0, 0)
    reference = estimate_reference_range(task)

    class CountingTask:
        task_id = task.task_id
        count = 0

        def evaluate(self, X):
            self.count += len(X)
            return task.evaluate(X)

    counted = CountingTask()
    model = LFBO([[0, 1]], seed=99, n_estimators=5, n_candidates=32)
    actual = run_single_task(model, counted, 4, n_init, 7, {"algorithm": "lfbo"}, reference=reference)
    random = run_single_task(RandomSearch([[0, 1]], seed=111), task, 4, n_init, 7, reference=reference)
    assert counted.count == 4
    assert len(model.history.y) == len(actual.observations) == 4
    assert [r["phase"] for r in actual.observations] == ["initial"] * n_init + ["sequential"] * (4 - n_init)
    assert [(r["x"], r["y"]) for r in actual.observations[:n_init]] == [(r["x"], r["y"]) for r in random.observations[:n_init]]
    assert actual.optimizer_diagnostics["fit_count"] == 3
    assert np.all(np.diff([r["best_so_far"] for r in actual.observations]) <= 0)


def test_full_experiment_repeat_pairing_nonmeta_and_resume(lfbo_config, tmp_path, monkeypatch):
    constructed = []

    class ObservedOnlyLFBO(LFBO):
        def __init__(self, bounds, seed, **options):
            assert set(options) == {"n_estimators", "n_candidates"}
            super().__init__(bounds, seed, **options)
            constructed.append(self)

        def meta_train(self, tasks):
            pytest.fail("LFBO must never receive source tasks")

    original = copy.deepcopy(lfbo_config)
    first = run_experiment(lfbo_config, optimizer_factory=ObservedOnlyLFBO)
    assert lfbo_config == original
    assert len(constructed) == 4 and len({id(m) for m in constructed}) == 4
    assert all(len(m.history.y) == 4 and m.fit_count == 3 for m in constructed)
    repeat_config = copy.deepcopy(lfbo_config)
    repeat_config["output_dir"] = str(tmp_path / "repeat")
    repeat = run_experiment(repeat_config)  # default dispatch must use LFBO
    random_config = copy.deepcopy(lfbo_config)
    random_config.update(model="random", output_dir=str(tmp_path / "random"))
    random_config.pop("model_options")
    random = run_experiment(random_config)
    paths = [Path(r["output_directory"]) for r in (first, repeat, random)]
    tables = [list(csv.DictReader((p / "observations.csv").open())) for p in paths]
    strip_time = lambda rows: [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in rows]
    assert strip_time(tables[0]) == strip_time(tables[1])
    for key in ("metrics", "exploratory_cvar", "config_hash", "manifest_sha256"):
        assert first["summary"][key] == repeat["summary"][key]
    assert len({r["summary"]["manifest_sha256"] for r in (first, repeat, random)}) == 1
    initials = lambda rows: [(r["task_id"], r["seed"], r["x"], r["y"]) for r in rows if r["phase"] == "initial"]
    assert initials(tables[0]) == initials(tables[2])
    env = json.loads((paths[0] / "environment.json").read_text())
    assert env["packages"]["scikit-learn"]
    diagnostics = list((paths[0] / "runs").glob("*/optimizer_diagnostics.json"))
    assert len(diagnostics) == 4
    for p in diagnostics:
        payload = json.loads(p.read_text())
        assert payload["method"] == "lfbo_ei_rf" and payload["fit_count"] == 3
        assert not {"oracle", "a", "b", "c", "target_task"} & set(payload)
    assert not (paths[0] / "wandb_run_ids.json").exists()
    # This is stronger than checking a skipped counter: no model or objective
    # may be created/called when every verified cache already exists.
    caches = {p.name: p.read_bytes() for p in (paths[0] / "runs").glob("*.json")}
    monkeypatch.setattr(ForresterTask, "evaluate", lambda *a: pytest.fail("Resume evaluated an objective"))
    monkeypatch.setattr(evaluate, "generate_manifest", lambda *a: json.loads((paths[0] / "task_manifest.json").read_text()))
    resumed = run_experiment(lfbo_config, resume=True,
                             optimizer_factory=lambda **kw: pytest.fail("Resume constructed an optimizer"))
    assert resumed["skipped_completed_runs"] == 4 and resumed["summary"] == first["summary"]
    assert caches == {p.name: p.read_bytes() for p in (paths[0] / "runs").glob("*.json")}


@pytest.mark.parametrize("change", ["meta", "gamma", "unknown", "hidden", "missing"])
def test_lfbo_bad_config_fails_before_run(lfbo_config, change):
    if change == "meta":
        lfbo_config["evaluation"]["meta_train"] = True
    elif change == "missing":
        del lfbo_config["benchmark"]
    else:
        key, value = {"gamma": ("gamma", 0), "unknown": ("treez", 1), "hidden": ("oracle", -10)}[change]
        lfbo_config["model_options"][key] = value
    with pytest.raises(ValueError):
        validate_config(lfbo_config)


def test_cli_dispatch_and_refusal_gates(tmp_path, monkeypatch):
    called = []

    def capture(config, **kwargs):
        called.append((config, kwargs))
        assert kwargs["optimizer_factory"] is LFBO
        return {"test_capture_only": True}

    monkeypatch.setattr(run_forrester, "run_experiment", capture)
    assert main(["--config", "configs/lfbo_smoke.yaml", "--model", "lfbo", "--mode", "smoke",
                 "--output-dir", str(tmp_path)]) == 0
    assert len(called) == 1
    for arguments in (["--model", "unknown"], ["--config", "configs/lfbo_paper_candidate.yaml"],
                      ["--config", "configs/lfbo_development.yaml", "--mode", "smoke"]):
        with pytest.raises(SystemExit) as exc:
            main(arguments)
        assert exc.value.code != 0
    assert len(called) == 1


def test_prespecified_e2_configs_are_fair_and_e4_is_explicit():
    lfbo = load_config("configs/lfbo_development.yaml")
    random = load_config("configs/random_paired_development.yaml")
    assert lfbo["benchmark"] == random["benchmark"]
    assert lfbo["evaluation"] == random["evaluation"]
    assert len(lfbo["evaluation"]["seeds"]) == 10 and lfbo["evaluation"]["budget"] == 16
    candidate = validate_config(load_config("configs/lfbo_paper_candidate.yaml"))
    assert candidate["protocol_label"] == "paper_candidate_UNVERIFIED"
    assert candidate["benchmark"]["num_meta_tasks"] == 0
    assert candidate["evaluation"]["meta_train"] is False


def test_diagnostics_tampering_rejected_on_resume(lfbo_config):
    first = run_experiment(lfbo_config)
    path = next(Path(first["output_directory"]).glob("runs/*/optimizer_diagnostics.json"))
    payload = json.loads(path.read_text())
    payload["fit_count"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="diagnostics"):
        run_experiment(lfbo_config, resume=True)
