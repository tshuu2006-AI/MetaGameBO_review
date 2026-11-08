"""Paired evaluator tests with isolated doubles, plus explicitly selected REAL PFN tests."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.forrester import ForresterTask
from experiments import run_forrester
from experiments.evaluate import run_experiment, run_single_task, validate_config
from experiments.metrics import estimate_reference_range
from experiments.run_forrester import load_config, main
from models.pfn import PFNOptimizer
from models.random_search import RandomSearch
from models.lfbo import LFBO


class TestOnlyBackend:
    __test__ = False
    metadata = {"backend": "ISOLATED_TEST_DOUBLE_ONLY"}

    def score(self, X, y, candidates, *, seed):
        assert X.shape == (len(y), 1) and candidates.shape[1] == 1
        assert np.isfinite(y).all() and np.all((X >= 0) & (X <= 1))
        return np.linspace(.01, .5, len(candidates))


@pytest.mark.parametrize("n_init", [0, 2, 4])
def test_budget_and_initial_xy_paired_pfn_lfbo_random(n_init):
    task = ForresterTask("paired-pfn", 1, 0, 0)
    reference = estimate_reference_range(task)
    class Counter:
        task_id = task.task_id
        count = 0
        def evaluate(self, X):
            self.count += len(X)
            return task.evaluate(X)
    counted = Counter()
    pfn = run_single_task(PFNOptimizer([[0, 1]], seed=17, backend=TestOnlyBackend(), n_candidates=4),
                          counted, 4, n_init, 0, {"algorithm": "pfn"}, reference=reference)
    others = [run_single_task(m, task, 4, n_init, 0, reference=reference) for m in
              [RandomSearch([[0, 1]], seed=18), LFBO([[0, 1]], seed=19, n_estimators=5, n_candidates=16)]]
    assert counted.count == 4
    for other in others:
        assert [(r["x"], r["y"]) for r in pfn.observations[:n_init]] == [(r["x"], r["y"]) for r in other.observations[:n_init]]
    assert pfn.optimizer_diagnostics["gradient_updates"] == 0


def test_factory_receives_no_hidden_data_exports_and_resume(small_config, tmp_path, monkeypatch):
    from models import pfn_backend
    monkeypatch.setattr(pfn_backend, "checkpoint_metadata", lambda *a: {"sha256": "ISOLATED_UNIT_TEST_ONLY"})
    small_config.update(model="pfn", model_options={"n_candidates": 4})
    made = []
    def factory(bounds, seed, **options):
        assert not {"a", "b", "c", "reference", "task", "oracle"} & set(options)
        optimizer = PFNOptimizer(bounds, seed, backend=TestOnlyBackend(), **options)
        made.append(optimizer)
        return optimizer
    actual = run_experiment(small_config, optimizer_factory=factory)
    assert len(made) == 4 and len({id(m) for m in made}) == 4
    path = Path(actual["output_directory"])
    assert len(list(path.glob("runs/*/optimizer_diagnostics.json"))) == 4
    assert json.loads((path / "environment.json").read_text())["pfn_checkpoint"]["sha256"] == "ISOLATED_UNIT_TEST_ONLY"
    result = run_experiment(small_config, resume=True, optimizer_factory=lambda **kw: pytest.fail("Resume instantiated PFN"))
    assert result["skipped_completed_runs"] == 4


@pytest.mark.parametrize("option,value", [("oracle", -100), ("backend", "fake"), ("bounds", [[0, 1]]),
    ("seed", 0), ("n_candidates", 0), ("device", "cuda")])
def test_unknown_or_invalid_pfn_config(small_config, option, value):
    small_config.update(model="pfn", model_options={option: value})
    with pytest.raises(ValueError):
        validate_config(small_config)


def test_group_source_meta_training_rejected(small_config):
    small_config.update(model="pfn")
    small_config["evaluation"]["meta_train"] = True
    with pytest.raises(ValueError, match="meta_train must be false"):
        validate_config(small_config)


def test_cli_pfn_registry_without_inference(monkeypatch):
    calls = []
    def capture(config, **kw):
        assert kw["optimizer_factory"] is PFNOptimizer
        calls.append(config)
        return {"isolated_test_capture": True}
    monkeypatch.setattr(run_forrester, "run_experiment", capture)
    assert main(["--config", "configs/pfn_tiny.yaml", "--model", "pfn", "--mode", "smoke"]) == 0
    assert calls[0]["benchmark"]["num_target_tasks"] == 1
    assert calls[0]["evaluation"]["seeds"] == [0] and calls[0]["evaluation"]["budget"] == 4
    with pytest.raises(SystemExit):
        main(["--config", "configs/pfn_paper_candidate.yaml"])
    assert len(calls) == 1


def test_config_pairing_with_phase2_and_shared_manifest():
    pfn, lfbo = load_config("configs/pfn_smoke.yaml"), load_config("configs/lfbo_smoke.yaml")
    assert pfn["evaluation"] == lfbo["evaluation"]
    pfn["benchmark"].pop("manifest_path")
    lfbo["benchmark"].pop("manifest_path")
    assert pfn["benchmark"] == lfbo["benchmark"]
    dev, lfbo_dev = load_config("configs/pfn_development.yaml"), load_config("configs/lfbo_development.yaml")
    assert dev["evaluation"] == lfbo_dev["evaluation"]


@pytest.fixture(scope="module")
def real_backend():
    from models.pfn_backend import PFNs4BOBackend
    return PFNs4BOBackend(candidate_batch_size=64, cpu_threads=2)


@pytest.mark.pfn_real
def test_real_official_weights_eval_and_forward_without_parameter_updates(real_backend):
    def parameter_hash():
        digest = hashlib.sha256()
        for parameter in real_backend.model.parameters():
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    before = parameter_hash()
    scores = real_backend.score(np.array([[.1], [.9]]), np.array([0., -2.]),
                               np.linspace(.2, .8, 64)[:, None], seed=7)
    assert scores.shape == (64,) and np.isfinite(scores).all() and real_backend.forward_calls > 0
    assert not real_backend.model.training and all(not p.requires_grad and p.grad is None for p in real_backend.model.parameters())
    assert real_backend.metadata["checkpoint"]["sha256"]
    assert before == parameter_hash()


@pytest.mark.pfn_real
@pytest.mark.parametrize("y", [[3.], [3., 3.], [-2., 0.]])
def test_real_single_two_constant_context_produces_actual_ei(real_backend, y):
    X = np.array([[.1], [.9]])[:len(y)]
    scores = real_backend.score(X, -np.array(y), np.linspace(.2, .8, 32)[:, None], seed=17)
    assert scores.shape == (32,) and np.isfinite(scores).all() and np.all(scores >= 0)


@pytest.mark.pfn_real
def test_real_reset_reproducible_and_torch_rng_isolation(real_backend):
    import torch
    pfn = PFNOptimizer([[0, 1]], seed=123, backend=real_backend, n_candidates=64)
    pfn.observe([[.1], [.9]], [0, 2])
    state = torch.get_rng_state().clone()
    first = pfn.suggest()
    assert torch.equal(state, torch.get_rng_state())
    pfn.reset()
    pfn.observe([[.2]], [3])
    pfn.suggest()
    assert pfn.context_y.tolist() == [3] and pfn.history.y == [3]
    repeat = PFNOptimizer([[0, 1]], seed=123, backend=real_backend, n_candidates=64)
    repeat.observe([[.1], [.9]], [0, 2])
    np.testing.assert_array_equal(first, repeat.suggest())


@pytest.mark.pfn_real
def test_real_forward_guard_does_not_retain_backend_cycle():
    import weakref
    from models.pfn_backend import PFNs4BOBackend
    backend = PFNs4BOBackend()
    model = backend.model
    reference = weakref.ref(backend)
    del backend
    # Even while an external caller holds the model, its hook must not retain
    # the whole backend. No gc.collect() is needed to break a reference cycle.
    assert reference() is None
    del model
