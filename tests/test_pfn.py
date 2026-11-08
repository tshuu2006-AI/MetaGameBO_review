"""Isolated adapter/backend contracts. Test doubles never enter real experiment logs."""
import builtins
import copy
import subprocess
import sys

import numpy as np
import pytest

from models.pfn import PFNOptimizer
from models import pfn_backend
from models.pfn_backend import checkpoint_metadata, standardize_observed_y


class FakeBackend:
    """UNIT TEST ONLY: deliberately not a PFN or a measured model."""
    metadata = {"backend": "TEST_ONLY_FAKE", "checkpoint": {"sha256": "TEST_ONLY"}}
    forward_calls = 0

    def __init__(self, scores=None):
        self.calls = []
        self.scores = scores

    def reset_context(self):
        self.calls.clear()

    def score(self, X, y, candidates, *, seed):
        self.calls.append((X.copy(), y.copy(), candidates.copy(), seed))
        return np.ones(len(candidates)) if self.scores is None else np.asarray(self.scores)


def model(**kwargs):
    return PFNOptimizer([[0, 1]], seed=7, backend=FakeBackend(), **kwargs)


def test_sign_flip_changes_maximize_to_minimize_and_stable_tie(monkeypatch):
    class SignSensitive(FakeBackend):
        def score(self, X, y, candidates, *, seed):
            super().score(X, y, candidates, seed=seed)
            best_coordinate = X[int(np.argmax(y)), 0]
            return np.exp(-np.abs(candidates[:, 0] - best_coordinate))
    backend = SignSensitive()
    opt = PFNOptimizer([[0, 1]], seed=7, backend=backend, n_candidates=2)
    opt.observe([[.1], [.9]], [0, 10])
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.array([[.2], [.8]]))
    np.testing.assert_equal(opt.suggest(), [[.2]])
    np.testing.assert_equal(backend.calls[0][1], [0, -10])
    # Adapter always chooses first maximum, irrespective of upstream random index.
    backend.score = lambda *a, **kw: np.array([.5, .5])
    np.testing.assert_equal(opt.suggest(), [[.2]])


def test_normalize_observations_and_candidates_with_same_fixed_bounds(monkeypatch):
    backend = FakeBackend()
    opt = PFNOptimizer([[-2, 2]], seed=8, backend=backend, n_candidates=2)
    opt.observe([[-1], [1]], [-4, 2])
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.array([[-.5], [.5]]))
    np.testing.assert_equal(opt.suggest(), [[-.5]])
    X, y, candidates, _ = backend.calls[0]
    np.testing.assert_equal(X, [[.25], [.75]])
    np.testing.assert_equal(candidates, [[.375], [.625]])
    np.testing.assert_equal(y, [4, -2])


def test_duplicate_filter_runs_before_backend_and_does_not_mutate_history(monkeypatch):
    backend = FakeBackend()
    opt = PFNOptimizer([[0, 1]], seed=1, backend=backend, n_candidates=4)
    opt.observe([[.1], [.9]], [1, 2])
    before = copy.deepcopy(opt.history.arrays)
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.array([[.1], [.1 + 5e-9], [.3], [.8]]))
    point = opt.suggest()
    assert point.shape == (1, 1) and np.isfinite(point).all()
    np.testing.assert_equal(backend.calls[0][2], [[.3], [.8]])
    for a, b in zip(before, opt.history.arrays):
        np.testing.assert_equal(a, b)


@pytest.mark.parametrize("scores", [[True, False], [np.nan, 1], [np.inf, 1], [-.1, 1], [[1], [2]], 1])
def test_invalid_scores_raise_instead_of_random_substitution(scores, monkeypatch):
    opt = PFNOptimizer([[0, 1]], seed=7, backend=FakeBackend(scores), n_candidates=2)
    opt.observe([[.1]], [0])
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.array([[.2], [.8]]))
    with pytest.raises(RuntimeError, match="no Random substitution"):
        opt.suggest()
    assert opt.events[-1]["status"] == "error"


def test_inference_exception_not_swallowed(monkeypatch):
    opt = model(n_candidates=2)
    opt.observe([[.1]], [1])
    def fail(*args, **kwargs):
        raise MemoryError("synthetic isolated backend failure")
    monkeypatch.setattr(opt.backend, "score", fail)
    with pytest.raises(RuntimeError, match="synthetic isolated backend failure"):
        opt.suggest()
    assert opt.events[-1]["exception_type"] == "MemoryError"


@pytest.mark.parametrize("pool", [[[np.nan]], [[1.1]], [np.inf], [[.2], [.3]]])
def test_invalid_candidates_fail_before_backend(pool, monkeypatch):
    opt = model(n_candidates=1)
    opt.observe([[.1]], [0])
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.asarray(pool))
    with pytest.raises(ValueError):
        opt.suggest()
    assert not opt.backend.calls


@pytest.mark.parametrize("options", [{"n_candidates": 0}, {"n_candidates": True}, {"candidate_batch_size": 0},
    {"cpu_threads": -1}, {"duplicate_atol": np.nan}, {"duplicate_atol": -.1}, {"device": "cuda"},
    {"model_name": "bnn"}, {"acquisition": "pi"}, {"candidate_strategy": "unknown"},
    {"context_transform": "future_y_minmax"}, {"checkpoint": 17}])
def test_invalid_model_options(options):
    with pytest.raises(ValueError):
        model(**options)


@pytest.mark.parametrize("n", [0, -1, True, 1.2, 2])
def test_unsupported_suggest_requests(n):
    with pytest.raises((ValueError, NotImplementedError)):
        model().suggest(n)


def test_cold_start_constant_y_and_bounded_exhaustion(monkeypatch):
    opt = model(n_candidates=2)
    assert opt.suggest().shape == (1, 1) and not opt.backend.calls
    assert opt.events[-1]["reason"] == "empty_context"
    opt.observe([[.25], [.75]], [3, 3])
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.array([[.25], [.75]]))
    monkeypatch.setattr(opt, "_uniform", lambda n: np.full((n, 1), .25))
    np.testing.assert_equal(opt.suggest(), [[.5]])
    assert opt.events[-1]["method"] == "deterministic_gap" and not opt.backend.calls
    blocked = model(n_candidates=2, duplicate_atol=.6)
    blocked.observe([[.25], [.75]], [3, 3])
    monkeypatch.setattr(blocked, "_candidate_pool", lambda: np.array([[.25], [.75]]))
    with pytest.raises(RuntimeError, match="64 attempts"):
        blocked.suggest()


def test_two_equal_observations_still_use_backend(monkeypatch):
    opt = model(n_candidates=1)
    opt.observe([[.1]], [3])
    monkeypatch.setattr(opt, "_candidate_pool", lambda: np.array([[.5]]))
    opt.suggest()
    assert opt.backend.calls[-1][1].tolist() == [-3]
    opt.observe([[.9]], [3])
    opt.suggest()
    assert opt.backend.calls[-1][1].tolist() == [-3, -3]
    assert opt.events[-1]["status"] == "scored"


@pytest.mark.parametrize("strategy", ["uniform", "sobol"])
def test_seed_reproducibility_reset_and_history_isolation(strategy):
    def trajectory(seed):
        opt = PFNOptimizer([[0, 1]], seed=seed, backend=FakeBackend(), n_candidates=3, candidate_strategy=strategy)
        for _ in range(5):
            x = opt.suggest()
            assert np.isfinite(x).all() and (0 <= x).all() and (x <= 1).all()
            opt.observe(x, x[:, 0]**2)
        return opt
    first, same, other = trajectory(11), trajectory(11), trajectory(12)
    np.testing.assert_equal(first.history.arrays[0], same.history.arrays[0])
    assert not np.array_equal(first.history.arrays[0], other.history.arrays[0])
    backend = first.backend
    first.reset()
    assert first.backend is backend and not first.history.y and first.context_X.shape == (0, 1)
    assert not first.events and first.inference_seconds == 0 and not backend.calls
    with pytest.raises(RuntimeError, match="meta_train is disabled"):
        first.meta_train([object()])


def test_observation_validation_is_atomic():
    opt = model()
    for X, y in [([[.1]], [np.nan]), ([[1.1]], [1]), ([[.1]], [1, 2])]:
        with pytest.raises(ValueError):
            opt.observe(X, y)
        assert not opt.history.y and not len(opt.context_y)


@pytest.mark.parametrize("y", [[5], [3, 3], [-2, 0, 2]])
def test_observed_standardization_constant_and_negative_y(y):
    transformed, info = standardize_observed_y(y)
    assert np.isfinite(transformed).all() and abs(transformed.mean()) < 1e-12
    if len(set(y)) == 1:
        np.testing.assert_equal(transformed, np.zeros(len(y)))
        assert info["scale"] == 1
    else:
        assert transformed.std(ddof=0) == pytest.approx(1)


def test_missing_or_untrusted_checkpoint_never_unpickles(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        checkpoint_metadata(tmp_path / "missing.pt")
    untrusted = tmp_path / "untrusted.pt"
    untrusted.write_bytes(b"this is not a trusted model")
    with pytest.raises(ValueError, match="refusing to unpickle"):
        checkpoint_metadata(untrusted)


def test_missing_torch_fails_explicitly(monkeypatch):
    monkeypatch.setattr(pfn_backend, "checkpoint_metadata", lambda *a: {"sha256": "UNIT_TEST_ONLY"})
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("isolated missing torch test")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="dependencies unavailable"):
        pfn_backend.PFNs4BOBackend()


def test_import_has_no_torch_or_download_side_effect():
    code = "import sys; import models.pfn, models.pfn_backend; assert 'torch' not in sys.modules; assert 'pfns4bo' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
