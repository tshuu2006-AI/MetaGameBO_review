"""Validation at the common interface, independent of future ML models."""
import numpy as np
import pytest

from models.base_model import History
from models.random_search import RandomSearch


@pytest.mark.parametrize("bounds", [[], [0, 1], [[0, 0]], [[1, 0]], [[0, np.inf]], [[0, np.nan]], [[0, 1, 2]]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        RandomSearch(bounds)


@pytest.mark.parametrize("budget,n_init", [(0, 0), (-1, 0), (2, 3), (2, -1), (True, 0), (2.1, 0), (2, 0.5)])
def test_invalid_budget_rejected_before_objective(budget, n_init):
    model = RandomSearch([[0, 1]])
    with pytest.raises(ValueError):
        model.optimize(lambda X: pytest.fail("Objective must not be called"), budget, n_init)


def test_history_no_truncation_or_aliasing():
    history = History()
    with pytest.raises(ValueError):
        history.add([[0.1], [0.2]], [1])
    assert not history.X and not history.y
    X, y = np.array([[0.1], [0.2]]), np.array([[2.0], [1.0]])
    history.add(X, y)
    X[0, 0] = 99
    y[0, 0] = 99
    np.testing.assert_equal(history.arrays[0], [[0.1], [0.2]])
    np.testing.assert_equal(history.arrays[1], [2, 1])
    with pytest.raises(ValueError):
        history.add([[0.1, 0.2]], [1])


@pytest.mark.parametrize("X,y", [([], []), ([[1]], [np.nan]), ([[np.inf]], [1]), ([[1]], [[1, 2]]), ([1], [1])])
def test_history_invalid_values(X, y):
    with pytest.raises(ValueError):
        History().add(X, y)


def test_best_budget_phase_and_reset_hook():
    class CachedRandom(RandomSearch):
        def _update(self, X, y):
            self.cached_X = X.copy()

        def _reset_task_state(self):
            self.cached_X = None

    model = CachedRandom([[0, 1]], seed=44)
    with pytest.raises(ValueError, match="No observations"):
        _ = model.best
    history = model.optimize(lambda X: X**2, budget=8, n_init=2)
    assert len(history.y) == 8
    assert model.phases == ["initial"] * 2 + ["sequential"] * 6
    assert model.best[1] == min(history.y)
    model.reset()
    assert not model.history.y and model.cached_X is None


def test_suggest_validation_before_objective():
    class Invalid(RandomSearch):
        def suggest(self, n=1):
            return np.array([[2.0]])
    with pytest.raises(ValueError, match="outside bounds"):
        Invalid([[0, 1]]).optimize(lambda X: pytest.fail("invalid X reached objective"), 1)


def test_random_uniform_reproducible_bounds_shape_and_stream_semantics():
    first, second = RandomSearch([[-2, 3], [4, 5]], seed=44), RandomSearch([[-2, 3], [4, 5]], seed=44)
    X = first.suggest(10000)
    np.testing.assert_array_equal(X, second.suggest(10000))
    assert X.shape == (10000, 2)
    assert np.all((X >= [-2, 4]) & (X <= [3, 5]))
    np.testing.assert_allclose(np.mean(X, axis=0), [0.5, 4.5], atol=0.06)
    # Reset retains the RNG stream; a newly seeded object restarts it.
    first.reset()
    np.testing.assert_array_equal(first.suggest(3), second.suggest(3))
    for n in [0, -1, 1.2, True]:
        with pytest.raises(ValueError):
            first.suggest(n)
