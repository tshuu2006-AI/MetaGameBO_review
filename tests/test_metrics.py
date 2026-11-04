"""Manual metric examples and numerical oracle checks on multimodal functions."""
import numpy as np
import pytest

from benchmarks.forrester import ForresterTask, generate_target_tasks
from experiments.metrics import (aggregate_runs, best_so_far, empirical_cvar, estimate_global_minimum_1d,
                                 estimate_reference_range, normalized_regret, simple_regret)


def test_hand_calculated_regrets_and_statistics():
    np.testing.assert_array_equal(best_so_far([3, 2, 4, -1]), [3, 2, 2, -1])
    np.testing.assert_allclose(simple_regret([2, 0, -1], -1), [3, 1, 0])
    assert simple_regret(-1-1e-10, -1) == 0
    np.testing.assert_allclose(normalized_regret([2, 0, -1], -1, 3), [0.75, 0.25, 0])
    assert aggregate_runs([1, 2, 3]) == {"n": 3, "mean": 2, "sample_std": 1.0, "standard_error": 1/np.sqrt(3)}
    assert aggregate_runs([7]) == {"n": 1, "mean": 7, "sample_std": None, "standard_error": None}


@pytest.mark.parametrize("values,alpha,expected", [([1, 2, 3, 4, 5], 0.8, 5),
    ([1, 2, 3, 4, 5], 0.7, (5+0.5*4)/1.5), ([1, 2, 3, 4], 0.625, (4+0.5*3)/1.5),
    ([2, 2, 2], 0.8, 2), ([1, 4, 4], 0.5, 4), ([9], 0.8, 9), ([-5, -3, -1], 0.8, -1)])
def test_fractional_tail_cvar(values, alpha, expected):
    assert empirical_cvar(values, alpha) == pytest.approx(expected)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_metric_vectors(values):
    for function in (best_so_far, aggregate_runs, empirical_cvar):
        with pytest.raises(ValueError):
            function(values)


@pytest.mark.parametrize("alpha", [0, 1, -1, 1.1, np.nan, np.inf])
def test_invalid_cvar_alpha(alpha):
    with pytest.raises(ValueError):
        empirical_cvar([1, 2], alpha)


def test_invalid_regret_references_and_negative_values():
    for low, high in [(1, 1), (2, 1), (np.nan, 1), (0, np.inf)]:
        with pytest.raises(ValueError):
            normalized_regret(1, low, high)
    with pytest.raises(ValueError, match="below"):
        simple_regret(-2, -1)
    with pytest.raises(ValueError, match="exceeds"):
        normalized_regret(2, 0, 1)
    for value in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            simple_regret(value, 0)
    for value in [[], [[1, 2]]]:
        with pytest.raises(ValueError):
            simple_regret(value, 0)


@pytest.mark.parametrize("task", generate_target_tasks(6, 44) + [ForresterTask("standard", 1, 0, 0)])
def test_oracle_multiple_basins_matches_dense_grid(task):
    grid_values = task.evaluate(np.linspace(0, 1, 200001)[:, None])
    oracle = estimate_reference_range(task, grid_size=4097)
    assert oracle["reference_min"] <= grid_values.min()+1e-8
    assert oracle["reference_max"] >= grid_values.max()-1e-8
    assert abs(oracle["reference_min"]-grid_values.min()) < 1e-5
    assert abs(oracle["reference_max"]-grid_values.max()) < 1e-5
    assert oracle["exact_global_optimum_guaranteed"] is False


def test_oracle_known_standard_minimum_and_endpoints():
    task = ForresterTask("standard", 1, 0, 0)
    estimate = estimate_global_minimum_1d(task)
    assert estimate["x"] == pytest.approx(0.75724876, abs=2e-7)
    assert estimate["value"] == pytest.approx(-6.020740, abs=1e-5)
    # Restrict to a monotone subinterval: the minimum is the left endpoint.
    estimate = estimate_global_minimum_1d(task, bounds=(0.9, 1.0))
    assert estimate["x"] == pytest.approx(0.9)


@pytest.mark.parametrize("grid_size", [0, 256, 257.5, True])
def test_oracle_rejects_insufficient_or_invalid_grid(grid_size):
    with pytest.raises(ValueError):
        estimate_global_minimum_1d(ForresterTask("x", 1, 0, 0), grid_size=grid_size)
