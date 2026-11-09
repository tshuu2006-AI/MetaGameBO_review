import numpy as np

from benchmarks.forrester import ForresterTask, exact_task_bounds, forrester_function, sample_task


def test_forrester_is_finite_and_vectorized():
    task = ForresterTask(a=1.0, b=0.0, c=0.0)
    xs = np.linspace(0.0, 1.0, 20)
    ys = forrester_function(xs, task)
    assert ys.shape == xs.shape
    assert np.isfinite(ys).all()


def test_sampled_task_is_inside_official_ranges():
    task = sample_task(42)
    assert 0.2 <= task.a <= 3.0
    assert -5.0 <= task.b <= 15.0
    assert -5.0 <= task.c <= 5.0


def test_exact_bounds_are_ordered():
    low, high = exact_task_bounds(sample_task(7), grid_size=1001)
    assert np.isfinite(low) and np.isfinite(high)
    assert low < high
