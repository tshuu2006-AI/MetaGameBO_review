import numpy as np

from benchmarks import ForresterBenchmark
from benchmarks.forrester import forrester


def test_canonical_forrester_formula():
    x = np.array([0.0, 0.5, 1.0])
    expected = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    np.testing.assert_allclose(forrester(x), expected)


def test_interfaces_return_requested_data():
    benchmark = ForresterBenchmark(4, seed=1)
    tasks = benchmark.get_meta_train_tasks(3, 17)
    assert len(tasks) == 3
    assert all(t.X.shape[0] == 17 and t.y.shape[0] == 17 for t in tasks)
    assert callable(benchmark.get_target_task().objective)


def test_source_family_matches_formula_and_separates_target_stream():
    benchmark = ForresterBenchmark(3, seed=7)
    tasks = benchmark.get_meta_train_tasks(3, 16)
    for task, parameters in zip(tasks, benchmark.parameters):
        expected = parameters.a * forrester(task.X[:, 0]) + parameters.b * (task.X[:, 0] - .5) - parameters.c
        np.testing.assert_allclose(task.y, expected)
    assert benchmark.target_parameters(0) not in benchmark.parameters
    X = np.linspace(0, 1, 12)[:, None]
    target = benchmark.get_target_task(evaluation_seed=4)
    np.testing.assert_array_equal(target.objective(X), target.objective(X))
    other = ForresterBenchmark(8, seed=7).get_target_task(evaluation_seed=4)
    np.testing.assert_array_equal(target.objective(X), other.objective(X))
