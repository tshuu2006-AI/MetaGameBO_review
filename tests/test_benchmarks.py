import numpy as np

from benchmarks import ForresterBenchmark, MockBenchmark
from benchmarks.forrester import forrester


def test_canonical_forrester_formula():
    x = np.array([0.0, 0.5, 1.0])
    expected = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    np.testing.assert_allclose(forrester(x), expected)


def test_interfaces_return_requested_data():
    for benchmark in (ForresterBenchmark(4, seed=1), MockBenchmark(seed=1)):
        tasks = benchmark.get_meta_train_tasks(3, 17)
        assert len(tasks) == 3
        assert all(t.X.shape[0] == 17 and t.y.shape[0] == 17 for t in tasks)
        assert callable(benchmark.get_target_task().objective)
