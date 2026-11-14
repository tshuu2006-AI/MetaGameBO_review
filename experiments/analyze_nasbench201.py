"""NAS-Bench-201 benchmark, paper Figure 3 (middle).

Leave-one-task-out over CIFAR-10, CIFAR-100 and ImageNet16-120 (512 random
cells per source task as meta data), 500 BO steps on all 15,625 cells;
normalised regret of the validation error.

Outputs: ``results/nasbench201_benchmark.md``, ``results/nasbench201_plot.png``.

    python -m experiments.analyze_nasbench201
"""
from benchmarks import nasbench201
from experiments.common import bench_main


def group(key, seed):
    return [nasbench201.problems(seed)[key]]


NOTES = ("Setting: leave-one-task-out over cifar10, cifar100, ImageNet16-120; 512 meta cells per "
         "source task; cells encoded as 6 operation indices (PFN accepts at most 18 inputs).")

if __name__ == "__main__":
    bench_main("nasbench201", "NASBench201", list(range(len(nasbench201.TASKS))), group, NOTES)
