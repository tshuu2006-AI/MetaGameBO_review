"""HPOBench (FCNet) benchmark, paper Figure 3 (left).

Leave-one-task-out over the 4 UCI datasets: each is the target once, the other
3 give 512 random configurations each as meta-training data. 500 BO steps on
the full table of 62,208 configurations; normalised regret
``(best y - min) / (max - min)`` of the validation MSE. Non-meta methods start
from 5 random points; meta-learned methods start from their meta-learned prior.

Outputs: ``results/hpobench_benchmark.md``, ``results/hpobench_plot.png``.

    python -m experiments.analyze_hpobench                # 5 seeds x 4 targets
    python -m experiments.analyze_hpobench --report-only  # rebuild from the cache
"""
from benchmarks import hpobench
from experiments.common import bench_main


def group(key, seed):
    return [hpobench.problems(seed)[key]]


NOTES = ("Setting: leave-one-task-out over protein, naval, parkinsons, slice; 512 meta points per "
         "source task; regret normalised by the min/max of the full table (diverged configurations "
         "make the max very large, hence the tiny regrets, as in the paper's figure).")

if __name__ == "__main__":
    bench_main("hpobench", "HPOBench", list(range(len(hpobench.TASKS))), group, NOTES)
