"""HPO-B benchmark (continuous surrogates), paper Figure 3 (right).

For each of the 16 search spaces, every method is meta-trained once per seed on
the space's meta-train datasets (512 points each) and then run on each test
dataset from the 5 fixed HPO-B initial points of that seed (test0..test4).
500 BO steps on the XGBoost surrogates; regret
``(y_max - best accuracy) / (y_max - y_min)`` clipped to [0, 1].

Outputs: ``results/hpob_benchmark.md``, ``results/hpob_plot.png``.

    python -m experiments.analyze_hpob                         # all 16 spaces
    python -m experiments.analyze_hpob --groups 4796 5860      # a subset of spaces
"""
from benchmarks import hpob
from experiments.common import bench_main


def group(space, seed):
    return hpob.problems(space, hpob.SEEDS[seed])


NOTES = ("Setting: HPO-B v3 meta-train/test split, continuous search spaces with the official "
         "surrogates, 5 fixed initial points per seed; the 5 HPO-B seeds are test0..test4.")

if __name__ == "__main__":
    bench_main("hpob", "HPO-B", hpob.search_spaces(), group, NOTES)
