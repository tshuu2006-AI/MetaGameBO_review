"""HPO-B benchmark, paper Figure 3 (right).

For each of the 16 search spaces, every method is meta-trained once per seed on
the space's meta-train datasets (512 points each) and then run on each test
dataset from the 5 fixed HPO-B initial points of that seed (test0..test4).

Default: 500 evaluations on the continuous XGBoost surrogates; regret
``(y_max - best accuracy) / (y_max - y_min)`` clipped to [0, 1].
``--discrete``: BO chooses among the tabulated evaluations of the test dataset,
as in the MALIBO / MetaGameBO code (``run_hpob.py``); results ``hpob_discrete_*``.
The MALIBO paper shows 100 steps after the 5 initial points: ``--budget 105``
(our step count includes the initial points).

Outputs: ``results/hpob_benchmark.md``, ``results/hpob_plot.png``.

    python -m experiments.analyze_hpob                         # all 16 spaces
    python -m experiments.analyze_hpob --groups 4796 5860      # a subset of spaces
    python -m experiments.analyze_hpob --discrete --budget 105 # protocol of the official code
"""
from benchmarks import hpob
from experiments.common import bench_main


def group(space, seed, discrete=False):
    return hpob.problems(space, hpob.SEEDS[seed], discrete=discrete)


NOTES = ("Setting: HPO-B v3 meta-train/test split, continuous search spaces with the official "
         "surrogates, 5 fixed initial points per seed; the 5 HPO-B seeds are test0..test4.")

if __name__ == "__main__":
    bench_main("hpob", "HPO-B", hpob.search_spaces(), group, NOTES, discrete_ok=True)
