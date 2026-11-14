# MetaGameBO_review

Reproduction of *MetaGameBO* (hierarchical game-theoretic meta-learning for BO):
the authors' code and a re-implementation, compared with the paper's baselines on
Forrester (Table 1) and HPOBench / NAS-Bench-201 / HPO-B (Figure 3).

## Project layout

```
models/                     one sub-package per optimizer, with its own components
  base.py                   BaseOptimizer (box or discrete pool), History, Task, cvar
  random_search/            Random search
  gp/                       GP + Expected Improvement
  lfbo/                     LFBO + weighting.py (Eq. 3 labels/weights, shared by MALIBO/MetaGameBO)
  pfn/                      PFNs4BO wrapper; PFNs4BO/ = authors' code + pre-trained weights
  ablr/                     ABLR (meta-learned features + Bayesian linear regression)
  rgpe/                     RGPE (ranking-weighted GP ensemble)
  malibo/                   meta_net.py, training.py (meta-training loop), adapters.py, model.py
  metagamebo_remake/        game.py (CVaR leader, k-DPP follower), kdpp.py, model.py
  metagamebo_original/      model.py (wrapper); author/ = github.com/Allen0497/MetaGameBO, unchanged
benchmarks/                 forrester.py, hpobench.py (FCNet), nasbench201.py, hpob.py, common.py
  data/                     downloaded data (not in git, see below)
experiments/                analyze_<dataset>.py + common.py (parallel runner, tables, plots)
visualizer/                 plotting style and extra figures (data, training, distributions)
results/                    <dataset>_benchmark.md, <dataset>_plot.png, raw/ (per-job JSON cache)
```

`models.build(name, bounds, seed)` creates any method by its display name
(`models.METHODS`). MetaBO is not included.

## Running

```
python -m experiments.analyze_forrester        # Table 1   -> results/forrester_*
python -m experiments.analyze_hpobench         # Figure 3  -> results/hpobench_*
python -m experiments.analyze_nasbench201      #           -> results/nasbench201_*
python -m experiments.analyze_hpob             #           -> results/hpob_*
```

Every job is cached in `results/raw/<dataset>/`; an interrupted run resumes, and
`--report-only` rebuilds the table and figure. Useful options: `--seeds`,
`--methods`, `--groups` (subset of targets / search spaces), `--budget`, `--workers`.

## Metrics

**Forrester (Table 1)** reports a classifier loss, as the authors confirmed: per
meta task `task_loss = sum_i w_i BCE(p_i, z_i) / n` on a random 20% of the pooled
reweighted meta-training samples (split by sample), `Average = mean over tasks`,
`CVaR = mean of the max(1, int(M (1 - 0.8)))` largest task losses (author
`meta_learning/trainer.py`, lines 412-432). It is not a regret and can exceed 1.
Methods without meta-training (Random, GP, LFBO, PFN) are fitted per task on the
task's training points and evaluated with their probability `P(f(x) < tau)`.
Setting: 200 tasks x 32 points, 100 epochs, batch 10 (author `run_forrester.py`).

**AutoML benchmarks (Figure 3)** report normalised simple regret over 500 steps
(mean ± s.e. over target tasks x seeds), plus CVaR0.8 of the final regret.

## Data

`benchmarks/data/` (≈ 4.3 GB):
- `fcnet/`, `nasbench201/`: Syne Tune blackbox repository
  (huggingface.co/datasets/synetune/blackbox-repository); the original HPOBench /
  NAS-Bench-201 download links are dead.
- `hpob-data/`, `saved-surrogates/`: github.com/sebastianpinedaar/hpo-data
  (HPO-B v3); `hpob_cache/` is built by `benchmarks.hpob.prepare()`.
