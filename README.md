# MetaGameBO_review

## Project layout

```
models/        Optimizers compared in the experiments (one per file)
  random_search.py, gp_ei.py, lfbo.py
  meta_classifier_bo.py   MALIBO / MetaGameBO re-implementation (chosen by MetaTrainConfig)
  metagamebo_upstream.py  upstream MetaGameBO code (needs MetaGameBO/ on sys.path)
components/    Building blocks used by the models
  base.py        BaseOptimizer, History, Task
  weighting.py   Eq. (3) labels / utility weights
  risk.py        cvar, kde_quantile
  kdpp.py        exact k-DPP + upstream approximate DiagonalKDPP
  meta_net.py    MetaNet and embedding regulariser
  meta_training.py  MetaTrainConfig, meta_train (Algorithm 1, optional snapshot callback)
  adapters.py    BLRAdapter (Laplace posterior), ThompsonInit
  acquisition.py, upstream.py
visualizer/    Plots: data, training process, model distributions, BO results
dashboard/     Web UI (stdlib only) + RunTracker for live training progress
benchmarks/    Forrester task family
experiments/   Run / analyse / visualise scripts
```

### Metrics (as clarified by the paper's authors)

Paper Table 1 reports a **classifier loss**, not BO regret. Per meta task,
`task_loss = sum_i w_i * BCE(p_i, z_i) / n_task` (Eq. 3 weights), measured on a
random 20% of the pooled meta-training samples held out from training (split by
sample, so the tasks are the same as in training). `CVaR = mean of the
k = max(1, int(n_tasks * (1 - alpha)))` largest task losses, alpha = 0.8
(upstream `meta_learning/trainer.py`, lines 412-432). It can exceed 1. Here it is
`paper_val_cvar` in the results JSON (`MetaTrainConfig.val_frac = 0.2`). BO
quality is reported separately as normalised regret on new target tasks. Upstream
defaults: 200 tasks x 32 samples, 100 epochs, batch 10, seed 44, 16 BO steps on
one target task. The original experiment records are not available.

Figures: `python -m experiments.visualize_forrester --iters 300 --out results_remake/figures`

### Dashboard

```
python -m dashboard.server            # open http://127.0.0.1:8765
```

Tabs: project overview (packages, modules, classes), results tables and regret
curves, figure gallery, and live training. Runs started from the "Huấn luyện
trực tiếp" tab, or from the terminal with `--track`, stream their progress
(per-task loss, CVaR, selected tasks, embeddings, k-DPP weights, BO regret):

```
python -m experiments.run_forrester_remake --methods MALIBO MetaGameBO --seeds 0 \
    --iters 300 --track --out runs/my_results.json
```

In your own code: `tracker = RunTracker("name")`, then
`meta_train(tasks, cfg, callback=tracker)`, `tracker.bo(...)`, `tracker.finish()`.
