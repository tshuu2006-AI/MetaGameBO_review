"""Forrester benchmark: paper Table 1 (classifier loss) and meta-training curves.

Metric, as stated by the authors (email) and computed in their
``meta_learning/trainer.py``: for every meta-training task,
``task_loss = sum_i w_i BCE(p_i, z_i) / n`` on the held-out samples (a random
20% of the pooled reweighted samples, split by sample, Eq. 3 labels/weights of
the LFBO/author code); "Average Loss" is the mean over tasks and "CVaR" the
mean of the ``max(1, int(M (1 - 0.8)))`` largest task losses. Lower is better;
it is a classification loss, not a regret, and it is unbounded above.

Setting (author's ``run_forrester.py``): 200 meta tasks x 32 points, 100 epochs,
batch 10 for the author code (= 2000 steps of our loop for MALIBO/remake).

How each method gets a probability ``p = P(f(x) < tau)`` for the held-out points:
* MALIBO, MetaGameBO (remake): the meta-classifier with the task's embedding.
* MetaGameBO (original): the author's trainer reports its own held-out losses
  (its own random split of the same kind).
* Random, GP, LFBO, PFN (no meta-training): fitted per task on that task's
  training points only (points with a held-out copy are excluded):
  Random = constant P(good) of the context, GP = Phi((tau - mu) / sigma),
  LFBO = its weighted GB classifier, PFN = its posterior predictive.

Outputs: ``results/forrester_benchmark.md`` and ``results/forrester_plot.png``;
per-job JSON in ``results/raw/forrester``.

    python -m experiments.analyze_forrester                 # 10 seeds
    python -m experiments.analyze_forrester --seeds 100     # as in the paper
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.common import RESULTS, _init_worker, _avg_curves, mean_se, slug, train_curves

METHODS = ["Random", "GP", "LFBO", "PFN", "MALIBO", "MetaGameBO (original)", "MetaGameBO (remake)"]
META = {"MALIBO", "MetaGameBO (original)", "MetaGameBO (remake)"}
N_TASKS, N_PER_TASK, ALPHA = 200, 32, 0.8


def task_losses_without_meta(method: str, meta, seed: int) -> np.ndarray:
    """Held-out weighted BCE of every task for a method fitted per task (no meta-learning)."""
    from benchmarks.forrester import BOUNDS
    from models import build
    from models.lfbo import threshold
    from models.malibo import prepare_meta_data, weighted_bce
    md = prepare_meta_data(meta, 1 / 3, "lfbo", 0.2, seed)
    opt = build(method, BOUNDS, seed, **({"device": "cpu"} if method == "PFN" else {}))
    out = np.full(len(meta), np.nan)
    for i, t in enumerate(meta):
        rows = md.tid == i + 1
        val = rows & md.is_val
        if not val.any():
            continue
        held = np.unique(md.src[val])
        ctx = np.setdiff1d(np.arange(len(t.y)), held)
        p = opt.prob_good(t.X[ctx], t.y[ctx], md.X[val], threshold(t.y))
        out[i] = weighted_bce(p, md.lab[val], md.w[val]).mean()
    return out


def run_job(job):
    path, method, seed = job
    from benchmarks.forrester import BOUNDS, make_meta_tasks
    from models import build
    from models.base import cvar
    t0 = time.time()
    try:
        meta = make_meta_tasks(N_TASKS, N_PER_TASK, seed)
        res = dict(method=method, seed=seed)
        if method in META:
            opt = build(method, BOUNDS, seed)
            opt.meta_train(meta)
            cur = train_curves(opt)
            res.update(avg=cur["final"]["val_avg"], cvar=cur["final"]["val_cvar"], train=cur)
        else:
            L = task_losses_without_meta(method, meta, seed)
            res.update(avg=float(np.nanmean(L)), cvar=cvar(L, ALPHA), task_losses=L.tolist())
        res["time"] = time.time() - t0
    except Exception as e:
        res = dict(method=method, seed=seed, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    Path(path).write_text(json.dumps(res))
    return method, seed, res.get("error"), time.time() - t0


def table(results, methods) -> str:
    rows = ["## Forrester benchmark (paper Table 1)", "",
            f"Weighted BCE of the classifier on held-out samples of the {N_TASKS} meta-training tasks "
            f"({N_PER_TASK} points each); mean ± standard error over seeds. Lower is better.", "",
            "| Method | seeds | Average Loss ↓ | CVaR ↓ |", "|---|---|---|---|"]
    stats = {}
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        if rs:
            stats[m] = (len(rs), mean_se(np.array([r["avg"] for r in rs])), mean_se(np.array([r["cvar"] for r in rs])))
    if stats:
        b_avg = min(v[1][0] for v in stats.values())
        b_cv = min(v[2][0] for v in stats.values())
    for m, (n, (a, ase), (c, cse)) in stats.items():
        ca, cc = f"{a:.4f} ± {ase:.5f}", f"{c:.4f} ± {cse:.5f}"
        ca = f"**{ca}**" if a == b_avg else ca
        cc = f"**{cc}**" if c == b_cv else cc
        rows.append(f"| {m} | {n} | {ca} | {cc} |")
    if "MALIBO" in stats:
        rows += ["", "Relative to MALIBO (negative = lower loss):", ""]
        a0, c0 = stats["MALIBO"][1][0], stats["MALIBO"][2][0]
        for m, (_, (a, _), (c, _)) in stats.items():
            if m != "MALIBO":
                rows.append(f"- {m}: average loss {100 * (a / a0 - 1):+.1f}%, CVaR {100 * (c / c0 - 1):+.1f}%")
    rows += ["", "Notes:",
             "- Paper: MetaGameBO 0.0183 / 0.0934 (−95.7% average loss vs MALIBO 0.4277 / 0.8195).",
             "- All methods use the LFBO/author labelling of Eq. (3): every good point appears as class 1 "
             "(EI weight) and as class 0, so the loss has an irreducible floor > 0.",
             "- MetaGameBO (original) reports the author trainer's own held-out split; the other rows share one split per seed.",
             "- Random, GP, LFBO and PFN are not meta-trained: each task's classifier is fitted on that task's "
             "training points only (≈ 25 points)."]
    return "\n".join(rows) + "\n"


def plot(results, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualizer.style import method_style
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    keys = [("train_avg", "Train: avg task loss"), ("train_cvar", "Train: CVaR0.8 of task loss"),
            ("val_avg", "Held-out: avg task loss"), ("val_cvar", "Held-out: CVaR0.8 of task loss")]
    for ax, (k, title) in zip(axes, keys):
        for m in METHODS:
            c = _avg_curves(results, m, k) if m in META else None
            if c is None:
                continue
            x, mu, se = c
            st = method_style(m)
            ax.plot(x, mu, color=st["color"], ls=st["ls"], label=m)
            ax.fill_between(x, mu - se, mu + se, color=st["color"], alpha=0.15, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training progress (fraction of updates)")
        ax.set_ylabel("Weighted BCE")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Forrester: meta-training curves (mean ± s.e. over seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (paper: 100)")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    raw = RESULTS / "raw" / "forrester"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for m in a.methods:
        for s in range(a.seeds):
            p = raw / f"{slug(m)}__s{s}.json"
            if a.rerun or not p.exists() or "error" in json.loads(p.read_text()):
                jobs.append((str(p), m, s))
    jobs.sort(key=lambda j: j[1] not in META)
    print(f"[forrester] {len(jobs)} jobs to run", flush=True)
    if jobs:
        t0 = time.time()
        with get_context("spawn").Pool(a.workers, initializer=_init_worker) as pool:
            for i, (m, s, err, dt) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f"[forrester] {i}/{len(jobs)} {m} seed {s}: {'ERROR ' + err if err else f'{dt:.0f}s'} "
                      f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    results = []
    for p in sorted(raw.glob("*.json")):
        r = json.loads(p.read_text())
        if "error" in r:
            print("  skipped failed job", p.name, r["error"])
        elif r["method"] in a.methods:
            results.append(r)
    md = table(results, a.methods)
    (RESULTS / "forrester_benchmark.md").write_text(md)
    plot(results, RESULTS / "forrester_plot.png")
    print(md)
    print("wrote", RESULTS / "forrester_benchmark.md", "and", RESULTS / "forrester_plot.png")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
