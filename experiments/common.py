"""Shared machinery of the ``analyze_<dataset>.py`` experiments.

* ``run_jobs``: run (method, group, seed) jobs in parallel; each job meta-trains
  the method once on the group's meta tasks, then optimises every target problem
  of the group. Results are cached as JSON in ``results/raw/<dataset>/`` so an
  interrupted run resumes. PFN jobs run in a small separate pool on the GPU.
* ``regret_table`` / ``plot_benchmark``: the ``<dataset>_benchmark.md`` table and
  the ``<dataset>_plot.png`` figure (regret curves + meta-training curves).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRAINED = {"MALIBO", "MetaGameBO (original)", "MetaGameBO (remake)", "ABLR"}


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def train_curves(opt) -> Optional[dict]:
    """Meta-training curves of an optimizer in a common format (x = progress in [0, 1])."""
    log = getattr(opt, "log", None)
    if log is None:
        return None
    if isinstance(log, dict) and "epoch" in log:  # author code, every 5 epochs
        n = max(log["epoch"]) if log["epoch"] else 1
        return dict(x=[e / n for e in log["epoch"]], train_avg=log["train_loss"], train_cvar=log["train_cvar"],
                    val_avg=log["val_loss"], val_cvar=log["val_cvar"], final=log["final"])
    if isinstance(log, dict) and "neg_log_evidence" in log:  # ABLR
        n = max(log["iteration"]) or 1
        return dict(x=[i / n for i in log["iteration"]], neg_log_evidence=log["neg_log_evidence"])
    d = log.__dict__
    n = max(d["iteration"]) or 1
    return dict(x=[i / n for i in d["iteration"]], train_avg=d["avg_task_loss"], train_cvar=d["cvar_task_loss"],
                val_avg=d["val_avg_task_loss"], val_cvar=d["val_cvar_task_loss"],
                final={k: v for k, v in d["final"].items() if k != "val_task_losses"})


def _run_job(job):
    """Worker: one (method, group, seed) job; see ``run_jobs``."""
    out_path, method, group_fn, group_key, seed, budget, n_init, kw = job
    from models import build
    t0 = time.time()
    try:
        problems = group_fn(group_key, seed)
        P0 = problems[0]
        opt = build(method, P0.bounds, seed, **kw.get(method, {}))
        opt.set_pool(P0.pool)
        if opt.is_meta:
            opt.meta_train(P0.meta_tasks)
        t_train = time.time() - t0
        runs = []
        for P in problems:
            opt.set_pool(P.pool)
            k = 0 if (opt.is_meta or P.init is not None) else n_init
            h = opt.optimize(P.objective, budget, n_init=k, init=P.init)
            runs.append(dict(problem=P.name, regret=P.regret(np.asarray(h.y)).tolist()))
        res = dict(method=method, group=str(group_key), seed=seed, budget=budget, runs=runs, train=train_curves(opt),
                   train_time=t_train, time=time.time() - t0)
    except Exception as e:  # keep the other jobs running
        res = dict(method=method, group=str(group_key), seed=seed, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc())
    Path(out_path).write_text(json.dumps(res))
    return method, group_key, seed, res.get("error"), time.time() - t0


def run_jobs(dataset: str, methods: Sequence[str], groups: Sequence, seeds: Sequence[int],
             group_fn: Callable, budget: int, n_init: int = 5, workers: int = 8, gpu_workers: int = 2,
             method_kwargs: Optional[Dict[str, dict]] = None, rerun: bool = False) -> List[dict]:
    """Run every (method, group, seed) job not already cached; return all cached results.

    Args:
        dataset: Name used for ``results/raw/<dataset>``.
        methods: Display names (see ``models.METHODS``).
        groups: Keys of problem groups sharing meta tasks (e.g. target index, search space).
        seeds: Seeds.
        group_fn: ``group_fn(group_key, seed) -> list[Problem]`` (picklable, module-level).
        budget: Evaluations per target problem (initial points included).
        n_init: Random initial points of non-meta methods when a problem has no fixed init.
        workers: CPU worker processes.
        gpu_workers: Worker processes for PFN (GPU).
        method_kwargs: Extra constructor arguments per method.
        rerun: Ignore the cache.
    """
    raw = RESULTS / "raw" / dataset
    raw.mkdir(parents=True, exist_ok=True)
    kw = method_kwargs or {}
    jobs_cpu, jobs_gpu = [], []
    for m in methods:
        for g in groups:
            for s in seeds:
                path = raw / f"{slug(m)}__{slug(str(g))}__s{s}__b{budget}.json"
                if path.exists() and not rerun and "error" not in json.loads(path.read_text()):
                    continue
                (jobs_gpu if m == "PFN" else jobs_cpu).append((str(path), m, group_fn, g, s, budget, n_init, kw))
    jobs_cpu.sort(key=lambda j: j[1] not in TRAINED)  # long (meta-trained) jobs first
    total = len(jobs_cpu) + len(jobs_gpu)
    print(f"[{dataset}] {total} jobs to run ({len(jobs_gpu)} on GPU), results in {raw}", flush=True)
    if total:
        ctx = get_context("spawn")
        done, t0 = 0, time.time()
        pools = []
        if jobs_cpu:
            pools.append(ctx.Pool(workers, initializer=_init_worker).imap_unordered(_run_job, jobs_cpu))
        if jobs_gpu:
            pools.append(ctx.Pool(gpu_workers, initializer=_init_worker).imap_unordered(_run_job, jobs_gpu))
        import itertools
        for m, g, s, err, dt in itertools.chain(*pools) if len(pools) == 1 else _interleave(pools):
            done += 1
            status = f"ERROR {err}" if err else f"{dt:.0f}s"
            print(f"[{dataset}] {done}/{total} {m} | {g} | seed {s}: {status} "
                  f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    return load_results(dataset, budget)


def _interleave(iters):
    """Consume several result iterators (from different pools) concurrently."""
    import queue
    import threading
    q = queue.Queue()

    def pump(it):
        for r in it:
            q.put(r)
        q.put(None)
    for it in iters:
        threading.Thread(target=pump, args=(it,), daemon=True).start()
    left = len(iters)
    while left:
        r = q.get()
        if r is None:
            left -= 1
        else:
            yield r


def load_results(dataset: str, budget: Optional[int] = None) -> List[dict]:
    """Cached job results of ``dataset`` (for ``budget`` if given; failed jobs are reported and skipped)."""
    out = []
    pattern = f"*__b{budget}.json" if budget else "*.json"
    for p in sorted((RESULTS / "raw" / dataset).glob(pattern)):
        r = json.loads(p.read_text())
        if "error" in r:
            print(f"  skipped failed job {p.name}: {r['error']}")
        else:
            out.append(r)
    return out


def regret_matrix(results: List[dict], method: str, budget: int) -> np.ndarray:
    """Regret curves of ``method`` over all (problem, seed) runs, shape (runs, budget)."""
    curves = [np.asarray(r["regret"], float) for res in results if res["method"] == method for r in res["runs"]]
    if not curves:
        return np.zeros((0, budget))
    return np.stack([np.pad(c[:budget], (0, max(0, budget - len(c))), mode="edge") for c in curves])


def mean_se(x: np.ndarray, axis=0):
    x = np.asarray(x, float)
    n = x.shape[axis]
    return x.mean(axis), (x.std(axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(x.mean(axis)))


def regret_table(results: List[dict], methods: Sequence[str], budget: int, steps: Sequence[int],
                 title: str, notes: str = "") -> str:
    """Markdown table: mean normalised regret +- s.e. at ``steps`` and CVaR0.8 of the final regret."""
    from models.base import cvar
    head = "| Method | runs | " + " | ".join(f"step {s}" for s in steps) + " | CVaR0.8 (final) |"
    rows = [f"## {title}", "", "Normalised simple regret (lower is better), mean ± standard error over "
            "(target task × seed) runs. CVaR0.8 = mean of the worst 20% final regrets.", "",
            head, "|" + "---|" * (len(steps) + 3)]
    best = {}
    stats = {}
    for m in methods:
        R = regret_matrix(results, m, budget)
        if len(R) == 0:
            continue
        stats[m] = (len(R), [mean_se(R[:, s - 1]) for s in steps], cvar(R[:, -1]))
    for j in range(len(steps)):
        best[j] = min(v[1][j][0] for v in stats.values()) if stats else None
    for m, (n, ms, cv) in stats.items():
        cells = []
        for j, (mu, se) in enumerate(ms):
            c = f"{mu:.3e} ± {se:.1e}"
            cells.append(f"**{c}**" if mu == best[j] else c)
        rows.append(f"| {m} | {n} | " + " | ".join(cells) + f" | {cv:.3e} |")
    missing = [m for m in methods if m not in stats]
    if missing:
        rows += ["", f"No results for: {', '.join(missing)}."]
    if notes:
        rows += ["", notes]
    return "\n".join(rows) + "\n"


def _avg_curves(results, method, key):
    """Average a training curve over jobs (interpolated on a common progress grid)."""
    xs = np.linspace(0, 1, 101)
    ys = [np.interp(xs, r["train"]["x"], r["train"][key]) for r in results
          if r["method"] == method and r.get("train") and r["train"].get(key)]
    return (xs, *mean_se(np.stack(ys))) if ys else None


def plot_training(axes, results: List[dict], methods: Sequence[str]) -> None:
    """Meta-training panels: held-out avg task loss, held-out CVaR, ABLR -log evidence."""
    from visualizer.style import method_style
    keys = [("val_avg", "Held-out avg task loss (weighted BCE)"),
            ("val_cvar", "Held-out CVaR0.8 of task loss"),
            ("neg_log_evidence", "ABLR: −log evidence / point")]
    for ax, (key, title) in zip(axes, keys):
        for m in methods:
            c = _avg_curves(results, m, key)
            if c is None:
                continue
            x, mu, se = c
            st = method_style(m)
            ax.plot(x, mu, label=m, color=st["color"], ls=st["ls"], lw=st.get("lw", 1.6))
            ax.fill_between(x, mu - se, mu + se, color=st["color"], alpha=0.15, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training progress (fraction of updates)")
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=7)
        else:
            ax.set_visible(False)


def plot_benchmark(results: List[dict], methods: Sequence[str], budget: int, title: str, path: Path) -> None:
    """``<dataset>_plot.png``: regret curves (log scale, mean ± s.e.) + meta-training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualizer.style import method_style
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    ax = axes[0]
    x = np.arange(1, budget + 1)
    for m in methods:
        R = regret_matrix(results, m, budget)
        if len(R) == 0:
            continue
        mu, se = mean_se(R)
        st = method_style(m)
        lo = np.maximum(mu - se, mu[mu > 0].min() * 0.5 if np.any(mu > 0) else 1e-12)
        ax.plot(x, mu, label=m, color=st["color"], ls=st["ls"], lw=st.get("lw", 1.8))
        ax.fill_between(x, lo, mu + se, color=st["color"], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Normalized regret (log)")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    plot_training(axes[1:], results, [m for m in methods if m in TRAINED])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


BENCH_METHODS = ["ABLR", "Random", "GP", "LFBO", "RGPE", "PFN", "MALIBO",
                 "MetaGameBO (original)", "MetaGameBO (remake)"]
# meta-training lengths on the AutoML benchmarks (author's run_hpob.py: 500 epochs)
BENCH_KWARGS = {"MALIBO": dict(iters=2000), "MetaGameBO (remake)": dict(iters=2000),
                "MetaGameBO (original)": dict(epochs=500), "ABLR": dict(iters=1000)}


def bench_main(dataset: str, title: str, groups: Sequence, group_fn: Callable, notes: str,
               default_seeds: int = 5, budget: int = 500) -> None:
    """Command line of ``analyze_<dataset>.py``: run jobs, write the table and the figure."""
    import argparse
    ap = argparse.ArgumentParser(description=f"{title} benchmark")
    ap.add_argument("--seeds", type=int, default=default_seeds)
    ap.add_argument("--budget", type=int, default=budget)
    ap.add_argument("--methods", nargs="+", default=BENCH_METHODS)
    ap.add_argument("--groups", nargs="+", default=None, help="subset of target groups")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gpu-workers", type=int, default=2)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="only rebuild table/plot from the cache")
    a = ap.parse_args()
    groups = [g for g in groups if a.groups is None or str(g) in a.groups]
    if not a.report_only:
        run_jobs(dataset, a.methods, groups, range(a.seeds), group_fn, a.budget, workers=a.workers,
                 gpu_workers=a.gpu_workers, method_kwargs=BENCH_KWARGS, rerun=a.rerun)
    results = [r for r in load_results(dataset, a.budget) if r["method"] in a.methods]
    steps = [s for s in (1, 10, 50, 100, 200, 300, 500) if s <= a.budget]
    md = regret_table(results, a.methods, a.budget, steps, f"{title} benchmark", notes)
    (RESULTS / f"{dataset}_benchmark.md").write_text(md)
    plot_benchmark(results, a.methods, a.budget, title, RESULTS / f"{dataset}_plot.png")
    print(md)
    print("wrote", RESULTS / f"{dataset}_benchmark.md", "and", RESULTS / f"{dataset}_plot.png")
