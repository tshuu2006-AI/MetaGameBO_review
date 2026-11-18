"""Shared machinery of the ``analyze_<dataset>.py`` experiments.

* ``run_jobs``: run (method, group, seed) jobs in parallel; each job meta-trains
  the method once on the group's meta tasks, then optimises every target problem
  of the group. Results are cached as JSON in ``results/raw/<dataset>/`` so an
  interrupted run resumes; a running job also keeps a checkpoint next to it
  (``<job>.ckpt.pkl``: the optimizer after meta-training, the finished problems
  and, every ``METAGAMEBO_CKPT_SECONDS`` (300), the BO run in progress), which
  the next run continues from. ``--rerun`` ignores both.
  PFN jobs run in a small separate pool on the GPU.
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
# results folder; METAGAMEBO_RESULTS lets kaggle_run.py write straight into the Kaggle output
RESULTS = Path(os.environ.get("METAGAMEBO_RESULTS", ROOT / "results"))
CKPT_EVERY = float(os.environ.get("METAGAMEBO_CKPT_SECONDS", 300))  # seconds between BO checkpoints


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a file via a temporary file and a rename, so a killed process never leaves a half-written file.

    Args:
        path: Destination.
        data: File contents.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class _Checkpoint:
    """Checkpoint of one job in ``<job>.ckpt.pkl``: the optimizer, the finished runs and the run in progress.

    Args:
        out_path: Path of the job's result JSON; the checkpoint sits next to it.
    """

    def __init__(self, out_path: str):
        self.path = Path(out_path).with_suffix(".ckpt.pkl")
        self.last = time.time()  # time of the last save, for the periodic BO checkpoints

    def load(self) -> Optional[dict]:
        """Read the checkpoint.

        Returns:
            The saved state, or None if there is none or it cannot be read (the job starts over).
        """
        import pickle
        if not self.path.exists():
            return None
        try:
            return pickle.loads(self.path.read_bytes())
        except Exception:  # corrupt / incompatible checkpoint: start over
            return None

    def save(self, **state) -> None:
        """Pickle the job state atomically.

        Args:
            **state: ``opt``, ``runs``, ``train_time``, ``elapsed`` and ``current``
                (the problem in progress and its objective state, or None).
        """
        import pickle
        _atomic_write(self.path, pickle.dumps(state))
        self.last = time.time()

    def remove(self) -> None:
        """Delete the checkpoint (the job is finished)."""
        self.path.unlink(missing_ok=True)


def _objective_state(obj):
    """State of a target objective that must survive a restart (noise generator, values seen).

    Args:
        obj: The objective, e.g. a ``LookupTable``.

    Returns:
        Dict of the attributes ``rng`` and ``true_seen`` that the objective has.
    """
    return {k: getattr(obj, k) for k in ("rng", "true_seen") if hasattr(obj, k)}


# methods with a meta-training phase (training-curve panels, long jobs scheduled first)
TRAINED = {"MALIBO", "MetaGameBO (original)", "MetaGameBO (remake)", "ABLR"}


def _init_worker():
    """Worker initialiser: one thread per process and no warnings."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)


def slug(s: str) -> str:
    """File-name-safe version of a string.

    Args:
        s: Any string, e.g. ``"MetaGameBO (remake)"``.

    Returns:
        The string with every non-alphanumeric character replaced by ``_``.
    """
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def train_curves(opt) -> Optional[dict]:
    """Meta-training curves of an optimizer in a common format.

    Args:
        opt: A meta-trained optimizer (its ``log`` comes from our loop, the author code or ABLR).

    Returns:
        Dict with ``x`` (training progress in [0, 1]) and the curves, or None without a log.
    """
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
    """Run one (method, group, seed) job: meta-train once, then optimise every problem of the group.

    Resumes from the job's checkpoint if there is one; the result (or the error) is written as JSON.

    Args:
        job: ``(out_path, method, group_fn, group_key, seed, budget, n_init, method_kwargs)``.

    Returns:
        ``(method, group_key, seed, error or None, seconds)`` for the progress log.
    """
    out_path, method, group_fn, group_key, seed, budget, n_init, kw = job
    from models import build
    t0 = time.time()
    ck = _Checkpoint(out_path)
    try:
        problems = group_fn(group_key, seed)
        state = ck.load()
        if state is not None:  # resume: skip meta-training and the finished problems
            opt, runs, t_train, t_prev = state["opt"], state["runs"], state["train_time"], state["elapsed"]
            current = state.get("current")
        else:
            P0 = problems[0]
            opt = build(method, P0.bounds, seed, **kw.get(method, {}))
            opt.set_pool(P0.pool)
            if opt.is_meta:
                opt.meta_train(P0.meta_tasks)
            t_train, t_prev, runs, current = time.time() - t0, 0.0, [], None
            ck.save(opt=opt, runs=runs, train_time=t_train, elapsed=t_train, current=None)
        elapsed = lambda: t_prev + time.time() - t0
        for i, P in enumerate(problems):
            if i < len(runs):
                continue
            resume = current is not None and current["problem"] == i
            if resume:  # the restored optimizer already holds this pool and its evaluated rows
                for k, v in current["objective"].items():
                    setattr(P.objective, k, v)
            else:
                opt.set_pool(P.pool)

            def save(_n, i=i, P=P):
                """BO callback: checkpoint the run in progress every ``CKPT_EVERY`` seconds.

                Args:
                    _n: Number of evaluations so far (unused).
                    i: Index of the problem (bound at definition).
                    P: The problem (bound at definition).
                """
                if time.time() - ck.last >= CKPT_EVERY:
                    ck.save(opt=opt, runs=runs, train_time=t_train, elapsed=elapsed(),
                            current=dict(problem=i, objective=_objective_state(P.objective)))

            # meta-learned methods and problems with fixed inits start without random points
            k = 0 if (opt.is_meta or P.init is not None) else n_init
            h = opt.optimize(P.objective, budget, n_init=k, init=P.init, callback=save, resume=resume)
            runs.append(dict(problem=P.name, regret=P.regret(np.asarray(h.y)).tolist()))
            current = None
            if i + 1 < len(problems):
                ck.save(opt=opt, runs=runs, train_time=t_train, elapsed=elapsed(), current=None)
        res = dict(method=method, group=str(group_key), seed=seed, budget=budget, runs=runs, train=train_curves(opt),
                   train_time=t_train, time=elapsed(), resumed=state is not None)
    except Exception as e:  # keep the other jobs running
        res = dict(method=method, group=str(group_key), seed=seed, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc())
    _atomic_write(Path(out_path), json.dumps(res).encode())
    if "error" not in res:
        ck.remove()
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

    Returns:
        All cached results of ``dataset`` for ``budget`` (see ``load_results``).
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
                if rerun:  # start from scratch, not from a checkpoint
                    path.with_suffix(".ckpt.pkl").unlink(missing_ok=True)
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
    """Merge several result iterators (from different pools) in the order their items arrive.

    Args:
        iters: Iterators, each consumed by its own thread.

    Returns:
        A generator of their items.
    """
    import queue
    import threading
    q = queue.Queue()

    def pump(it):
        """Move the items of one iterator to the queue, then a None end marker.

        Args:
            it: The iterator.
        """
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
    """Read the cached job results; failed jobs are reported and skipped.

    Args:
        dataset: Folder name under ``results/raw``.
        budget: Only jobs with this budget (all if None).

    Returns:
        List of result dicts.
    """
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
    """Stack the regret curves of one method; shorter runs are padded with their last value.

    Args:
        results: Job results.
        method: Method name.
        budget: Curve length.

    Returns:
        Array of shape (runs, budget), one row per (problem, seed) run.
    """
    curves = [np.asarray(r["regret"], float) for res in results if res["method"] == method for r in res["runs"]]
    if not curves:
        return np.zeros((0, budget))
    return np.stack([np.pad(c[:budget], (0, max(0, budget - len(c))), mode="edge") for c in curves])


def mean_se(x: np.ndarray, axis=0):
    """Mean and standard error along an axis.

    Args:
        x: Data.
        axis: Axis to reduce.

    Returns:
        ``(mean, standard error)``; the error is 0 for a single sample.
    """
    x = np.asarray(x, float)
    n = x.shape[axis]
    return x.mean(axis), (x.std(axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(x.mean(axis)))


def regret_table(results: List[dict], methods: Sequence[str], budget: int, steps: Sequence[int],
                 title: str, notes: str = "") -> str:
    """Markdown table of the mean normalised regret ± s.e. at given steps and the CVaR0.8 of the final regret.

    Args:
        results: Job results.
        methods: Methods (rows), in order.
        budget: Number of steps.
        steps: Steps (columns), 1-based.
        title: Heading.
        notes: Text appended below the table.

    Returns:
        The table as Markdown; the best mean of each column is in bold.
    """
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
    """Average a training curve over jobs, interpolated on a common progress grid.

    Args:
        results: Job results.
        method: Method name.
        key: Curve name in ``train`` (e.g. ``"val_avg"``).

    Returns:
        ``(x, mean, s.e.)`` on 101 points, or None if no job has that curve.
    """
    xs = np.linspace(0, 1, 101)
    ys = [np.interp(xs, r["train"]["x"], r["train"][key]) for r in results
          if r["method"] == method and r.get("train") and r["train"].get(key)]
    return (xs, *mean_se(np.stack(ys))) if ys else None


def plot_training(axes, results: List[dict], methods: Sequence[str]) -> None:
    """Draw the meta-training panels: held-out average task loss, held-out CVaR and ABLR's -log evidence.

    Args:
        axes: Three matplotlib axes.
        results: Job results.
        methods: Meta-trained methods to draw.
    """
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


def regret_floor(results: List[dict], methods: Sequence[str], budget: int) -> float:
    """Height at which regret 0 (optimum found) is drawn on the log axis.

    Args:
        results: Job results.
        methods: Methods to consider.
        budget: Curve length.

    Returns:
        Half the smallest positive regret of any run (1e-5 if all are 0).
    """
    pos = [R[R > 0].min() for m in methods for R in [regret_matrix(results, m, budget)] if np.any(R > 0)]
    return 0.5 * min(pos) if pos else 1e-5


def _regret_stat(R: np.ndarray, stat: str, floor: float):
    """Centre line and band of regret curves.

    Args:
        R: Regret curves, shape (runs, steps).
        stat: ``"mean"`` (± s.e.), ``"median"`` (IQR band) or ``"geomean"`` (± s.e. in log space).
        floor: Value that replaces 0 before taking logs (``"geomean"``).

    Returns:
        ``(centre, low, high)``, each of shape (steps,).
    """
    if stat == "mean":
        mu, se = mean_se(R)
        return mu, mu - se, mu + se
    if stat == "median":
        return np.median(R, 0), np.quantile(R, 0.25, 0), np.quantile(R, 0.75, 0)
    L = np.log10(np.maximum(R, floor))  # "geomean": mean of log10 regret
    mu, se = mean_se(L)
    return 10 ** mu, 10 ** (mu - se), 10 ** (mu + se)


def _regret_panel(ax, results, methods, budget, title, floor, stat="mean", legend=True, ylim=None) -> None:
    """Draw the regret curves of all methods on a log axis.

    Args:
        ax: Matplotlib axis.
        results: Job results.
        methods: Methods to draw.
        budget: Number of steps.
        title: Panel title.
        floor: Lowest drawn value (regret 0 is drawn there).
        stat: Statistic of ``_regret_stat``.
        legend: Draw a legend.
        ylim: Fixed ``(low, high)`` of the axis, or None for automatic.
    """
    from visualizer.style import method_style
    x = np.arange(1, budget + 1)
    for m in methods:
        R = regret_matrix(results, m, budget)
        if len(R) == 0:
            continue
        mu, lo, hi = (np.maximum(v, floor) for v in _regret_stat(R, stat, floor))
        st = method_style(m)
        ax.plot(x, mu, label=f"{m} ({len(R)})", color=st["color"], ls=st["ls"], lw=st.get("lw", 1.8))
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, lw=0)
    ax.set_yscale("log")
    if ylim is None:
        ax.set_ylim(bottom=floor * 0.7)
        ax.axhline(floor, color="0.6", lw=0.8, ls=":")
    else:  # fixed axis: the floor is its lower edge, curves above the upper edge are cut
        ax.set_ylim(*ylim)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Normalized regret (log)")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    if legend:
        ax.legend(fontsize=7, ncol=2, title="method (runs)", title_fontsize=7)


def plot_benchmark(results: List[dict], methods: Sequence[str], budget: int, title: str, path: Path,
                   ylim: Optional[Sequence[float]] = None) -> None:
    """Save the regret + meta-training figure and a second figure comparing mean, median and geometric mean.

    Args:
        results: Job results.
        methods: Methods to draw.
        budget: Number of steps.
        title: Benchmark title.
        path: Path of ``<dataset>_plot.png``; the variants go to ``<dataset>_plot_variants.png``.
        ylim: Fixed ``(low, high)`` of the regret axes (e.g. ``(1e-9, 1e-3)``); regrets below
            ``low``, including 0, are drawn on the lower edge. None: automatic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if ylim is not None:
        ylim = (min(ylim), max(ylim))
    floor = regret_floor(results, methods, budget) if ylim is None else ylim[0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    _regret_panel(axes[0], results, methods, budget, title, floor, "mean", ylim=ylim)
    plot_training(axes[1:], results, [m for m in methods if m in TRAINED])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    stat_name = [("mean", "mean ± s.e."), ("median", "median, IQR band"), ("geomean", "geometric mean ± s.e.")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, (stat, name) in zip(axes, stat_name):
        _regret_panel(ax, results, methods, budget, f"{title}: {name}", floor, stat, legend=ax is axes[0], ylim=ylim)
    fig.suptitle(f"dotted line = regret 0 (optimum found), drawn at {floor:.2g}" if ylim is None else
                 f"regret below {floor:.0e} (including 0) is drawn on the lower edge", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_name(path.stem + "_variants.png"), dpi=150)
    plt.close(fig)


BENCH_METHODS = ["ABLR", "Random", "GP", "LFBO", "RGPE", "PFN", "MALIBO",
                 "MetaGameBO (original)", "MetaGameBO (remake)"]
# meta-training lengths on the AutoML benchmarks (author's run_hpob.py: 500 epochs)
BENCH_KWARGS = {"MALIBO": dict(iters=2000), "MetaGameBO (remake)": dict(iters=2000),
                "MetaGameBO (original)": dict(epochs=500), "ABLR": dict(iters=1000)}


def bench_main(dataset: str, title: str, groups: Sequence, group_fn: Callable, notes: str,
               default_seeds: int = 5, budget: int = 500, noisy_ok: bool = False,
               discrete_ok: bool = False) -> None:
    """Command line of ``analyze_<dataset>.py``: run the missing jobs, then write the table and the figures.

    Args:
        dataset: Results folder name.
        title: Benchmark title.
        groups: Keys of the problem groups (e.g. search spaces).
        group_fn: ``group_fn(group_key, seed) -> list[Problem]``.
        notes: Text below the table.
        default_seeds: Default of ``--seeds`` (seeds 0 .. n-1).
        budget: Default of ``--budget``.
        noisy_ok: Offer ``--noisy`` (``group_fn`` takes ``noisy=``).
        discrete_ok: Offer ``--discrete`` (``group_fn`` takes ``discrete=``).
    """
    import argparse
    ap = argparse.ArgumentParser(description=f"{title} benchmark")
    ap.add_argument("--seeds", type=int, default=default_seeds)  # number of seeds, not a seed value
    ap.add_argument("--budget", type=int, default=budget)
    ap.add_argument("--methods", nargs="+", default=BENCH_METHODS)
    ap.add_argument("--groups", nargs="+", default=None, help="subset of target groups")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gpu-workers", type=int, default=2)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="only rebuild table/plot from the cache")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="fixed log range of the regret axis, e.g. --ylim 1e-9 1e-3 (default: automatic)")
    if noisy_ok:
        ap.add_argument("--noisy", action="store_true",
                        help="observe one random training seed per evaluation; regret of the incumbent's "
                             "mean value (results under <dataset>_noisy)")
    if discrete_ok:
        ap.add_argument("--discrete", action="store_true",
                        help="choose among the tabulated evaluations of each test dataset, as in the "
                             "MALIBO/MetaGameBO code (results under <dataset>_discrete)")
    a = ap.parse_args()
    # each protocol writes to its own folder so results never mix
    if getattr(a, "discrete", False):
        import functools
        dataset, title = f"{dataset}_discrete", f"{title} (discrete)"
        group_fn = functools.partial(group_fn, discrete=True)
        notes = (notes.replace("continuous search spaces with the official surrogates",
                               "tabulated evaluations of each test dataset (discrete protocol of run_hpob.py)")
                 + " Regret normalised by the min/max accuracy of the test dataset; a dataset smaller than "
                   "the budget is evaluated completely. Steps include the 5 initial points.")
    if getattr(a, "noisy", False):
        import functools
        dataset, title = f"{dataset}_noisy", f"{title} (noisy)"
        group_fn = functools.partial(group_fn, noisy=True)
        notes += (" Noisy protocol: each evaluation returns the validation error of one random training "
                  "seed; regret = mean-over-seeds error of the incumbent (best observed point).")
    groups = [g for g in groups if a.groups is None or str(g) in a.groups]
    if not a.report_only:
        run_jobs(dataset, a.methods, groups, range(a.seeds), group_fn, a.budget, workers=a.workers,
                 gpu_workers=a.gpu_workers, method_kwargs=BENCH_KWARGS, rerun=a.rerun)
    results = [r for r in load_results(dataset, a.budget) if r["method"] in a.methods]
    steps = [s for s in (1, 10, 50, 100, 200, 300, 500) if s <= a.budget]
    md = regret_table(results, a.methods, a.budget, steps, f"{title} benchmark", notes)
    (RESULTS / f"{dataset}_benchmark.md").write_text(md)
    plot_benchmark(results, a.methods, a.budget, title, RESULTS / f"{dataset}_plot.png", ylim=a.ylim)
    print(md)
    print("wrote", RESULTS / f"{dataset}_benchmark.md", "and", RESULTS / f"{dataset}_plot.png")
