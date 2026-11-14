"""Visualise optimisation runs and benchmark results.

Results are the dicts written by ``experiments/run_forrester_*.py``: one per
(method, seed) with a list of normalised-regret curves per split.

Functions:
    group_results: Group result dicts by split and method.
    plot_regret_curves: Mean normalised regret vs evaluations, one panel per split.
    plot_regret_ecdf: ECDF of the final per-task regret (shows the tail CVaR targets).
    plot_bo_trace: One BO run on a 1-D task: queries in order and best-so-far.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from visualizer.style import (INK_2, MARKER, MUTED, SEQ_CMAP, label_line_end, method_style,
                              ordered, save, subplots)

SPLITS = ("in_dist", "shifted")


def group_results(res: Sequence[dict], splits: Sequence[str] = SPLITS) -> Dict[str, Dict[str, List[dict]]]:
    """Group result dicts as ``{split: {method: [result, ...]}}``."""
    out = {}
    for split in splits:
        by = defaultdict(list)
        for r in res:
            if split in r:
                by[r["method"]].append(r)
        out[split] = dict(by)
    return out


def plot_regret_curves(by_split: Dict[str, Dict[str, List[dict]]], log: bool = True,
                       path: Optional[str] = None):
    """Mean normalised regret per evaluation, averaged over seeds and target tasks.

    Args:
        by_split: Output of ``group_results``.
        log: Log-scale y axis.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    fig, axes = subplots(len(by_split), width=5.2, height=3.8, sharey=True)
    for ax, (split, by) in zip(axes, by_split.items()):
        for m in ordered(by):
            C = np.array([c for r in by[m] for c in r[split]])
            mean = np.maximum(C.mean(0), 1e-6)
            steps = np.arange(1, C.shape[1] + 1)
            ax.plot(steps, mean, label=m, **method_style(m))
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("evaluations"); ax.set_title(f"Forrester ({split})")
    axes[0].set_ylabel("mean normalised regret")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, path)
    return fig


def plot_regret_ecdf(by_split: Dict[str, Dict[str, List[dict]]], step: int = -1,
                     path: Optional[str] = None):
    """ECDF of per-task normalised regret at ``step`` (pooled over seeds).

    Curves further left / higher are better; the upper-right tail is what CVaR
    measures.

    Args:
        by_split: Output of ``group_results``.
        step: Evaluation index (-1 = final).
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    fig, axes = subplots(len(by_split), width=5.2, height=3.8, sharey=True)
    for ax, (split, by) in zip(axes, by_split.items()):
        for m in ordered(by):
            v = np.sort([max(c[step], 1e-7) for r in by[m] for c in r[split]])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=m, **method_style(m))
        ax.set_xscale("log")
        ax.axhline(0.8, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("normalised regret"); ax.set_title(f"Per-task regret ECDF ({split})")
    axes[0].set_ylabel("fraction of target tasks")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, path)
    return fig


def plot_bo_trace(objective, X: np.ndarray, y: np.ndarray, title: str = "",
                  grid: Optional[np.ndarray] = None, path: Optional[str] = None):
    """One BO run on a 1-D task: objective, numbered queries and best-so-far.

    Args:
        objective: Task callable, (n, 1) -> (n,).
        X: Queried points in order, shape (n, 1).
        y: Observed values, shape (n,).
        title: Panel title (e.g. the method name).
        grid: Grid for the objective, shape (g, 1).
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    grid = np.linspace(0, 1, 500)[:, None] if grid is None else grid
    n = len(y)
    order = np.arange(n)
    colors = SEQ_CMAP(0.25 + 0.75 * order / max(n - 1, 1))
    fig, axes = subplots(2, width=4.6)
    ax = axes[0]
    f = objective(grid)
    ax.plot(grid[:, 0], f, color=MUTED, lw=1.0)
    ax.scatter(X[:, 0], y, c=colors, s=30, edgecolors="white", lw=0.6, zorder=3)
    for i in (0, n - 1):
        ax.annotate(str(i + 1), (X[i, 0], y[i]), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    i = int(np.argmin(f))
    ax.plot(grid[i, 0], f[i], "*", color=INK_2, ms=MARKER + 4, label="global minimum")
    ax.set_title(f"{title} queries (light -> dark = order)"); ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend()

    ax = axes[1]
    best = np.minimum.accumulate(y)
    ax.plot(order + 1, y, "o", color=MUTED, ms=4, label="observed")
    ax.plot(order + 1, best, color=INK_2, drawstyle="steps-post", label="best so far")
    ax.axhline(f.min(), color=MUTED, ls=":", lw=0.8)
    label_line_end(ax, n, best[-1], f"{best[-1]:.2f}")
    ax.set_title("Progress"); ax.set_xlabel("evaluation"); ax.set_ylabel("y"); ax.legend()
    save(fig, path)
    return fig
