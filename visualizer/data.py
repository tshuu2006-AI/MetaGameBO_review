"""Visualise the task distribution and the classifier's training data.

Functions:
    plot_task_family: Objective curves of a set of 1-D tasks per split.
    plot_task_parameters: Histograms of task parameters per split.
    plot_optimum_locations: Where the global minima of each split lie.
    plot_reweighting: One meta task with its Eq. (3) labels and weights.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from models.lfbo.weighting import reweight_samples
from visualizer.style import CATEGORICAL, INK_2, MARKER, MUTED, save, subplots

GRID_1D = np.linspace(0.0, 1.0, 501)


def _split_colors(names):
    return {n: CATEGORICAL[i % len(CATEGORICAL)] for i, n in enumerate(names)}


def _hist(ax, values, edges, color, label):
    """Overlay-friendly histogram: light fill plus a solid outline (no blended colours)."""
    ax.hist(values, bins=edges, color=color, alpha=0.25, lw=0)
    ax.hist(values, bins=edges, histtype="step", color=color, lw=1.4, label=label)


def plot_task_family(tasks_by_split: Dict[str, Sequence[Callable]], n_show: int = 30,
                     grid: np.ndarray = GRID_1D, path: Optional[str] = None):
    """Overlay the objective curves of 1-D tasks, one panel per split.

    Args:
        tasks_by_split: Split name -> tasks (callables mapping (n, 1) -> (n,)).
        n_show: Maximum number of curves drawn per split.
        grid: Evaluation grid in [0, 1].
        path: Output image path (figure is returned unsaved if None).

    Returns:
        The matplotlib figure.
    """
    colors = _split_colors(tasks_by_split)
    fig, axes = subplots(len(tasks_by_split), sharey=True)
    for ax, (name, tasks) in zip(axes, tasks_by_split.items()):
        for t in list(tasks)[:n_show]:
            f = t(grid[:, None])
            ax.plot(grid, f, color=colors[name], lw=0.8, alpha=0.45)
            i = int(np.argmin(f))
            ax.plot(grid[i], f[i], "o", ms=3, color=INK_2)
        ax.set_title(f"{name}: {min(n_show, len(tasks))} tasks")
        ax.set_xlabel("x")
    axes[0].set_ylabel("f(x)   (dots: global minima)")
    save(fig, path)
    return fig


def plot_task_parameters(params_by_split: Dict[str, Dict[str, np.ndarray]], bins: int = 30,
                         path: Optional[str] = None):
    """Histogram of each task parameter, splits overlaid.

    Args:
        params_by_split: Split name -> {parameter name -> values (n,)}.
            For Forrester tasks: ``{"a": [t.a for t in tasks], ...}``.
        bins: Histogram bins.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    colors = _split_colors(params_by_split)
    keys = list(next(iter(params_by_split.values())))
    fig, axes = subplots(len(keys), width=3.4)
    for ax, k in zip(axes, keys):
        allv = np.concatenate([np.asarray(p[k]) for p in params_by_split.values()])
        edges = np.linspace(allv.min(), allv.max(), bins + 1)
        for name, p in params_by_split.items():
            _hist(ax, p[k], edges, colors[name], name)
        ax.set_title(f"parameter {k}"); ax.set_xlabel(k)
    axes[0].set_ylabel("tasks")
    axes[-1].legend()
    save(fig, path)
    return fig


def plot_optimum_locations(tasks_by_split: Dict[str, Sequence[Callable]], bins: int = 40,
                           grid: np.ndarray = GRID_1D, path: Optional[str] = None):
    """Histogram of the global-minimum location x* per split (shows distribution shift).

    Args:
        tasks_by_split: Split name -> 1-D tasks.
        bins: Histogram bins over [0, 1].
        grid: Grid used to locate the minimum.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    colors = _split_colors(tasks_by_split)
    fig, (ax,) = subplots(width=5.5)
    edges = np.linspace(0, 1, bins + 1)
    for name, tasks in tasks_by_split.items():
        xs = [grid[int(np.argmin(t(grid[:, None])))] for t in tasks]
        _hist(ax, xs, edges, colors[name], f"{name} (n={len(xs)})")
    ax.set_title("Location of the global minimum x*")
    ax.set_xlabel("x*"); ax.set_ylabel("tasks"); ax.legend()
    save(fig, path)
    return fig


def plot_reweighting(task, gamma: float = 1 / 3, scheme: str = "lfbo",
                     grid: np.ndarray = GRID_1D, path: Optional[str] = None):
    """Show how one task's data become classifier targets (paper Eq. 3).

    Marker area is proportional to the sample weight; filled = class 1 ("good",
    y < tau), hollow = class 0. With ``scheme="lfbo"`` the positives also appear
    as class 0 (drawn as hollow rings around them).

    Args:
        task: ``models.base.Task`` with ``objective``, ``X`` (n, 1) and ``y`` (n,).
        gamma: Quantile threshold.
        scheme: ``"lfbo"`` or ``"paper"``, see ``reweight_samples``.
        grid: Grid for the objective curve.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    x, lab, w = reweight_samples(task.X, task.y, gamma, scheme)
    tau = np.quantile(np.unique(task.y), gamma)
    fig, axes = subplots(2, width=4.6)
    ax = axes[0]
    ax.plot(grid, task.objective(grid[:, None]), color=MUTED, lw=1.0, label="f(x)")
    ax.axhline(tau, color=INK_2, lw=0.8, ls=":", label=f"tau ({gamma:.2f}-quantile)")
    good = task.y < tau
    ax.plot(task.X[good, 0], task.y[good], "o", color=CATEGORICAL[0], ms=MARKER, label="y < tau")
    ax.plot(task.X[~good, 0], task.y[~good], "o", mfc="none", color=CATEGORICAL[1], ms=MARKER,
            label="y >= tau")
    ax.set_title("Observations"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()

    ax = axes[1]
    s = 30 * w / w.mean()
    pos = lab == 1
    ax.scatter(x[pos, 0], w[pos], s=s[pos], color=CATEGORICAL[0], label="class 1", zorder=3)
    ax.scatter(x[~pos, 0], w[~pos], s=s[~pos], facecolors="none", edgecolors=CATEGORICAL[1],
               label="class 0", zorder=2)
    ax.set_title(f"Classifier targets ({scheme}): {len(x)} samples")
    ax.set_xlabel("x"); ax.set_ylabel("sample weight w"); ax.legend()
    save(fig, path)
    return fig

