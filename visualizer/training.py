"""Visualise the meta-training process.

Use ``TrainRecorder`` as the ``callback`` of ``models.malibo.training.meta_train``
to keep snapshots, then plot them:

    rec = TrainRecorder()
    net, log = meta_train(tasks, cfg, callback=rec)
    plot_train_logs({"MetaGameBO": log}, path="fig/train.png")
    plot_task_loss_evolution(rec, path="fig/task_losses.png")

Functions:
    plot_train_logs: Avg / CVaR task loss and selected-task fraction of several runs.
    plot_task_loss_evolution: Quantile bands of per-task losses and the CVaR threshold.
    plot_task_loss_histogram: Per-task loss distribution at one snapshot.
    plot_sample_weight_evolution: Eq. (6) sample weights of one task over training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np

from models.malibo.training import TrainLog, TrainSnapshot
from models.base import cvar
from visualizer.style import (CATEGORICAL, INK_2, MARKER, MUTED, SEQ_CMAP, SEQUENTIAL,
                              label_line_end, method_style, ordered, save, subplots)


@dataclass
class Record:
    """Compact copy of a ``TrainSnapshot`` (no network, no shared arrays).

    Attributes:
        it: Iteration.
        task_losses: Per-task losses, shape (M,).
        hard: Selected task indices.
        xi: CVaR threshold or None.
        sample_weights: Eq. (6) weights, shape (n,).
        embeddings: Meta-task embeddings, shape (M, num_features).
    """
    it: int
    task_losses: np.ndarray
    hard: np.ndarray
    xi: Optional[float]
    sample_weights: np.ndarray
    embeddings: np.ndarray


@dataclass
class TrainRecorder:
    """Callback for ``meta_train`` that stores a ``Record`` per logged iteration.

    Attributes:
        records: Records in iteration order.
        X, labels, task_index, util_weights: Static training data (from the first snapshot).
    """
    records: List[Record] = field(default_factory=list)
    X: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    task_index: Optional[np.ndarray] = None
    util_weights: Optional[np.ndarray] = None

    def __call__(self, s: TrainSnapshot) -> None:
        if self.X is None:
            self.X, self.labels = s.X.copy(), s.labels.copy()
            self.task_index, self.util_weights = s.task_index.copy(), s.util_weights.copy()
        emb = s.net.emb[1:].detach().cpu().numpy().copy()
        self.records.append(Record(s.it, s.task_losses, s.hard, s.xi, s.sample_weights, emb))


def _as_dict(log: Union[TrainLog, dict]) -> dict:
    d = dict(log.__dict__) if isinstance(log, TrainLog) else dict(log)
    if not d.get("iteration"):  # logs saved before `iteration` existed (every 50 its)
        d["iteration"] = list(range(0, 50 * len(d["avg_task_loss"]), 50))
    return d


def plot_train_logs(logs: Dict[str, Union[TrainLog, dict]], path: Optional[str] = None):
    """Training curves of several runs: mean task loss, CVaR task loss, selected fraction.

    Args:
        logs: Method name -> ``TrainLog`` (or its ``__dict__``, as stored in results JSON).
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    panels = [("avg_task_loss", "Mean task loss (train)"), ("cvar_task_loss", "CVaR0.8 task loss (train)"),
              ("hard_frac", "Fraction of tasks in the gradient step")]
    if any(_as_dict(lg).get("val_cvar_task_loss") for lg in logs.values()):
        panels.insert(2, ("val_cvar_task_loss", "CVaR0.8 held-out loss (Table 1)"))
    fig, axes = subplots(len(panels), width=3.8)
    for name in ordered(logs):
        d = _as_dict(logs[name])
        for ax, (k, _) in zip(axes, panels):
            if d.get(k):
                ax.plot(d["iteration"], d[k], label=name, **method_style(name))
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title); ax.set_xlabel("meta-training iteration")
    axes[0].set_ylabel("weighted BCE")
    axes[-1].set_ylim(0, 1.05)
    if len(logs) > 1:
        axes[-1].legend()
    save(fig, path)
    return fig


def plot_task_loss_evolution(rec: TrainRecorder, alpha: float = 0.8, path: Optional[str] = None):
    """Distribution of per-task losses over training (median, 10-90% band, max) and xi.

    Args:
        rec: Filled recorder.
        alpha: CVaR level for the CVaR line.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    its = np.array([r.it for r in rec.records])
    L = np.stack([r.task_losses for r in rec.records])
    q10, q50, q90 = np.quantile(L, [0.1, 0.5, 0.9], axis=1)
    fig, (ax,) = subplots(width=6.0, height=3.6)
    ax.fill_between(its, q10, q90, color=SEQUENTIAL[1], alpha=0.6, lw=0, label="10-90% of tasks")
    ax.plot(its, q50, color=SEQUENTIAL[4], label="median task")
    cv = [cvar(l, alpha) for l in L]
    ax.plot(its, cv, color=CATEGORICAL[1], label=f"CVaR{alpha}")
    ax.plot(its, L.max(1), color=MUTED, lw=1.0, ls=":", label="worst task")
    xi = [r.xi for r in rec.records]
    if any(v is not None for v in xi):
        ax.plot(its, [np.nan if v is None else v for v in xi], color=INK_2, lw=1.0, ls="--",
                label="xi (KDE quantile)")
    label_line_end(ax, its[-1], q50[-1], "median")
    ax.set_title("Per-task loss during meta-training")
    ax.set_xlabel("meta-training iteration"); ax.set_ylabel("weighted BCE"); ax.legend()
    save(fig, path)
    return fig


def plot_task_loss_histogram(rec: TrainRecorder, which: int = -1, bins: int = 30,
                             path: Optional[str] = None):
    """Per-task loss histogram at one snapshot, selected (hard) tasks highlighted.

    Args:
        rec: Filled recorder.
        which: Record index (default: last).
        bins: Histogram bins.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    r = rec.records[which]
    mask = np.zeros(len(r.task_losses), bool); mask[r.hard] = True
    edges = np.linspace(r.task_losses.min(), r.task_losses.max(), bins + 1)
    fig, (ax,) = subplots(width=5.5)
    if mask.all():
        ax.hist(r.task_losses, bins=edges, color=SEQUENTIAL[3], edgecolor="white", lw=0.5,
                label=f"all {mask.size} tasks selected")
    else:
        ax.hist(r.task_losses[~mask], bins=edges, color=SEQUENTIAL[2], edgecolor="white", lw=0.5,
                label="not selected")
        ax.hist(r.task_losses[mask], bins=edges, color=CATEGORICAL[1], edgecolor="white", lw=0.5,
                label=f"selected ({mask.sum()} tasks)")
    if r.xi is not None:
        ax.axvline(r.xi, color=INK_2, ls="--", lw=1.0, label="xi")
    ax.set_title(f"Per-task loss at iteration {r.it}")
    ax.set_xlabel("weighted BCE"); ax.set_ylabel("tasks"); ax.legend()
    save(fig, path)
    return fig


def plot_sample_weight_evolution(rec: TrainRecorder, task: Optional[int] = None,
                                 n_snapshots: int = 6, path: Optional[str] = None):
    """Eq. (6) sample weights of one meta task across training (1-D inputs).

    Colour goes light -> dark with the iteration; filled = class 1, hollow = class 0.

    Args:
        rec: Filled recorder.
        task: 0-based meta-task index; default: the task whose weights moved
            most from their Eq. (3) initialisation.
        n_snapshots: Number of records shown, evenly spaced.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    if task is None:
        shift = np.abs(np.log(rec.records[-1].sample_weights / rec.util_weights))
        per_task = np.bincount(rec.task_index, weights=shift)[1:]
        task = int(np.argmax(per_task))
    sel = rec.task_index == task + 1
    x, lab = rec.X[sel, 0], rec.labels[sel]
    picks = np.unique(np.linspace(0, len(rec.records) - 1, n_snapshots).round().astype(int))
    fig, (ax,) = subplots(width=6.0, height=3.6)
    for j, p in enumerate(picks):
        r = rec.records[p]
        c = SEQ_CMAP(0.25 + 0.75 * j / max(len(picks) - 1, 1))
        w = r.sample_weights[sel]
        ax.plot(x[lab == 1], w[lab == 1], "o", color=c, ms=MARKER)
        ax.plot(x[lab == 0], w[lab == 0], "o", mfc="none", color=c, ms=MARKER,
                label=f"it {r.it}")
    ax.plot(x, rec.util_weights[sel], "x", color=INK_2, ms=MARKER, label="Eq. (3) init")
    ax.set_yscale("log")
    ax.set_title(f"Sample weights of meta task {task} (filled = class 1)")
    ax.set_xlabel("x"); ax.set_ylabel("w (log)"); ax.legend(ncol=2)
    save(fig, path)
    return fig
