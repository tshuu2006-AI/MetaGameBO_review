"""Visualise the model's distributions: k-DPP, task embeddings, target posterior.

Functions:
    plot_kdpp: Marginals and sampled subsets of the diagonal vs quality-diversity k-DPP.
    plot_embeddings: Meta-task embeddings vs their N(0, I) prior.
    plot_posterior: Target-task acquisition: posterior mean and Thompson samples.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from models.malibo.adapters import BLRAdapter
from models.metagamebo_remake.kdpp import KDPP
from models.malibo.meta_net import MetaNet
from visualizer.style import (CATEGORICAL, DIVERGING, INK_2, MARKER, MUTED, SEQ_CMAP,
                              SEQUENTIAL, save, subplots)


def plot_kdpp(w: np.ndarray, X: np.ndarray, k: int, lengthscale: float = 0.1, n_draws: int = 8,
              seed: int = 0, path: Optional[str] = None):
    """Compare the paper's diagonal k-DPP with a quality-diversity k-DPP (1-D inputs).

    Left: inclusion probabilities vs x. Right: ``n_draws`` sampled subsets per kernel
    (one row each). A diagonal kernel has no repulsion, so its subsets can cluster.

    Args:
        w: Item weights, shape (n,).
        X: Item inputs, shape (n, 1).
        k: Subset size.
        lengthscale: RBF lengthscale of the quality-diversity kernel.
        n_draws: Samples drawn per kernel.
        seed: Seed for sampling.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    x = np.asarray(X)[:, 0]
    o = np.argsort(x)
    kernels = {"diagonal (paper)": KDPP.diagonal(w, k),
               "quality-diversity": KDPP.quality_diversity(w, X, k, lengthscale)}
    colors = dict(zip(kernels, CATEGORICAL[:2]))
    fig, axes = subplots(2, width=4.8)
    ax = axes[0]
    ax.plot(x[o], np.clip(k * w[o] / w.sum(), 0, 1), color=MUTED, lw=1.0, ls=":",
            label="k * w / sum(w)")
    for name, d in kernels.items():
        ax.plot(x[o], d.marginals()[o], "o-", color=colors[name], ms=3, label=name)
    ax.set_title(f"Inclusion probability P(i in S), k={k}")
    ax.set_xlabel("x"); ax.set_ylabel("marginal"); ax.legend()

    ax = axes[1]
    rng = np.random.default_rng(seed)
    row = 0
    for name, d in kernels.items():
        for _ in range(n_draws):
            S = d.sample(rng)
            ax.plot(x[S], np.full(len(S), row), "|", color=colors[name], ms=9, mew=2)
            row += 1
        row += 1
    ax.plot(x, np.full(len(x), -1.2), "|", color=MUTED, ms=6, label="all items")
    ax.set_yticks([n_draws / 2 - 0.5, 1.5 * n_draws + 0.5])
    ax.set_yticklabels(list(kernels))
    ax.grid(axis="y", visible=False)
    ax.set_title("Sampled subsets (one row per draw)")
    ax.set_xlabel("x")
    save(fig, path)
    return fig


def plot_embeddings(net: MetaNet, color_by: Optional[Sequence[float]] = None,
                    color_label: str = "", path: Optional[str] = None):
    """Meta-task embeddings z_tau vs their N(0, I) prior.

    Panels: histogram of all entries with the N(0, 1) density; empirical
    covariance (should be ~identity); first two principal components, optionally
    coloured by a task parameter.

    Args:
        net: Trained ``MetaNet`` (row 0 = target task is skipped).
        color_by: Optional value per meta task, shape (M,), e.g. Forrester ``b``.
        color_label: Colour-bar label for ``color_by``.
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    E = net.emb[1:].detach().cpu().numpy()
    fig, axes = subplots(3, width=3.8, height=3.4)
    ax = axes[0]
    lo, hi = min(E.min(), -3.5), max(E.max(), 3.5)
    ax.hist(E.ravel(), bins=np.linspace(lo, hi, 61), density=True, color=SEQUENTIAL[3],
            edgecolor="white", lw=0.3, label="embedding entries")
    g = np.linspace(lo, hi, 300)
    ax.plot(g, norm.pdf(g), color=INK_2, lw=1.2, label="N(0, 1) prior")
    ax.set_yscale("log"); ax.set_ylim(bottom=1e-3)
    ax.set_title(f"Marginal of z (std {E.std():.3g})"); ax.set_xlabel("value")
    ax.set_ylabel("density (log)"); ax.legend(loc="upper left")

    ax = axes[1]
    C = np.cov(E, rowvar=False)
    vmax = np.abs(C).max() or 1.0  # own scale, so structure shows even if z collapsed
    im = ax.imshow(C, cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    ax.grid(False)
    ax.set_title(f"Covariance of z (prior I; mean diag {np.diag(C).mean():.2g})"); ax.set_xlabel("dim"); ax.set_ylabel("dim")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    Ec = E - E.mean(0)
    _, s, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:2].T
    ev = s ** 2 / (s ** 2).sum()
    if color_by is None:
        ax.scatter(P[:, 0], P[:, 1], s=14, color=CATEGORICAL[0], edgecolors="white", lw=0.4)
    else:
        sc = ax.scatter(P[:, 0], P[:, 1], s=14, c=color_by, cmap=SEQ_CMAP, edgecolors="white", lw=0.4)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_label)
    ax.set_title("Tasks in embedding space (PCA)")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    save(fig, path)
    return fig


def plot_posterior(adapter: BLRAdapter, X_obs: Optional[np.ndarray] = None,
                   y_obs: Optional[np.ndarray] = None, objective=None, n_samples: int = 20,
                   seed: int = 0, grid: Optional[np.ndarray] = None, path: Optional[str] = None):
    """Target-task acquisition P(good | x) of a fitted ``BLRAdapter`` (1-D inputs).

    Draws the posterior predictive (``prob_max``) and ``n_samples`` Thompson
    samples (``prob_ts``); with ``objective`` a second panel shows the task and
    the observations.

    Args:
        adapter: Adapter after ``fit`` (or freshly reset = meta-learned prior).
        X_obs: Observed inputs, shape (n, 1).
        y_obs: Observed values, shape (n,).
        objective: Optional task callable for the lower panel.
        n_samples: Number of Thompson samples.
        seed: Seed for the samples.
        grid: Inputs to evaluate, shape (g, 1); default 400 points in [0, 1].
        path: Output image path.

    Returns:
        The matplotlib figure.
    """
    grid = np.linspace(0, 1, 400)[:, None] if grid is None else grid
    g = grid[:, 0]
    rng = np.random.default_rng(seed)
    nrows = 2 if objective is not None else 1
    fig, axes = subplots(1, nrows, width=6.0, height=2.9, sharex=True)
    ax = axes[0]
    for i in range(n_samples):
        ax.plot(g, adapter.prob_ts(grid, adapter.sample_z(rng)), color=SEQUENTIAL[1], lw=0.8,
                label="Thompson samples" if i == 0 else None)
    ax.plot(g, adapter.prob_max(grid), color=SEQUENTIAL[5], label="posterior predictive")
    if X_obs is not None and len(X_obs):
        ax.plot(X_obs[:, 0], np.zeros(len(X_obs)), "|", color=INK_2, ms=10, mew=1.5,
                label="observed x")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Acquisition P(good | x) after {0 if X_obs is None else len(X_obs)} observations")
    ax.set_ylabel("probability"); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if objective is not None:
        ax = axes[1]
        ax.plot(g, objective(grid), color=MUTED, lw=1.0, label="f(x)")
        if X_obs is not None and len(X_obs):
            ax.plot(X_obs[:, 0], y_obs, "o", color=CATEGORICAL[1], ms=MARKER, label="observations")
        ax.set_ylabel("f(x)"); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xlabel("x")
    save(fig, path)
    return fig
