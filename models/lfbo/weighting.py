"""Labels and utility weights of observations for classifier-based BO.

LFBO (Song et al. 2022) turns "is x better than the gamma-quantile?" into a
weighted classification problem; MALIBO and MetaGameBO reuse it (paper Eq. 3).
"""
from __future__ import annotations

import numpy as np

SCHEMES = ("lfbo", "paper")


def reweight_samples(x: np.ndarray, y: np.ndarray, gamma: float = 1 / 3,
                     scheme: str = "lfbo", return_index: bool = False):
    """Label and weight observations (paper Eq. 3).

    ``tau`` is the gamma-quantile of the unique values. Points with ``y < tau``
    are positives weighted by the improvement ``tau - y``; weights are
    class-balanced and normalised to mean 1.

    Args:
        x: Inputs, shape (n, d).
        y: Objective values, shape (n,).
        gamma: Quantile that defines the threshold ``tau``.
        scheme: ``"lfbo"`` (LFBO/MALIBO/author code): the reference class 0 is
            *all* points, so every positive appears twice. ``"paper"`` (Eq. 3 as
            written): class 0 is only ``y >= tau``.
        return_index: Also return, for every output row, the index of its source point.

    Returns:
        ``(x, lab, w)`` with shapes (m, d), (m,), (m,), plus the source indices
        (m,) if ``return_index``; m = n + #positives for ``"lfbo"``, n for ``"paper"``.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown weighting scheme {scheme!r}, choose from {SCHEMES}")
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.atleast_2d(x)
    tau = threshold(y, gamma)
    z = y < tau
    if len(x) == 1:
        out = (x, z.astype(float), np.ones(1))
        return (*out, np.zeros(1, int)) if return_index else out
    i1 = np.flatnonzero(z)
    i0 = np.arange(len(y)) if scheme == "lfbo" else np.flatnonzero(~z)
    w1 = (tau - y)[i1]
    w1 = w1 / w1.mean() if len(w1) else w1
    s1, s0 = len(i1), len(i0)
    w = np.concatenate([w1 * (s1 + s0) / max(s1, 1), np.ones(s0) * (s1 + s0) / max(s0, 1)])
    w = w / w.mean()
    idx = np.concatenate([i1, i0])
    lab = np.concatenate([np.ones(s1), np.zeros(s0)])
    return (x[idx], lab, w, idx) if return_index else (x[idx], lab, w)


def threshold(y: np.ndarray, gamma: float = 1 / 3) -> float:
    """``tau``: the gamma-quantile of the unique values of ``y``."""
    return float(np.quantile(np.unique(np.asarray(y, dtype=float)), q=gamma))
