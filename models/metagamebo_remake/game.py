"""The hierarchical game of MetaGameBO (paper Algorithm 1), as selectors for
``models.malibo.training.meta_train``.

* Leader (task level, ``CVaRTasks``): each step, the per-task losses L_i give
  xi = KDE alpha-quantile, and the gradient step uses T_hard = {i : L_i >= xi}.
* Follower (sample level, ``KDPPSamples``): per selected task, a subset
  S ~ k-DPP(diag(w_t)) (or a quality-diversity kernel), the gradient on S is
  weighted by w_t (optionally w_t / q_i), then Eq. (6):
  w_{t+1,i} = w_{t,i} exp(eta_s (l_i - mean l) / q_i).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from models.malibo.meta_net import DT
from models.metagamebo_remake.kdpp import KDPP, DiagonalKDPP


def kde_quantile(values: np.ndarray, alpha: float) -> float:
    """Alpha-quantile of a Gaussian KDE of the values (paper: "estimate xi with KDE").

    Args:
        values: Task losses (NaNs dropped).
        alpha: Quantile level.

    Returns:
        The quantile xi; the minimum if there are fewer than 2 values or they are all equal.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or v.std() == 0:
        return float(v.min())
    h = 1.06 * v.std() * len(v) ** (-1 / 5)  # Silverman's bandwidth
    # invert the KDE CDF on a grid over the observed range
    grid = np.linspace(v.min(), v.max(), 1000)
    mat = (grid[:, None] - v[None, :]) / h
    cdf = norm.cdf(mat).mean(1)
    return float(grid[np.argmin(np.abs(cdf - alpha))])


class CVaRTasks:
    """Leader: train on the tasks whose loss is above the KDE alpha-quantile.

    Args:
        alpha: Confidence level (0.8 -> about the worst 20% of tasks).
    """

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self.xi = None  # last threshold, logged by the training loop

    def select(self, task_losses: np.ndarray) -> np.ndarray:
        """Select the tasks with ``L_i >= xi``.

        Args:
            task_losses: Loss per task, shape (M,).

        Returns:
            Indices of the hard tasks (the worst task alone if none passes the threshold).
        """
        self.xi = kde_quantile(task_losses, self.alpha)
        hard = np.flatnonzero(task_losses >= self.xi)
        return hard if len(hard) else np.array([int(np.nanargmax(task_losses))])


class KDPPSamples:
    """Follower: k-DPP subsets per task with the multiplicative weight update of Eq. (6).

    Args:
        frac: Subset size k as a fraction of each task's samples.
        eta: Sample-weight learning rate eta_s.
        is_grad: Weight the gradient by ``w_i / q_i`` (unbiased for the full
            weighted loss) instead of ``w_i``.
        kernel: ``"diag"`` (paper, L = diag(w)) or ``"qd"`` (quality-diversity,
            L = diag(sqrt w) S_rbf diag(sqrt w); not in the paper).
    """

    def __init__(self, frac: float = 0.4, eta: float = 0.1, is_grad: bool = True, kernel: str = "diag"):
        if kernel not in ("diag", "qd"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.frac, self.eta, self.is_grad, self.kernel = frac, eta, is_grad, kernel

    def init(self, W_util: torch.Tensor, slices: list, X: torch.Tensor, rng) -> None:
        """Start from the utility weights; keep the task slices, inputs and generator.

        Args:
            W_util: Eq. (3) weights of the training samples, tensor (n,) (updated in place).
            slices: Per task, the indices of its samples in the training arrays.
            X: Training inputs, tensor (n, d) (used by the ``"qd"`` kernel).
            rng: Numpy random generator of the k-DPP draws.
        """
        self.W, self.slices, self.X, self.rng = W_util, slices, X, rng

    @property
    def weights(self) -> torch.Tensor:
        """Current sample weights w_t.

        Returns:
            Tensor of shape (n,).
        """
        return self.W

    def select(self, hard: np.ndarray):
        """Draw a k-DPP subset of every selected task and compute its gradient weights.

        Args:
            hard: Indices of the tasks selected by the leader.

        Returns:
            ``(idx, weights)``: sample indices and gradient weights normalised to mean 1.
        """
        idx_l, q_l = [], []
        for i in hard:
            sl = self.slices[i]
            wt = self.W[sl].numpy()
            k = max(1, int(self.frac * len(sl)))
            d = DiagonalKDPP(wt, k) if self.kernel == "diag" else KDPP.quality_diversity(wt, self.X[sl].numpy(), k)
            S = d.sample(self.rng)
            idx_l.append(sl[S]); q_l.append(d.marginals()[S])  # q_i = P(i in S)
        # kept for ``update``: subset size per task and inclusion probabilities
        self._sizes = [len(s) for s in idx_l]
        self._q = np.concatenate(q_l)
        idx = np.concatenate(idx_l)
        gw = self.W[idx]
        if self.is_grad:
            gw = gw / torch.as_tensor(np.maximum(self._q, 1e-3), dtype=DT)
        return idx, gw / gw.mean()

    def update(self, hard, idx, per_sample_loss) -> None:
        """Multiplicative weight update of Eq. (6) with the correction 1/q_i; weights stay mean-1 per task.

        Args:
            hard: Indices of the selected tasks.
            idx: Indices of the samples used in the step (from ``select``).
            per_sample_loss: Unweighted BCE of those samples.
        """
        upd = np.zeros_like(per_sample_loss)
        pos = 0
        for n in self._sizes:  # centre the losses within each task
            li = per_sample_loss[pos:pos + n]
            upd[pos:pos + n] = li - li.mean()
            pos += n
        expo = np.clip(self.eta * upd / np.maximum(self._q, 1e-3), -5, 5)  # clipped for stability
        idx_t = torch.as_tensor(idx)
        with torch.no_grad():
            self.W[idx_t] = self.W[idx_t] * torch.tensor(np.exp(expo), dtype=DT)
            for i in hard:
                sl = self.slices[i]
                self.W[sl] = torch.clamp(self.W[sl] / self.W[sl].mean(), 1e-3, 1e3)
