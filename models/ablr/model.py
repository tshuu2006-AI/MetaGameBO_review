"""ABLR (Perrone et al. 2018): a neural network shared by all tasks feeds one
Bayesian linear regression (BLR) per task.

Meta-training maximises the sum of the BLR log marginal likelihoods of the
source tasks over the network weights and each task's precisions (alpha, beta).
On the target task the features stay fixed and only the target alpha, beta are
fitted by evidence maximisation; the acquisition is Expected Improvement.
(The original also refits the network as target data arrive; we keep it fixed.)
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from models.base import N_CAND, BaseOptimizer, Task
from models.gp.model import expected_improvement

DT = torch.float64


class FeatureNet(torch.nn.Module):
    """Two tanh hidden layers -> ``num_features`` features."""

    def __init__(self, d: int, hidden: int = 50, num_features: int = 50):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, num_features), torch.nn.Tanh())

    def forward(self, x):
        return self.net(x)


def blr_log_evidence(Phi: torch.Tensor, y: torch.Tensor, log_alpha, log_beta) -> torch.Tensor:
    """Log marginal likelihood of BLR with weight prior N(0, I/alpha) and noise 1/beta."""
    n, D = Phi.shape
    a, b = log_alpha.exp(), log_beta.exp()
    A = b * Phi.T @ Phi + a * torch.eye(D, dtype=Phi.dtype)
    L = torch.linalg.cholesky(A)
    m = b * torch.cholesky_solve((Phi.T @ y).unsqueeze(1), L).squeeze(1)
    r = y - Phi @ m
    return (0.5 * n * log_beta + 0.5 * D * log_alpha - 0.5 * b * (r @ r) - 0.5 * a * (m @ m)
            - torch.log(torch.diagonal(L)).sum() - 0.5 * n * math.log(2 * math.pi))


class ABLR(BaseOptimizer):
    """Adaptive Bayesian linear regression with meta-learned features.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed.
        iters: Meta-training steps (Adam on the summed evidence).
        lr: Adam learning rate.
        max_per_task: Subsample source tasks to at most this many points.
    """

    is_meta = True
    name = "ABLR"

    def __init__(self, bounds, seed: Optional[int] = None, iters: int = 1000, lr: float = 1e-3,
                 max_per_task: Optional[int] = 512):
        super().__init__(bounds, seed)
        self.iters, self.lr, self.max_per_task = iters, lr, max_per_task
        self.net = None
        self.log = None
        self._ab = np.zeros(2)

    def meta_train(self, tasks: Sequence[Task], **kwargs) -> None:
        torch.manual_seed(int(self.rng.integers(1 << 30)))
        data = []
        for t in tasks:
            X, y = self._normalize(t.X), np.asarray(t.y, float)
            if self.max_per_task and len(y) > self.max_per_task:
                keep = self.rng.choice(len(y), self.max_per_task, replace=False)
                X, y = X[keep], y[keep]
            y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
            data.append((torch.tensor(X, dtype=DT), torch.tensor(y, dtype=DT)))
        self.net = FeatureNet(self.dim).to(DT)
        hyp = torch.zeros(len(data), 2, dtype=DT, requires_grad=True)  # log alpha, log beta per task
        opt = torch.optim.Adam(list(self.net.parameters()) + [hyp], lr=self.lr)
        self.log = dict(iteration=[], neg_log_evidence=[])
        for it in range(self.iters):
            loss = -sum(blr_log_evidence(self.net(X), y, hyp[i, 0], hyp[i, 1]) / len(y)
                        for i, (X, y) in enumerate(data)) / len(data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0 or it == self.iters - 1:
                self.log["iteration"].append(it)
                self.log["neg_log_evidence"].append(float(loss))
        self._ab = hyp.detach().mean(0).numpy()  # initial target precisions
        self.net.eval()

    def _features(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(self._normalize(X), dtype=DT)).numpy()

    def _posterior(self, Phi, y):
        """Fit the target (log alpha, log beta) and return the predictive function."""
        def nle(p):
            with torch.no_grad():
                return -float(blr_log_evidence(torch.tensor(Phi), torch.tensor(y),
                                               torch.tensor(p[0], dtype=DT), torch.tensor(p[1], dtype=DT)))
        res = minimize(nle, self._ab, method="L-BFGS-B", bounds=[(-10, 10), (-10, 10)])
        a, b = np.exp(res.x)
        A = b * Phi.T @ Phi + a * np.eye(Phi.shape[1])
        Ainv = np.linalg.inv(A)
        m = b * Ainv @ Phi.T @ y

        def predict(P):
            return P @ m, np.sqrt(1 / b + np.einsum("ij,jk,ik->i", P, Ainv, P))
        return predict

    def suggest(self, n: int = 1) -> np.ndarray:
        """EI maximiser over ``N_CAND`` candidates (prior mean on the first step)."""
        X, y = self.history.arrays
        C = self._candidates(N_CAND)
        if len(y) == 0:
            # no target data: features carry the meta-learned prior only through alpha; random
            return self._candidates(n)
        ys = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
        predict = self._posterior(self._features(X), ys)
        mu, sd = predict(self._features(C))
        return C[[int(np.argmax(expected_improvement(mu, sd, ys.min())))]]
