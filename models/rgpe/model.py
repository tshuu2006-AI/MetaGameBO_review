"""RGPE (Feurer et al. 2018): ranking-weighted ensemble of Gaussian processes.

One GP per source task plus a GP on the target data. Ensemble weights are the
probability (over posterior samples) that a model has the lowest ranking loss
on the target observations (number of mis-ordered pairs); the target model uses
leave-one-out samples. Base models whose median loss exceeds the 95th percentile
of the target model's loss are dropped (weight dilution). The ensemble mean and
variance are the weighted sums; the acquisition is Expected Improvement.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from models.base import N_CAND, BaseOptimizer, Task
from models.gp.model import expected_improvement


def _std(y):
    """Standardise values to zero mean and unit variance.

    Args:
        y: Values, shape (n,).

    Returns:
        Standardised values, shape (n,); only centred if the values are constant.
    """
    y = np.asarray(y, float)
    s = y.std()
    return (y - y.mean()) / (s if s > 0 else 1.0)


def _gp(X, y, seed):
    """Fit one ensemble member: a Matern-5/2 ARD GP with noise on standardised targets.

    Args:
        X: Inputs in the unit cube, shape (n, d).
        y: Standardised targets, shape (n,).
        seed: Seed of the hyperparameter restart.

    Returns:
        The fitted ``GaussianProcessRegressor``.
    """
    d = X.shape[1]
    k = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), nu=2.5, length_scale_bounds=(1e-3, 1e2)) \
        + WhiteKernel(1e-3, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=1, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def ranking_losses(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Count the pairs each posterior sample orders differently from the observations.

    Args:
        samples: Posterior samples at the observed points, shape (S, n).
        y: Observed values, shape (n,).

    Returns:
        Number of mis-ordered pairs per sample, shape (S,).
    """
    obs = y[:, None] < y[None, :]
    pred = samples[:, :, None] < samples[:, None, :]
    return (pred ^ obs).sum((1, 2))


class RGPE(BaseOptimizer):
    """Ranking-weighted GP ensemble.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed.
        n_samples: Posterior samples for the ranking-loss weights.
        max_per_task: Subsample source tasks to at most this many points.
        max_points: Use at most this many target points (best half + random rest).
    """

    is_meta = True
    name = "RGPE"

    def __init__(self, bounds, seed: Optional[int] = None, n_samples: int = 256,
                 max_per_task: int = 256, max_points: int = 200):
        super().__init__(bounds, seed)
        self.n_samples, self.max_per_task, self.max_points = n_samples, max_per_task, max_points
        self.base = []  # one GP per source task

    def meta_train(self, tasks: Sequence[Task], **kwargs) -> None:
        """Fit one GP per source task.

        Args:
            tasks: Source tasks with ``X``/``y``.
            **kwargs: Unused.
        """
        self.base = []
        for t in tasks:
            X, y = self._normalize(t.X), np.asarray(t.y, float)
            if len(y) > self.max_per_task:
                keep = self.rng.choice(len(y), self.max_per_task, replace=False)
                X, y = X[keep], y[keep]
            self.base.append(_gp(X, _std(y), int(self.rng.integers(1 << 30))))

    def _subset(self, X, y):
        """Keep the best half of ``max_points`` target points and fill the rest at random.

        Args:
            X: Observed points, shape (n, d).
            y: Observed values, shape (n,).

        Returns:
            ``(X, y)`` with at most ``max_points`` rows.
        """
        if len(y) <= self.max_points:
            return X, y
        order = np.argsort(y)
        h = self.max_points // 2
        idx = np.concatenate([order[:h], self.rng.choice(order[h:], self.max_points - h, replace=False)])
        return X[idx], y[idx]

    def _weights(self, U, ys, target):
        """Ensemble weights: how often each model has the lowest ranking loss, with weight dilution.

        Args:
            U: Target points in the unit cube, shape (n, d).
            ys: Standardised target values, shape (n,).
            target: GP fitted on ``(U, ys)``.

        Returns:
            Weights of (base models..., target), summing to 1.
        """
        n, S = len(ys), self.n_samples
        if n < 3:  # too few points to rank: uniform over the base models
            w = np.zeros(len(self.base) + 1)
            w[:-1] = 1.0 / max(len(self.base), 1)
            return w if len(self.base) else np.array([1.0])
        sub = self.rng.choice(n, min(n, 60), replace=False)  # pairs on a subset keep this O(60^2)
        Us, yo = U[sub], ys[sub]
        losses = []
        for gp in self.base:
            smp = gp.sample_y(Us, S, random_state=int(self.rng.integers(1 << 30))).T
            losses.append(ranking_losses(smp, yo))
        # target: leave-one-out marginals of the target GP
        K = target.kernel_(U)
        Kinv = np.linalg.inv(K + 1e-8 * np.eye(n))
        alpha = Kinv @ ys
        mu_loo = ys - alpha / np.diag(Kinv)
        sd_loo = np.sqrt(1 / np.diag(Kinv))
        smp = mu_loo[sub][None] + sd_loo[sub][None] * self.rng.standard_normal((S, len(sub)))
        losses.append(ranking_losses(smp, yo))
        L = np.stack(losses)  # (models, S)
        best = L == L.min(0, keepdims=True)
        tie = best / best.sum(0, keepdims=True)  # ties share the win
        w = tie.mean(1)
        # weight dilution: drop base models that are usually worse than the target model
        thr = np.percentile(L[-1], 95)
        drop = np.median(L[:-1], axis=1) > thr
        w[:-1][drop] = 0.0
        return w / w.sum()

    def suggest(self, n: int = 1) -> np.ndarray:
        """Pick the EI maximiser of the weighted ensemble among ``N_CAND`` random candidates.

        Args:
            n: Number of points (random points if there is no data and no base model).

        Returns:
            Points of shape (1, d), or (n, d) when falling back to random points.
        """
        X, y = self.history.arrays
        C = self._candidates(N_CAND)
        UC = self._normalize(C)
        if len(y) < 2 or np.ptp(y) == 0:
            # no usable target data: minimise the mean prediction of the source GPs
            if not self.base:
                return self._candidates(n)
            mu = np.mean([gp.predict(UC) for gp in self.base], 0)
            return C[[int(np.argmin(mu))]]
        Xs, ys_raw = self._subset(X, y)
        U, ys = self._normalize(Xs), _std(ys_raw)
        target = _gp(U, ys, int(self.rng.integers(1 << 30)))
        w = self._weights(U, ys, target)
        mu = np.zeros(len(C)); var = np.zeros(len(C))
        for wi, gp in zip(w, self.base + [target]):
            if wi > 0:
                m, s = gp.predict(UC, return_std=True)
                mu += wi * m; var += wi * s ** 2
        return C[[int(np.argmax(expected_improvement(mu, np.sqrt(var), ys.min())))]]
