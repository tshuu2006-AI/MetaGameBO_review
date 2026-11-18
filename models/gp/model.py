"""GP-EI baseline: Gaussian process with Expected Improvement (Snoek et al. 2012)."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from models.base import N_CAND, BaseOptimizer


def fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0, restarts: int = 2) -> GaussianProcessRegressor:
    """Fit a GP with a Matern-5/2 ARD kernel plus a noise term.

    Args:
        X: Inputs in the unit cube, shape (n, d).
        y: Targets, shape (n,) (normalised inside the GP).
        seed: Seed of the hyperparameter restarts.
        restarts: Number of extra optimizer restarts of the marginal likelihood.

    Returns:
        The fitted ``GaussianProcessRegressor``.
    """
    d = X.shape[1]
    k = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), nu=2.5,
                                     length_scale_bounds=(1e-3, 1e2)) \
        + WhiteKernel(1e-4, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=restarts, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)
    return gp


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """Expected Improvement for minimisation under a Gaussian predictive.

    Args:
        mu: Predictive means, shape (m,).
        sd: Predictive standard deviations, shape (m,).
        best: Incumbent value.

    Returns:
        EI values, shape (m,).
    """
    sd = np.maximum(sd, 1e-12)
    imp = best - mu
    return imp * norm.cdf(imp / sd) + sd * norm.pdf(imp / sd)


class GPEI(BaseOptimizer):
    """GP (Matern-5/2 ARD + noise) with Expected Improvement.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed.
        max_points: Fit the GP on at most this many observations (the best half
            plus a random rest) to bound the cubic cost on long runs.
    """

    name = "GP"

    def __init__(self, bounds, seed: Optional[int] = None, max_points: int = 300):
        super().__init__(bounds, seed)
        self.max_points = max_points

    def _subset(self, X, y):
        """Keep the best half of ``max_points`` and fill the rest at random.

        Args:
            X: Observed points, shape (n, d).
            y: Observed values, shape (n,).

        Returns:
            ``(X, y)`` with at most ``max_points`` rows.
        """
        if len(y) <= self.max_points:
            return X, y
        order = np.argsort(y)
        keep = order[: self.max_points // 2]
        rest = self.rng.choice(order[self.max_points // 2:], self.max_points - len(keep), replace=False)
        idx = np.concatenate([keep, rest])
        return X[idx], y[idx]

    def suggest(self, n: int = 1) -> np.ndarray:
        """Pick the EI maximiser among ``N_CAND`` random candidates.

        Args:
            n: Number of points (random points while fewer than 2 distinct values are observed).

        Returns:
            Points of shape (1, d), or (n, d) when falling back to random points.
        """
        X, y = self.history.arrays
        if len(y) < 2 or np.ptp(y) == 0:
            return self._candidates(n)
        Xs, ys = self._subset(X, y)
        gp = fit_gp(self._normalize(Xs), ys, seed=int(self.rng.integers(1 << 30)))
        C = self._candidates(N_CAND)
        mu, sd = gp.predict(self._normalize(C), return_std=True)
        return C[[int(np.argmax(expected_improvement(mu, sd, y.min())))]]

    def prob_good(self, X, y, Xq, tau):
        """P(f(x) < tau) under the GP posterior fitted on the context data.

        Args:
            X: Context points, shape (n, d).
            y: Context values, shape (n,).
            Xq: Query points, shape (m, d).
            tau: Threshold of the "good" class.

        Returns:
            Probabilities of shape (m,); 0.5 everywhere with fewer than 2 distinct values.
        """
        if len(y) < 2 or np.ptp(y) == 0:
            return np.full(len(Xq), 0.5)
        gp = fit_gp(self._normalize(X), np.asarray(y), seed=int(self.rng.integers(1 << 30)))
        mu, sd = gp.predict(self._normalize(Xq), return_std=True)
        return norm.cdf((tau - mu) / np.maximum(sd, 1e-9))
