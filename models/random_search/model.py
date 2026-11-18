"""Random search baseline (Bergstra and Bengio 2012)."""
from __future__ import annotations

import numpy as np

from models.base import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """Uniform random search over the box, or over the unevaluated pool points.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed.
    """

    name = "Random"

    def suggest(self, n: int = 1) -> np.ndarray:
        """Draw random points.

        Args:
            n: Number of points.

        Returns:
            Points of shape (n, d).
        """
        return self._candidates(n)

    def prob_good(self, X, y, Xq, tau):
        """Constant classifier: the fraction of context points with ``y < tau``.

        Args:
            X: Context points, shape (n, d) (unused).
            y: Context values, shape (n,).
            Xq: Query points, shape (m, d).
            tau: Threshold of the "good" class.

        Returns:
            The same probability for every query, clipped to [1e-3, 1 - 1e-3], shape (m,).
        """
        p = np.mean(np.asarray(y) < tau) if len(y) else 0.5
        return np.full(len(Xq), np.clip(p, 1e-3, 1 - 1e-3))
