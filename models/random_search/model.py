"""Random search baseline (Bergstra and Bengio 2012)."""
from __future__ import annotations

import numpy as np

from models.base import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """Uniform random search over the box (or over the unevaluated pool points)."""

    name = "Random"

    def suggest(self, n: int = 1) -> np.ndarray:
        """``n`` random points, shape (n, d)."""
        return self._candidates(n)

    def prob_good(self, X, y, Xq, tau):
        """No model: the constant P(good) = fraction of context points with ``y < tau``."""
        p = np.mean(np.asarray(y) < tau) if len(y) else 0.5
        return np.full(len(Xq), np.clip(p, 1e-3, 1 - 1e-3))
