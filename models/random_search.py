"""Uniform Random Search, independently written as a pipeline sanity baseline."""
from numbers import Integral

import numpy as np

from models.base_model import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """Suggest uniform points; observations require no model fitting."""

    def suggest(self, n: int = 1) -> np.ndarray:
        """Return n independent uniform points of shape (n,d) inside bounds."""
        if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("n must be a positive integer")
        return self._uniform(n)
