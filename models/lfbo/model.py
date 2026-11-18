"""LFBO: likelihood-free BO with a gradient-boosting classifier (Song et al. 2022)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from models.base import N_CAND, BaseOptimizer
from models.lfbo.weighting import reweight_samples


class LFBO(BaseOptimizer):
    """LFBO without meta-learning: the acquisition is P(good | x) of an EI-weighted classifier.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed.
        gamma: Quantile threshold that labels observations as good.
    """

    name = "LFBO"

    def __init__(self, bounds, seed: Optional[int] = None, gamma: float = 1 / 3):
        super().__init__(bounds, seed)
        self.gamma = gamma

    def _fit(self, X, y):
        """Fit the weighted gradient-boosting classifier of Eq. (3).

        Args:
            X: Observed points, shape (n, d).
            y: Observed values, shape (n,).

        Returns:
            The fitted classifier, or None if a class has fewer than 2 points.
        """
        if len(y) < 2:
            return None
        x, lab, w = reweight_samples(X, y, self.gamma)
        if (lab == 0).sum() < 2 or (lab == 1).sum() < 2:
            return None
        clf = GradientBoostingClassifier(random_state=int(self.rng.integers(1 << 30)))
        return clf.fit(x, lab, sample_weight=w)

    def suggest(self, n: int = 1) -> np.ndarray:
        """Pick the candidate with the highest P(good | x) among ``N_CAND`` random candidates.

        Args:
            n: Number of points (random points while the classifier cannot be fitted).

        Returns:
            Points of shape (1, d), or (n, d) when falling back to random points.
        """
        clf = self._fit(*self.history.arrays)
        if clf is None:
            return self._candidates(n)
        C = self._candidates(N_CAND)
        return C[[int(np.argmax(clf.predict_proba(C)[:, 1]))]]

    def prob_good(self, X, y, Xq, tau):
        """P(good | x) of a classifier trained on the context data only.

        Args:
            X: Context points, shape (n, d).
            y: Context values, shape (n,).
            Xq: Query points, shape (m, d).
            tau: Threshold of the "good" class (the classifier uses its own gamma-quantile).

        Returns:
            Probabilities of shape (m,); 0.5 everywhere if the classifier cannot be fitted.
        """
        clf = self._fit(X, y)
        return np.full(len(Xq), 0.5) if clf is None else clf.predict_proba(Xq)[:, 1]
