
"""Abstract interface shared by Random, GP, LFBO, PFN, MALIBO and MetaGameBO.

Convention: minimisation (the table reports Loss / CVaR with a down arrow).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class History:
    X: list = field(default_factory=list)  # each entry: array of shape (d,)
    y: list = field(default_factory=list)  # each entry: float

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        for xi, yi in zip(np.atleast_2d(X), np.atleast_1d(y)):
            self.X.append(np.asarray(xi, dtype=float))
            self.y.append(float(yi))

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.X), np.asarray(self.y)


class BaseOptimizer(ABC):
    """Common API: (optional) meta_train -> loop of suggest / observe."""

    # Subclasses that learn from related tasks (PFN, MALIBO, MetaGameBO)
    # set this to True and override `meta_train`.
    is_meta: bool = False

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        """bounds: array of shape (d, 2) with [low, high] per dimension."""
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.history = History()

    # ---------- must be implemented ----------
    @abstractmethod
    def suggest(self, n: int = 1) -> np.ndarray:
        """Return the next n query points, shape (n, d), inside `bounds`."""

    # ---------- optional hooks ----------
    def meta_train(self, tasks: Sequence["Task"]) -> None:
        """Offline training on source tasks. No-op for non-meta methods."""

    def _update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Refit / update the internal model after new data. No-op by default
        (Random needs nothing; GP refits; LFBO retrains its classifier; PFN
        and MALIBO just condition on the new context)."""

    def reset(self) -> None:
        """Clear history before a new task (meta-learned weights are kept)."""
        self.history = History()

    # ---------- shared logic ----------
    def observe(self, X: np.ndarray, y: np.ndarray) -> None:
        self.history.add(X, y)
        self._update(*self.history.arrays)

    @property
    def best(self) -> tuple[np.ndarray, float]:
        X, y = self.history.arrays
        i = int(np.argmin(y))
        return X[i], float(y[i])

    def optimize(self, objective: Callable[[np.ndarray], np.ndarray],
                 budget: int, n_init: int = 0) -> History:
        """Standard sequential loop on one task."""
        self.reset()
        if n_init > 0:
            X0 = self._uniform(n_init)
            self.observe(X0, objective(X0))
        while len(self.history.y) < budget:
            X = self.suggest(1)
            self.observe(X, objective(X))
        return self.history

    def _uniform(self, n: int) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * self.rng.random((n, self.dim))

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.bounds[:, 0], self.bounds[:, 1])


@dataclass
class Task:
    """One source/target function for meta-learning."""
    objective: Callable[[np.ndarray], np.ndarray]
    X: Optional[np.ndarray] = None  # optional pre-collected data
    y: Optional[np.ndarray] = None