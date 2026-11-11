"""Abstract optimizer interface shared by all BO methods in this project.

All optimizers minimise the objective.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class History:
    """Observations collected on one task.

    Attributes:
        X: Query points, each an array of shape (d,).
        y: Observed objective values, one float per query.
    """
    X: list = field(default_factory=list)  # each entry: array of shape (d,)
    y: list = field(default_factory=list)  # each entry: float

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        """Append observations.

        Args:
            X: Points of shape (n, d), or a single point of shape (d,).
            y: Values of shape (n,), or a scalar.
        """
        for xi, yi in zip(np.atleast_2d(X), np.atleast_1d(y)):
            self.X.append(np.asarray(xi, dtype=float))
            self.y.append(float(yi))

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the history as arrays ``(X, y)`` of shapes (n, d) and (n,)."""
        return np.asarray(self.X), np.asarray(self.y)


class BaseOptimizer(ABC):
    """Base class for sequential optimizers.

    Usage: optionally ``meta_train`` on source tasks, then ``optimize`` (or a
    manual ``suggest`` / ``observe`` loop) on each target task.

    Args:
        bounds: Search box of shape (d, 2), one ``[low, high]`` row per dimension.
        seed: Seed for the optimizer's random generator.

    Attributes:
        is_meta: True for methods that learn from related tasks.
    """

    # Subclasses that learn from related tasks (PFN, MALIBO, MetaGameBO)
    # set this to True and override `meta_train`.
    is_meta: bool = False

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.history = History()

    # ---------- must be implemented ----------
    @abstractmethod
    def suggest(self, n: int = 1) -> np.ndarray:
        """Propose the next query points.

        Args:
            n: Number of points to propose.

        Returns:
            Array of shape (n, d) inside ``bounds``.
        """

    # ---------- optional hooks ----------
    def meta_train(self, tasks: Sequence["Task"]) -> None:
        """Train offline on source tasks (no-op for non-meta methods).

        Args:
            tasks: Source tasks with pre-collected ``X`` and ``y``.
        """

    def _update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the internal model after new data (no-op by default).

        Args:
            X: All observed points on the current task, shape (n, d).
            y: All observed values, shape (n,).
        """

    def reset(self) -> None:
        """Clear the history before a new task; meta-learned weights are kept."""
        self.history = History()

    # ---------- shared logic ----------
    def observe(self, X: np.ndarray, y: np.ndarray) -> None:
        """Record new observations and update the model.

        Args:
            X: New points, shape (n, d).
            y: New values, shape (n,).
        """
        self.history.add(X, y)
        self._update(*self.history.arrays)

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Return the best observed point and its value ``(x, y)``."""
        X, y = self.history.arrays
        i = int(np.argmin(y))
        return X[i], float(y[i])

    def optimize(self, objective: Callable[[np.ndarray], np.ndarray],
                 budget: int, n_init: int = 0) -> History:
        """Run sequential optimisation on one task.

        Args:
            objective: Function mapping points (n, d) to values (n,).
            budget: Total number of evaluations, including the initial ones.
            n_init: Number of uniform random initial points.

        Returns:
            The history of all evaluations on this task.
        """
        self.reset()
        if n_init > 0:
            X0 = self._uniform(n_init)
            self.observe(X0, objective(X0))
        while len(self.history.y) < budget:
            X = self.suggest(1)
            self.observe(X, objective(X))
        return self.history

    def _uniform(self, n: int) -> np.ndarray:
        """Sample ``n`` points uniformly in ``bounds``; returns shape (n, d)."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * self.rng.random((n, self.dim))

    def _clip(self, X: np.ndarray) -> np.ndarray:
        """Clip points ``X`` of shape (n, d) to ``bounds``."""
        return np.clip(X, self.bounds[:, 0], self.bounds[:, 1])


@dataclass
class Task:
    """One source or target function for meta-learning.

    Attributes:
        objective: Function mapping points (n, d) to values (n,).
        X: Optional pre-collected points, shape (n, d).
        y: Optional values at ``X``, shape (n,).
    """
    objective: Callable[[np.ndarray], np.ndarray]
    X: Optional[np.ndarray] = None  # optional pre-collected data
    y: Optional[np.ndarray] = None
