
"""Abstract interface shared by Random, GP, LFBO, PFN, MALIBO and MetaGameBO.

Convention: minimisation (the table reports Loss / CVaR with a down arrow).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable, Optional, Sequence
import numpy as np


def validate_budget(budget: int, n_init: int) -> None:
    """Validate a positive total evaluation budget and its initialization count."""
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
        raise ValueError("budget must be a positive integer (including initialization)")
    if isinstance(n_init, bool) or not isinstance(n_init, Integral) or not 0 <= n_init <= budget:
        raise ValueError("n_init must be an integer satisfying 0 <= n_init <= budget")


def validate_points(X, bounds=None, expected_n=None) -> np.ndarray:
    """Return a copied finite (n,d) array; reject invalid shapes or bounds violations."""
    points = np.asarray(X, dtype=np.float64)
    if points.ndim != 2 or min(points.shape) == 0:
        raise ValueError("X must have nonempty shape (n,d)")
    if expected_n is not None and points.shape[0] != expected_n:
        raise ValueError(f"Expected {expected_n} points, got {len(points)}")
    if not np.isfinite(points).all():
        raise ValueError("X must contain only finite numbers")
    if bounds is not None:
        if points.shape[1] != len(bounds):
            raise ValueError("X dimension does not match bounds")
        if np.any(points < bounds[:, 0]) or np.any(points > bounds[:, 1]):
            raise ValueError("Proposed points are outside bounds; clipping is not permitted")
    return points.copy()


def validate_values(y, n: int) -> np.ndarray:
    """Normalize finite objective outputs of shape (n,) or (n,1) to (n,)."""
    values = np.asarray(y, dtype=np.float64)
    if values.shape == (n, 1):
        values = values[:, 0]
    if values.shape != (n,) or n == 0 or not np.isfinite(values).all():
        raise ValueError(f"y must contain {n} finite values with shape (n,) or (n,1)")
    return values.copy()


@dataclass
class History:
    X: list = field(default_factory=list)  # each entry: array of shape (d,)
    y: list = field(default_factory=list)  # each entry: float

    def add(self, X: np.ndarray, y: np.ndarray) -> None:
        """Append matching finite observations atomically, preserving FIFO order."""
        points = validate_points(X)
        values = validate_values(y, len(points))
        if self.X and len(self.X[0]) != points.shape[1]:
            raise ValueError("Observation dimension changed within a history")
        for xi, yi in zip(points, values):
            self.X.append(xi.copy())
            self.y.append(float(yi))

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of X and y, with empty X represented by shape (0,0)."""
        X = np.asarray(self.X, dtype=float) if self.X else np.empty((0, 0))
        return X, np.asarray(self.y, dtype=float)


class BaseOptimizer(ABC):
    """Common API: (optional) meta_train -> loop of suggest / observe."""

    # Subclasses that learn from related tasks (PFN, MALIBO, MetaGameBO)
    # set this to True and override `meta_train`.
    is_meta: bool = False

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        """bounds: array of shape (d, 2) with [low, high] per dimension."""
        self.bounds = np.array(bounds, dtype=float, copy=True)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or not len(self.bounds)
                or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError("bounds must be finite shape (d,2) with lower < upper")
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.history = History()
        self.phases = []

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
        """Clear task history/state, keeping learned weights and the current RNG stream.

        Subclasses clear task-specific caches in _reset_task_state; create a new
        seeded optimizer per repeat to restart its RNG deterministically.
        """
        self.history = History()
        self.phases = []
        self._reset_task_state()

    def _reset_task_state(self) -> None:
        """Optional subclass hook to clear task caches without removing learned weights."""

    # ---------- shared logic ----------
    def observe(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate and append observed values, then update using observed history only."""
        points = validate_points(X, self.bounds)
        values = validate_values(y, len(points))
        self.history.add(points, values)
        self._update(*self.history.arrays)

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Return the lowest observed objective and its point; fail clearly if empty."""
        if not self.history.y:
            raise ValueError("No observations yet: best is undefined for an empty history")
        X, y = self.history.arrays
        i = int(np.argmin(y))
        return X[i], float(y[i])

    def optimize(self, objective: Callable[[np.ndarray], np.ndarray],
                 budget: int, n_init: int = 0) -> History:
        """Run a total budget including initialization, returning validated observations."""
        validate_budget(budget, n_init)
        self.reset()
        if n_init > 0:
            X0 = self._uniform(n_init)
            self.observe(X0, objective(X0))
            self.phases.extend(["initial"] * n_init)
        while len(self.history.y) < budget:
            X = validate_points(self.suggest(1), self.bounds, expected_n=1)
            self.observe(X, objective(X))
            self.phases.append("sequential")
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
