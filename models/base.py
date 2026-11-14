"""Interface shared by every optimizer in ``models`` (all optimizers minimise).

Attributes:
    N_CAND: Number of candidates used to maximise an acquisition function
        (random search over the box, or a random subset of a discrete pool).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

N_CAND = 5120


@dataclass
class History:
    """Observations collected on one task.

    Attributes:
        X: Query points, each an array of shape (d,).
        y: Observed objective values, one float per query.
    """
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        """Append points ``x`` (n, d) or (d,) with values ``y`` (n,) or scalar."""
        for xi, yi in zip(np.atleast_2d(x), np.atleast_1d(y)):
            self.X.append(np.asarray(xi, dtype=float))
            self.y.append(float(yi))

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` of shapes (n, d) and (n,)."""
        if not self.X:
            return np.zeros((0, 0)), np.zeros(0)
        return np.asarray(self.X), np.asarray(self.y)


@dataclass
class Task:
    """A source task for meta-learning: an objective and/or pre-collected data.

    Attributes:
        objective: Function mapping points (n, d) to values (n,), or None.
        X: Pre-collected points in the search box, shape (n, d).
        y: Values at ``X``, shape (n,).
    """
    objective: Optional[Callable[[np.ndarray], np.ndarray]] = None
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


class BaseOptimizer(ABC):
    """Sequential optimizer: optional ``meta_train``, then ``optimize`` per target task.

    The search space is the box ``bounds`` or, after ``set_pool``, a finite set of
    points (tabular benchmarks): candidates are then drawn from the not yet
    evaluated pool points.

    Args:
        bounds: Search box of shape (d, 2), one ``[low, high]`` row per dimension.
        seed: Seed of the optimizer's random generator.

    Attributes:
        is_meta: True for methods that learn from related tasks.
        name: Display name used in tables and plots.
    """

    is_meta: bool = False
    name: str = "base"

    def __init__(self, bounds: np.ndarray, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.default_rng(seed)
        self.history = History()
        self.pool: Optional[np.ndarray] = None
        self._pool_index: dict = {}
        self._pool_free: Optional[np.ndarray] = None

    # ---------- must be implemented ----------
    @abstractmethod
    def suggest(self, n: int = 1) -> np.ndarray:
        """Return the next ``n`` query points, shape (n, d)."""

    # ---------- optional hooks ----------
    def meta_train(self, tasks: Sequence[Task], **kwargs) -> None:
        """Learn from source tasks with pre-collected ``X``/``y`` (no-op by default)."""

    def _update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Refit the model on all observations ``x`` (n, d), ``y`` (n,) (no-op by default)."""

    def prob_good(self, X: np.ndarray, y: np.ndarray, Xq: np.ndarray, tau: float) -> np.ndarray:
        """Classifier view of the model for the Table-1 metric.

        Given context data ``(X, y)`` of one task, return the predicted probability
        that ``f(Xq) < tau`` (the "good" class of LFBO/MALIBO), shape (m,).
        """
        raise NotImplementedError(f"{type(self).__name__} has no classifier view")

    # ---------- shared logic ----------
    def set_pool(self, pool: Optional[np.ndarray]) -> None:
        """Restrict the search space to the rows of ``pool`` (n_pool, d); None = box."""
        self.pool = None if pool is None else np.asarray(pool, dtype=float)
        self._pool_index = {} if pool is None else {r.tobytes(): i for i, r in enumerate(self.pool)}
        self._pool_free = None if pool is None else np.ones(len(self.pool), bool)

    def reset(self) -> None:
        """Forget the observations of the current task (meta-learned state is kept)."""
        self.history = History()
        if self.pool is not None:
            self._pool_free[:] = True

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        """Record new points ``x`` (n, d) with values ``y`` (n,) and refit."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.pool is not None:
            for r in x:
                i = self._pool_index.get(r.tobytes())
                if i is not None:
                    self._pool_free[i] = False
        self.history.add(x, y)
        self._update(*self.history.arrays)

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Best observed point and value ``(x, y)``."""
        X, y = self.history.arrays
        i = int(np.argmin(y))
        return X[i], float(y[i])

    def optimize(self, objective: Callable[[np.ndarray], np.ndarray], budget: int,
                 n_init: int = 0, init: Optional[np.ndarray] = None,
                 callback: Optional[Callable[[int], None]] = None) -> History:
        """Run sequential optimisation on one task.

        Args:
            objective: Maps points (n, d) to values (n,).
            budget: Total number of evaluations, initial ones included.
            n_init: Number of random initial points (ignored if ``init`` is given).
            init: Fixed initial points, shape (k, d).
            callback: Called with the number of evaluations after each step.

        Returns:
            The history of all evaluations.
        """
        self.reset()
        X0 = init if init is not None else (self._candidates(n_init) if n_init > 0 else None)
        if X0 is not None and len(X0):
            self.observe(X0, objective(X0))
        while len(self.history.y) < budget:
            X = self.suggest(1)
            self.observe(X, objective(X))
            if callback is not None:
                callback(len(self.history.y))
        return self.history

    def _candidates(self, n: int) -> np.ndarray:
        """``n`` random candidates: uniform in the box, or unevaluated pool points."""
        if self.pool is None:
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            return lo + (hi - lo) * self.rng.random((n, self.dim))
        free = np.flatnonzero(self._pool_free)
        if len(free) == 0:
            free = np.arange(len(self.pool))
        idx = self.rng.choice(free, size=min(n, len(free)), replace=False)
        return self.pool[idx]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Map points of the box to the unit cube."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X, dtype=float) - lo) / np.where(hi > lo, hi - lo, 1.0)


def cvar(values: np.ndarray, alpha: float = 0.8) -> float:
    """Empirical CVaR as in the author's ``meta_learning/trainer.py``.

    Sort descending and average the ``k = max(1, int(n * (1 - alpha)))`` largest
    values (NaNs dropped). With alpha = 0.8 this is about the worst 20%.
    """
    v = np.asarray(values, dtype=float)
    v = np.sort(v[~np.isnan(v)])[::-1]
    k = max(1, int(len(v) * (1 - alpha)))
    return float(v[:k].mean())
