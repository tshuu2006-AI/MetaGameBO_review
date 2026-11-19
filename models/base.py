"""Interface shared by every optimizer in ``models`` (all optimizers minimize).

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
    """Observations collected on one task, in evaluation order.

    Attributes:
        X: Query points, each an array of shape (d,).
        y: Observed objective values, one float per query.
    """
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        """Append one or several observations.

        Args:
            x: Points, shape (n, d) or (d,).
            y: Values, shape (n,) or a scalar.
        """
        for xi, yi in zip(np.atleast_2d(x), np.atleast_1d(y)):
            self.X.append(np.asarray(xi, dtype=float))
            self.y.append(float(yi))

    @property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """All observations as arrays.

        Returns:
            ``(X, y)`` of shapes (n, d) and (n,); empty arrays when nothing is observed.
        """
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
    """Sequential optimizer: optional ``meta_train``, then ``optimize`` on each target task.

    The search space is the box ``bounds`` or, after ``set_pool``, the not yet
    evaluated rows of a finite pool (tabular benchmarks).

    Args:
        bounds: Search box of hyperparams of shape (d, 2), one ``[low, high]`` row per dimension.
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
        # discrete search space: the pool, a row -> index map and a "not yet evaluated" mask
        self.pool: Optional[np.ndarray] = None
        self._pool_index: dict = {}
        self._pool_free: Optional[np.ndarray] = None

    @abstractmethod
    def suggest(self, n: int = 1) -> np.ndarray:
        """Propose the next query points.

        Args:
            n: Number of points.

        Returns:
            Points of shape (n, d).
        """

    def meta_train(self, tasks: Sequence[Task], **kwargs) -> None:
        """Learn from source tasks with pre-collected data (no-op for non-meta methods).

        Args:
            tasks: Source tasks with ``X``/``y`` filled in.
            **kwargs: Method-specific options.
        """

    def _update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Refit the model on all observations of the current task (no-op by default).

        Args:
            x: All observed points, shape (n, d).
            y: Their values, shape (n,).
        """

    def prob_good(self, x: np.ndarray, y: np.ndarray, Xq: np.ndarray, tau: float) -> np.ndarray:
        """Classifier view of the model, used by the Table-1 metric.

        Args:
            x: Context points of one task, shape (n, d).
            y: Context values, shape (n,).
            Xq: Query points, shape (m, d).
            tau: Threshold of the "good" class (``f(x) < tau``).

        Returns:
            Predicted probability that ``f(Xq) < tau``, shape (m,).
        """
        raise NotImplementedError(f"{type(self).__name__} has no classifier view")

    # ---------- shared logic ----------
    def set_pool(self, pool: Optional[np.ndarray]) -> None:
        """Restrict the search space to a finite set of points and mark them all unevaluated.

        Args:
            pool: Points of shape (n_pool, d), or None for the continuous box.
        """
        self.pool = None if pool is None else np.asarray(pool, dtype=float)
        self._pool_index = {} if pool is None else {r.tobytes(): i for i, r in enumerate(self.pool)}
        self._pool_free = None if pool is None else np.ones(len(self.pool), bool)

    def reset(self) -> None:
        """Forget the observations of the current task; the meta-learned state is kept."""
        self.history = History()
        if self.pool is not None:
            self._pool_free[:] = True

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        """Record new observations, mark them as evaluated in the pool and refit the model.

        Args:
            x: Points, shape (n, d).
            y: Values, shape (n,).
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.pool is not None:
            for r in x:
                i = self._pool_index.get(r.tobytes())
                if i is not None:  # points outside the pool (e.g. fixed inits) are ignored
                    self._pool_free[i] = False
        self.history.add(x, y)
        self._update(*self.history.arrays)

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Incumbent of the current task.

        Returns:
            The best observed point and its value ``(x, y)``.
        """
        X, y = self.history.arrays
        i = int(np.argmin(y))
        return X[i], float(y[i])

    def optimize(self, objective: Callable[[np.ndarray], np.ndarray], budget: int,
                 n_init: int = 0, init: Optional[np.ndarray] = None,
                 callback: Optional[Callable[[int], None]] = None, resume: bool = False) -> History:
        """Run sequential optimization on one task.

        Args:
            objective: Maps points (n, d) to values (n,).
            budget: Total number of evaluations, initial ones included.
            n_init: Number of random initial points (ignored if ``init`` is given).
            init: Fixed initial points, shape (k, d).
            callback: Called with the number of evaluations after each step.
            resume: Continue the current history (e.g. restored from a checkpoint)
                instead of starting a new run.

        Returns:
            The history of all evaluations.
        """
        if resume and len(self.history.y):
            return self._loop(objective, budget, callback)
        self.reset()
        X0 = init if init is not None else (self._candidates(n_init) if n_init > 0 else None)
        if X0 is not None and len(X0):
            self.observe(X0, objective(X0))
        return self._loop(objective, budget, callback)

    def _loop(self, objective, budget, callback) -> History:
        """Suggest-evaluate-observe until the budget is spent or the pool is exhausted.

        Args:
            objective: Maps points (n, d) to values (n,).
            budget: Total number of evaluations.
            callback: Called with the number of evaluations after each step, or None.

        Returns:
            The history of all evaluations.
        """
        while len(self.history.y) < budget:
            if self.pool is not None and not self._pool_free.any():
                break  # every pool point is evaluated (small tabular task): the run ends early
            X = self.suggest(1)
            self.observe(X, objective(X))
            if callback is not None:
                callback(len(self.history.y))
        return self.history

    def _candidates(self, n: int) -> np.ndarray:
        """Draw random candidates: uniform in the box, or unevaluated pool points.

        Args:
            n: Number of candidates (fewer if the pool has fewer free points).

        Returns:
            Candidates of shape (n, d).
        """
        if self.pool is None:
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            return lo + (hi - lo) * self.rng.random((n, self.dim))
        free = np.flatnonzero(self._pool_free)
        if len(free) == 0:  # never happens inside ``optimize`` (the loop stops first)
            free = np.arange(len(self.pool))
        idx = self.rng.choice(free, size=min(n, len(free)), replace=False)
        return self.pool[idx]

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Map points of the box to the unit cube.

        Args:
            X: Points in the box, shape (n, d).

        Returns:
            Points in [0, 1]^d, shape (n, d); degenerate dimensions are left unscaled.
        """
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X, dtype=float) - lo) / np.where(hi > lo, hi - lo, 1.0)


def cvar(values: np.ndarray, alpha: float = 0.8) -> float:
    """Empirical CVaR as in the author's ``meta_learning/trainer.py``: the mean of the worst ``1 - alpha`` share.

    Args:
        values: Losses (NaNs are dropped).
        alpha: Confidence level; the ``max(1, int(n * (1 - alpha)))`` largest values are averaged.

    Returns:
        The CVaR as a float.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)] # drop NaNs
    v = np.sort(v)[::-1]  # descending

    # Choose hard tasks
    k = max(1, int(len(v) * (1 - alpha)))
    return float(v[:k].mean())
