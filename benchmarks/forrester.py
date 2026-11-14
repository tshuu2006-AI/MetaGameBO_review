"""Vectorised Forrester task family, identical to the upstream benchmark.

    f(x) = a (6x-2)^2 sin(12x-4) + b (x-0.5) - c,  x in [0, 1]

with a ~ U[0.2, 3], b ~ U[-5, 15], c ~ U[-5, 5] (upstream `Forrester` descriptors).

Attributes:
    BOUNDS: Search box, shape (1, 2).
    IN_DIST: Parameter ranges of the upstream task distribution.
    SHIFTED: Shifted ranges (larger b) for out-of-distribution targets;
        added in this remake, not part of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.base import Task

BOUNDS = np.array([[0.0, 1.0]])
IN_DIST = {"a": (0.2, 3.0), "b": (-5.0, 15.0), "c": (-5.0, 5.0)}
# Distribution shift: much stronger linear trend -> global optimum often moves
# to the left basin (x ~ 0.14), which meta-training tasks rarely show.
SHIFTED = {"a": (0.2, 3.0), "b": (15.0, 30.0), "c": (-5.0, 5.0)} # Nhóm thêm vào để đánh giá khi task đích lệch phân phối
_GRID = np.linspace(0.0, 1.0, 2001)


def forrester(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Evaluate the Forrester function.

    Args:
        x: Inputs in [0, 1], any shape (flattened).
        a: Amplitude of the high-fidelity term.
        b: Slope of the linear term.
        c: Constant offset.

    Returns:
        Values, shape (n,).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return a * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + b * (x - 0.5) - c


@dataclass
class ForresterTask:
    """One Forrester task with fixed parameters; callable as the objective.

    Attributes:
        a: Amplitude of the high-fidelity term.
        b: Slope of the linear term.
        c: Constant offset.
    """
    a: float
    b: float
    c: float

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the task at ``X`` (shape (n, 1)); returns shape (n,)."""
        return forrester(X, self.a, self.b, self.c)

    @property
    def f_range(self) -> tuple[float, float]:
        """Return ``(f_min, f_max)`` on a 2001-point grid over [0, 1]."""
        f = self(_GRID)
        return float(f.min()), float(f.max())

    def normalized_regret(self, y_seen: np.ndarray) -> np.ndarray:
        """Normalised regret curve (MALIBO protocol).

        Args:
            y_seen: Observed values in evaluation order, shape (n,).

        Returns:
            Running min of ``(y - f_min) / (f_max - f_min)``, shape (n,), in [0, 1].
        """
        lo, hi = self.f_range
        return (np.minimum.accumulate(np.asarray(y_seen)) - lo) / (hi - lo)


def sample_tasks(n: int, seed: int, ranges: dict = IN_DIST) -> list[ForresterTask]:
    """Sample tasks with parameters drawn uniformly from ``ranges``.

    Args:
        n: Number of tasks.
        seed: Random seed.
        ranges: Dict mapping ``"a"``, ``"b"``, ``"c"`` to ``(low, high)``.

    Returns:
        List of ``n`` tasks.
    """
    rng = np.random.default_rng(seed)
    return [ForresterTask(*(rng.uniform(*ranges[k]) for k in "abc")) for _ in range(n)]


def make_meta_tasks(n_tasks: int = 200, n_per_task: int = 32, seed: int = 0) -> list[Task]:
    """Build meta-training data from in-distribution tasks.

    Args:
        n_tasks: Number of meta tasks.
        n_per_task: Uniform random evaluations per task.
        seed: Seed for the task parameters (inputs use ``seed + 10000``).

    Returns:
        List of ``Task`` with ``X`` of shape (n_per_task, 1) and ``y`` of shape (n_per_task,).
    """
    rng = np.random.default_rng(seed + 10_000)
    out = []
    for t in sample_tasks(n_tasks, seed):
        X = rng.random((n_per_task, 1))
        out.append(Task(objective=t, X=X, y=t(X)))
    return out
