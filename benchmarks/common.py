"""Shared pieces of the benchmarks: a BO problem and tabular (lookup) tasks.

A ``Problem`` is one target task with its meta-training tasks:
``meta_tasks`` (``models.base.Task`` with ``X``, ``y``), the target ``objective``,
the search ``bounds`` (and ``pool`` for tabular benchmarks), optional fixed
``init`` points, and ``regret(y_seen)`` -> normalised simple-regret curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from models.base import Task


@dataclass
class Problem:
    """One target task of a benchmark (see module docstring)."""
    name: str
    bounds: np.ndarray
    objective: Callable[[np.ndarray], np.ndarray]
    regret: Callable[[np.ndarray], np.ndarray]
    meta_tasks: List[Task]
    pool: Optional[np.ndarray] = None
    init: Optional[np.ndarray] = None


class LookupTable:
    """Objective of a tabular task: exact lookup of pool rows.

    Args:
        X: Encoded configurations (pool), shape (n, d).
        y: Objective values to minimise, shape (n,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.index = {r.tobytes(): i for i, r in enumerate(self.X)}

    def __call__(self, Q: np.ndarray) -> np.ndarray:
        out = []
        for q in np.atleast_2d(np.asarray(Q, dtype=float)):
            i = self.index.get(np.ascontiguousarray(q).tobytes())
            if i is None:  # not a pool row: nearest configuration
                i = int(np.argmin(((self.X - q) ** 2).sum(1)))
            out.append(self.y[i])
        return np.array(out)

    def regret(self, y_seen: np.ndarray) -> np.ndarray:
        """Running min of ``(y - min) / (max - min)`` over the whole table."""
        lo, hi = self.y.min(), self.y.max()
        return (np.minimum.accumulate(np.asarray(y_seen, float)) - lo) / (hi - lo)


def leave_one_task_out(name: str, X: np.ndarray, Y: np.ndarray, task_names: List[str],
                       n_meta: int, seed: int) -> List[Problem]:
    """Tabular problems where each task is the target once and the others are meta tasks.

    Args:
        name: Benchmark name.
        X: Encoded configurations (unit cube), shape (n, d).
        Y: Objective per task, shape (T, n).
        task_names: Names of the T tasks.
        n_meta: Random configurations sampled per meta task (paper: 512).
        seed: Seed of the meta-data sampling.
    """
    rng = np.random.default_rng(seed)
    bounds = np.column_stack([np.zeros(X.shape[1]), np.ones(X.shape[1])])
    probs = []
    for t, tname in enumerate(task_names):
        meta = []
        for s in range(len(task_names)):
            if s != t:
                idx = rng.choice(len(X), n_meta, replace=False)
                meta.append(Task(objective=None, X=X[idx], y=Y[s, idx]))
        table = LookupTable(X, Y[t])
        probs.append(Problem(f"{name}/{tname}", bounds, table, table.regret, meta, pool=table.X))
    return probs
