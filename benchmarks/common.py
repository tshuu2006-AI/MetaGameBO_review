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
        y: Objective values to minimise (mean over training seeds), shape (n,).
        y_seeds: Optional per-seed values, shape (S, n). If given, the table is
            *noisy*: each evaluation returns the value of one random training
            seed, and ``regret`` scores the incumbent (best *observed* point)
            by its mean value ``y``. Regret then stays > 0 when the noise makes
            a worse configuration look best.
        seed: Seed of the noise.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, y_seeds: Optional[np.ndarray] = None, seed: int = 0):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.y_seeds = None if y_seeds is None else np.asarray(y_seeds, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.true_seen: List[float] = []
        self.index = {r.tobytes(): i for i, r in enumerate(self.X)}

    def __call__(self, Q: np.ndarray) -> np.ndarray:
        out = []
        for q in np.atleast_2d(np.asarray(Q, dtype=float)):
            i = self.index.get(np.ascontiguousarray(q).tobytes())
            if i is None:  # not a pool row: nearest configuration
                i = int(np.argmin(((self.X - q) ** 2).sum(1)))
            if self.y_seeds is None:
                out.append(self.y[i])
            else:
                vals = self.y_seeds[:, i]
                vals = vals[~np.isnan(vals)]
                out.append(self.rng.choice(vals) if len(vals) else self.y[i])
                self.true_seen.append(self.y[i])
        return np.array(out)

    def regret(self, y_seen: np.ndarray) -> np.ndarray:
        """Normalised simple regret ``(y(incumbent) - min) / (max - min)`` over the whole table.

        Noise-free: the incumbent value is the running min of ``y_seen``.
        Noisy: the incumbent is the running argmin of the observed values and is
        scored by its mean value (the values seen by this table, in order).
        """
        lo, hi = self.y.min(), self.y.max()
        y_seen = np.asarray(y_seen, float)
        if self.y_seeds is None:
            return (np.minimum.accumulate(y_seen) - lo) / (hi - lo)
        true = np.asarray(self.true_seen[-len(y_seen):], float)
        inc = np.zeros(len(y_seen), int)
        for t in range(1, len(y_seen)):
            inc[t] = t if y_seen[t] < y_seen[inc[t - 1]] else inc[t - 1]
        return (true[inc] - lo) / (hi - lo)


def leave_one_task_out(name: str, X: np.ndarray, Y: np.ndarray, task_names: List[str],
                       n_meta: int, seed: int, Y_seeds: Optional[np.ndarray] = None) -> List[Problem]:
    """Tabular problems where each task is the target once and the others are meta tasks.

    Args:
        name: Benchmark name.
        X: Encoded configurations (unit cube), shape (n, d).
        Y: Objective per task, shape (T, n).
        task_names: Names of the T tasks.
        n_meta: Random configurations sampled per meta task (paper: 512).
        seed: Seed of the meta-data sampling.
        Y_seeds: Optional per-seed objective, shape (T, S, n): noisy targets
            (see ``LookupTable``); meta data always uses the mean ``Y``.
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
        table = LookupTable(X, Y[t], None if Y_seeds is None else Y_seeds[t], seed=1000 * seed + t)
        probs.append(Problem(f"{name}/{tname}", bounds, table, table.regret, meta, pool=table.X))
    return probs
