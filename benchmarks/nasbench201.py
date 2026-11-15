"""NAS-Bench-201 (Dong and Yang 2020): 15,625 cells (6 edges x 5 operations) on
CIFAR-10, CIFAR-100 and ImageNet16-120. Objective: validation error after 200
epochs, averaged over seeds. Data: Syne Tune blackbox repository (folder
``nasbench201``) in ``benchmarks/data/nasbench201``.

Each edge's operation is encoded as its index / 4 (6 dims), because the
pre-trained PFN accepts at most 18 inputs (a one-hot code would need 30).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from benchmarks.common import Problem, leave_one_task_out

DATA = Path(__file__).resolve().parent / "data" / "nasbench201"
TASKS = ["cifar10", "cifar100", "ImageNet16-120"]
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


def load(cache: Path = DATA / "compact.npz"):
    """Return ``(X, Y)``: encoded cells (n, 6) and final validation error (3, n)."""
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["Y"]
    import pandas as pd
    hp = pd.read_parquet(DATA / "hyperparameters.parquet")
    X = np.column_stack([hp[c].map(OPS.index).to_numpy() / (len(OPS) - 1) for c in hp.columns]).astype(float)
    ev = np.load(DATA / "objectives_evaluations.npy", mmap_mode="r")
    Y = np.asarray(ev[:, :, :, -1, 0], dtype=float)
    Y = np.nanmean(Y, axis=2)
    np.savez(cache, X=X, Y=Y)
    return X, Y


def load_seeds(cache: Path = DATA / "compact_seeds.npz") -> np.ndarray:
    """Final validation objective per training seed, shape (tasks, seeds, n)."""
    if cache.exists():
        return np.load(cache)["Ys"]
    ev = np.load(DATA / "objectives_evaluations.npy", mmap_mode="r")
    Ys = np.asarray(ev[:, :, :, -1, 0], dtype=float).transpose(0, 2, 1)
    np.savez(cache, Ys=Ys)
    return Ys


def problems(seed: int, n_meta: int = 512, noisy: bool = False) -> List[Problem]:
    """Leave-one-task-out problems for one seed (``noisy``: observe one random training seed)."""
    X, Y = load()
    return leave_one_task_out("nasbench201", X, Y, TASKS, n_meta, seed, load_seeds() if noisy else None)
