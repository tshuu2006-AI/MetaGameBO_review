"""HPOBench FCNet tabular benchmark (Klein and Hutter 2019): a two-layer MLP on
4 UCI regression datasets (protein, naval, parkinsons, slice), 9 hyperparameters,
62,208 configurations. Objective: validation MSE after 100 epochs, averaged
over the 4 seeds. Data: Syne Tune blackbox repository (HuggingFace
``synetune/blackbox-repository``, folder ``fcnet``) in ``benchmarks/data/fcnet``.

Every hyperparameter is encoded as the rank of its value in [0, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from benchmarks.common import Problem, leave_one_task_out

DATA = Path(__file__).resolve().parent / "data" / "fcnet"
TASKS = ["protein_structure", "naval_propulsion", "parkinsons_telemonitoring", "slice_localization"]


def encode_ranks(df) -> np.ndarray:
    """Encode every column as the rank of its value among the sorted unique values, in [0, 1]."""
    cols = []
    for c in df.columns:
        v = df[c].to_numpy()
        try:
            u = np.unique(v.astype(float)); v = v.astype(float)
        except (TypeError, ValueError):
            u = np.unique(v.astype(str)); v = v.astype(str)
        cols.append(np.searchsorted(u, v) / max(len(u) - 1, 1))
    return np.column_stack(cols).astype(float)


def load(cache: Path = DATA / "compact.npz"):
    """Return ``(X, Y)``: encoded configurations (n, 9) and final validation loss (4, n)."""
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["Y"]
    import pandas as pd
    X = encode_ranks(pd.read_parquet(DATA / "hyperparameters.parquet"))
    ev = np.load(DATA / "objectives_evaluations.npy", mmap_mode="r")  # (task, config, seed, epoch, objective)
    Y = np.asarray(ev[:, :, :, -1, 0], dtype=float).mean(2)
    np.savez(cache, X=X, Y=Y)
    return X, Y


def problems(seed: int, n_meta: int = 512) -> List[Problem]:
    """Leave-one-task-out problems for one seed."""
    X, Y = load()
    return leave_one_task_out("hpobench", X, Y, TASKS, n_meta, seed)
