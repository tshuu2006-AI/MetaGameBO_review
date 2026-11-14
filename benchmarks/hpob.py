"""HPO-B (Arango et al. 2021), continuous version: 16 search spaces (2-18 dims),
86 test datasets. The objective of a test dataset is its XGBoost surrogate of
the accuracy (we minimise ``-accuracy``); the regret is
``(y_max - best accuracy) / (y_max - y_min)`` with the surrogate statistics,
clipped to [0, 1]. Each run starts from the 5 fixed initial points of the
HPO-B seed ``test0..test4``. Meta-training uses the provided meta-train
datasets of the same search space (each subsampled to 512 points).

Data (``hpob-data``, ``saved-surrogates``) in ``benchmarks/data``; the meta-train
JSON is large, so ``prepare()`` writes one compact ``.npz`` per search space.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from benchmarks.common import Problem
from models.base import Task

DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "hpob_cache"
SEEDS = ["test0", "test1", "test2", "test3", "test4"]


def prepare(n_meta: int = 512, seed: int = 0) -> None:
    """Build ``hpob_cache/<space>.npz`` from the HPO-B JSON files (once; files loaded one at a time)."""
    import gc
    CACHE.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    init = json.load(open(DATA / "hpob-data" / "bo-initializations.json"))
    test = json.load(open(DATA / "hpob-data" / "meta-test-dataset.json"))
    out = {space: {} for space in test}
    for space, sets in test.items():
        for ds, d in sets.items():
            out[space][f"test_X_{ds}"] = np.array(d["X"], float)
            out[space][f"test_y_{ds}"] = np.array(d["y"], float).reshape(-1)
            for s in SEEDS:
                out[space][f"init_{ds}_{s}"] = np.array(init[space][ds][s], int)
    del test, init
    gc.collect()
    train = json.load(open(DATA / "hpob-data" / "meta-train-dataset.json"))
    for space in out:
        for i, (ds, d) in enumerate(train.get(space, {}).items()):
            X, y = np.array(d["X"], float), np.array(d["y"], float).reshape(-1)
            if len(y) > n_meta:
                keep = rng.choice(len(y), n_meta, replace=False)
                X, y = X[keep], y[keep]
            out[space][f"train_X_{i}"], out[space][f"train_y_{i}"] = X, y
        train[space] = None
    del train
    gc.collect()
    for space, arrays in out.items():
        np.savez_compressed(CACHE / f"{space}.npz", **arrays)


def search_spaces() -> List[str]:
    """IDs of the 16 test search spaces."""
    return sorted(json.load(open(DATA / "hpob-data" / "meta-test-dataset.json")).keys()) \
        if not CACHE.exists() else sorted(p.stem for p in CACHE.glob("*.npz"))


class Surrogate:
    """XGBoost surrogate of one test dataset; returns ``-accuracy`` (to minimise)."""

    def __init__(self, space: str, ds: str):
        import xgboost as xgb
        import warnings
        name = f"surrogate-{space}-{ds}"
        self.booster = xgb.Booster()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.booster.load_model(str(DATA / "saved-surrogates" / f"{name}.json"))
        stats = json.load(open(DATA / "saved-surrogates" / "summary-stats.json"))[name]
        self.y_min, self.y_max = stats["y_min"], stats["y_max"]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        return -self.booster.predict(xgb.DMatrix(np.atleast_2d(X))).astype(float)

    def regret(self, y_seen: np.ndarray) -> np.ndarray:
        acc = np.maximum.accumulate(-np.asarray(y_seen, float))
        return np.clip((self.y_max - acc) / (self.y_max - self.y_min), 0.0, 1.0)


def problems(space: str, seed_id: str = "test0") -> List[Problem]:
    """One problem per test dataset of ``space`` for the HPO-B seed ``seed_id``."""
    d = np.load(CACHE / f"{space}.npz")
    meta = [Task(objective=None, X=d[f"train_X_{i}"], y=-d[f"train_y_{i}"])
            for i in range(sum(k.startswith("train_X_") for k in d.files))]
    probs = []
    for k in sorted(f for f in d.files if f.startswith("test_X_")):
        ds = k[len("test_X_"):]
        X = d[k]
        sur = Surrogate(space, ds)
        bounds = np.column_stack([np.zeros(X.shape[1]), np.ones(X.shape[1])])
        init = X[d[f"init_{ds}_{seed_id}"]]
        probs.append(Problem(f"hpob/{space}/{ds}", bounds, sur, sur.regret, meta, init=init))
    return probs
