"""MALIBO: meta-learned likelihood-free BO (Pan et al. 2024).

Meta-training: ERM on all tasks with the Eq. (3) utility weights
(``models.malibo.training``). Target task: Bayesian logistic regression on the
task embedding (Laplace posterior), Thompson sampling, and a gradient-boosting
residual that starts from the Thompson sample (``models.malibo.adapters``).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from models.base import N_CAND, BaseOptimizer, Task
from models.lfbo.weighting import reweight_samples
from models.malibo.adapters import BLRAdapter, ThompsonInit
from models.malibo.training import AllSamples, AllTasks, MetaTrainConfig, meta_train


class MALIBO(BaseOptimizer):
    """Meta-learned classifier BO.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed (also the meta-training seed unless ``cfg`` sets one).
        cfg: Meta-training configuration (default ``MetaTrainConfig(seed=seed)``).
        use_gb: Add the gradient-boosting residual on the target task.
        hessian: Laplace Hessian of the adapter (``"upstream"`` as in the MALIBO
            code, or ``"exact"``), see ``BLRAdapter``.
        iters: Override ``cfg.iters`` (meta-training steps).

    Attributes:
        net: Trained ``MetaNet``.
        log: ``TrainLog`` of meta-training (training curves, Table-1 losses).
    """

    is_meta = True
    name = "MALIBO"

    def __init__(self, bounds, seed: Optional[int] = None, cfg: Optional[MetaTrainConfig] = None,
                 use_gb: bool = True, hessian: str = "upstream", iters: Optional[int] = None):
        super().__init__(bounds, seed)
        self.cfg = cfg or MetaTrainConfig(seed=seed or 0)
        if iters is not None:
            self.cfg.iters = iters
        self.gamma = self.cfg.gamma
        self.use_gb = use_gb
        self.hessian = hessian
        self.net = self.log = self.adapter = None
        self.gb = None

    def _selectors(self):
        """Task and sample selectors of meta-training (ERM for MALIBO)."""
        return AllTasks(), AllSamples()

    def meta_train(self, tasks: Sequence[Task], callback=None, callback_every: Optional[int] = None) -> None:
        """Meta-train on source tasks (``X`` in the box, ``y``) and build the target adapter."""
        rng = np.random.default_rng(self.cfg.seed)
        unit = []
        for t in tasks:
            X, y = np.asarray(t.X, float), np.asarray(t.y, float)
            if self.cfg.max_per_task and len(y) > self.cfg.max_per_task:
                keep = rng.choice(len(y), self.cfg.max_per_task, replace=False)
                X, y = X[keep], y[keep]
            unit.append(SimpleNamespace(X=self._normalize(X), y=y))
        ts, ss = self._selectors()
        self.net, self.log = meta_train(unit, self.cfg, ts, ss, callback=callback, callback_every=callback_every)
        self.adapter = BLRAdapter(self.net, hessian=self.hessian)

    def reset(self) -> None:
        """Clear the target-task history, the adapter posterior and the GB residual."""
        super().reset()
        if self.adapter is not None:
            self.adapter.reset()
        self.gb = None

    def _update(self, x, y) -> None:
        """Refit the adapter, draw a Thompson sample, refit the GB residual."""
        self.gb = None
        x, lab, w = reweight_samples(self._normalize(x), y, self.gamma, self.cfg.weighting)
        self.adapter.fit(x, lab, w)
        self._z_ts = self.adapter.sample_z(self.rng)
        if self.use_gb and (lab == 0).sum() >= 2 and (lab == 1).sum() >= 2:
            rs = int(self.rng.integers(1 << 30))
            probe = GradientBoostingClassifier(init=ThompsonInit(self), validation_fraction=0.3,
                                               n_iter_no_change=5, random_state=rs)
            probe.fit(x, lab, sample_weight=w)
            self.gb = GradientBoostingClassifier(init=ThompsonInit(self), n_estimators=probe.n_estimators_,
                                                 random_state=rs)
            self.gb.fit(x, lab, sample_weight=w)

    def suggest(self, n: int = 1) -> np.ndarray:
        """Acquisition maximiser over ``N_CAND`` candidates.

        First step: meta-learned mean (warm start); then the GB residual if fitted,
        else the Thompson-sampled classifier. A duplicate is replaced by a random point.
        """
        C = self._candidates(N_CAND)
        U = self._normalize(C)
        if len(self.history.y) == 0:
            a = self.adapter.prob_max(U)
        elif self.gb is not None:
            a = self.gb.predict_proba(U)[:, 1]
        else:
            a = self.adapter.prob_ts(U, self._z_ts)
        x = C[[int(np.argmax(a))]]
        X, _ = self.history.arrays
        if len(X) and np.any(np.all(np.isclose(X, x), axis=1)):
            x = self._candidates(1)
        return x
