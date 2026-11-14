"""The authors' MetaGameBO code (github.com/Allen0497/MetaGameBO, in ``author/``),
wrapped in the ``BaseOptimizer`` interface. The author code is used unchanged.

Variants follow the author's ``run_forrester.py`` flags: ``use_dro`` (CVaR task
level) and ``use_kdpp`` (k-DPP sample level); both on = MetaGameBO. Defaults:
100 epochs, batch 10, alpha 0.8, kdpp_alpha 0.4, kdpp_lr 0.1, 5 layers,
50 features, 64 hidden units.

The author trainer writes ``results/<name>.txt`` (train/val loss and CVaR every
5 epochs) in the current directory; training runs in a temporary directory and
the file is parsed into ``self.log``.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from models.base import N_CAND, BaseOptimizer, Task

AUTHOR_DIR = Path(__file__).resolve().parent / "author"


def _import_author():
    """Import the author's ``MetaGameBO`` module without shadowing our packages."""
    if "MetaGameBO" in sys.modules:
        return sys.modules["MetaGameBO"]
    sys.path.insert(0, str(AUTHOR_DIR))
    try:
        import MetaGameBO as mod  # noqa: N813  (author module name)
    finally:
        sys.path.remove(str(AUTHOR_DIR))
    return mod


@contextlib.contextmanager
def _in_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parse_author_log(path: Path) -> dict:
    """Parse the author's training file into curves and final losses."""
    log = dict(epoch=[], train_loss=[], train_cvar=[], val_loss=[], val_cvar=[], final={})
    keys = {"Final Train Loss": "train_avg", "Final Train CVaR": "train_cvar",
            "Final Val Loss": "val_avg", "Final Val CVaR": "val_cvar"}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 5 and parts[0].isdigit():
            for k, v in zip(["epoch", "train_loss", "train_cvar", "val_loss", "val_cvar"], parts):
                log[k].append(float(v))
        for k, name in keys.items():
            if line.startswith(k + ":"):
                log["final"][name] = float(line.split(":")[1])
    return log


class MetaGameBOOriginal(BaseOptimizer):
    """Author implementation of MetaGameBO.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed (meta-training and acquisition).
        use_dro: CVaR task level.
        use_kdpp: k-DPP sample level.
        epochs: Meta-training epochs.
        batch_size: Meta-training batch size.
        max_per_task: Subsample source tasks to at most this many points.
        device: Torch device of the author's classifier.
    """

    is_meta = True
    name = "MetaGameBO (original)"

    def __init__(self, bounds, seed: Optional[int] = None, use_dro: bool = True, use_kdpp: bool = True,
                 epochs: int = 100, batch_size: int = 10, max_per_task: Optional[int] = None,
                 device: str = "cpu"):
        super().__init__(bounds, seed)
        self.seed = 44 if seed is None else int(seed)
        self.use_dro, self.use_kdpp = use_dro, use_kdpp
        self.epochs, self.batch_size, self.max_per_task, self.device = epochs, batch_size, max_per_task, device
        self.opt = None
        self.log = None

    def _space(self):
        import parameterspace as ps
        space = ps.ParameterSpace()
        for j in range(self.dim):
            space.add(ps.ContinuousParameter(name=f"x{j}", bounds=[0.0, 1.0]))
        return space

    def meta_train(self, tasks: Sequence[Task], **kwargs) -> None:
        """Run the author's ``meta_fit`` on the source tasks."""
        import torch
        from blackboxopt import EvaluationSpecification
        mod = _import_author()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        meta_data = {}
        for uid, t in enumerate(tasks, start=1):
            X, y = self._normalize(t.X), np.asarray(t.y, float)
            if self.max_per_task and len(y) > self.max_per_task:
                keep = rng.choice(len(y), self.max_per_task, replace=False)
                X, y = X[keep], y[keep]
            meta_data[uid] = [
                EvaluationSpecification(configuration={f"x{j}": float(v) for j, v in enumerate(x)})
                .create_evaluation(objectives={"loss": float(v)}) for x, v in zip(X, y)]
        clf = dict(num_layers=5, num_features=50, num_hidden_units=64, device=self.device)
        self.opt = mod.MetaGameBO(self._space(), seed=self.seed, use_kdpp=self.use_kdpp, alpha=0.4,
                                  kdpp_lr=0.1, num_samples_acquisition_function=N_CAND, **clf)
        train = dict(num_epochs=self.epochs, batch_size=self.batch_size, use_dro=self.use_dro,
                     alpha=0.8, seed=self.seed)
        if self.use_kdpp:
            train.update(use_kdpp=True, kdpp_alpha=0.4, kdpp_lr=0.1)
        with tempfile.TemporaryDirectory() as tmp, _in_dir(tmp), contextlib.redirect_stdout(open(os.devnull, "w")):
            self.opt.meta_fit(meta_data, **train)
            files = list(Path(tmp, "results").glob("*.txt"))
            self.log = parse_author_log(files[0]) if files else None

    def reset(self) -> None:
        super().reset()
        if self.opt is not None:
            self.opt.X, self.opt.losses, self.opt.classifier_gb, self.opt.kdpp = [], [], None, None

    def suggest(self, n: int = 1) -> np.ndarray:
        """The author's ``observe_and_suggest`` over ``N_CAND`` candidates of our search space."""
        X, y = self.history.arrays
        C = self._candidates(N_CAND)
        Xo = self._normalize(X) if len(y) else np.zeros((0, self.dim))
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            i = self.opt.observe_and_suggest(Xo, np.asarray(y, float), X_pen=self._normalize(C),
                                             seed=int(self.rng.integers(32767)))
        return C[[int(i)]]
