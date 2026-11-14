"""PFNs4BO (Müller et al. 2023): a transformer pre-trained on a prior of functions
does in-context BO; it is not trained on our tasks (pre-trained weights in
``PFNs4BO/pfns4bo/final_models``).

Uses the authors' ``TransformerBOMethod`` (EI over a candidate set) with the
HEBO+ prior model. The model accepts at most 18 input dimensions.
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from models.base import N_CAND, BaseOptimizer

PFN_DIR = Path(__file__).resolve().parent / "PFNs4BO"
_MODELS: dict = {}


def load_pfn(name: str = "hebo_plus_model", device: str = "cpu"):
    """Load (and cache per process) a pre-trained PFN by its ``pfns4bo`` name."""
    if name not in _MODELS:
        import torch
        if str(PFN_DIR) not in sys.path:
            sys.path.insert(0, str(PFN_DIR))
        import pfns4bo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODELS[name] = torch.load(getattr(pfns4bo, name), map_location="cpu", weights_only=False)
    return _MODELS[name].to(device).eval()


class PFN(BaseOptimizer):
    """PFN-based BO with Expected Improvement.

    Args:
        bounds: Search box of shape (d, 2), d <= 18.
        seed: Random seed.
        model: ``"hebo_plus_model"`` (default) or ``"bnn_model"``.
        device: Torch device; ``None`` picks CUDA if available.
    """

    name = "PFN"

    def __init__(self, bounds, seed: Optional[int] = None, model: str = "hebo_plus_model",
                 device: Optional[str] = None):
        super().__init__(bounds, seed)
        import torch
        if self.dim > 18:
            raise ValueError("the pre-trained PFNs accept at most 18 input dimensions")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        self._bo = None

    def _method(self):
        if self._bo is None:
            m = load_pfn(self.model_name, self.device)  # also puts PFNs4BO on sys.path
            from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
            self._bo = TransformerBOMethod(m, device=self.device)
        return self._bo

    def suggest(self, n: int = 1) -> np.ndarray:
        """EI maximiser over ``N_CAND`` candidates (random until 1 observation)."""
        X, y = self.history.arrays
        if len(y) < 1:
            return self._candidates(n)
        import torch
        torch.manual_seed(int(self.rng.integers(1 << 30)))
        bo = self._method()
        span = np.ptp(y)
        y01 = (y.max() - y) / span if span > 0 else np.full(len(y), 0.5)  # PFN maximises
        C = self._candidates(N_CAND)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            i = bo.observe_and_suggest(self._normalize(X), y01, self._normalize(C))
        return C[[int(i)]]

    def prob_good(self, X, y, Xq, tau):
        """P(f(Xq) < tau) from the PFN posterior predictive (bar distribution)."""
        import torch
        y = np.asarray(y, float)
        if len(y) < 1:
            return np.full(len(Xq), 0.5)
        m = load_pfn(self.model_name, self.device)
        mu, sd = y.mean(), (y.std() if y.std() > 0 else 1.0)
        yt = torch.tensor(-(y - mu) / sd, dtype=torch.float32, device=self.device)  # maximisation scale
        xg = torch.tensor(self._normalize(X), dtype=torch.float32, device=self.device)
        out = []
        with torch.no_grad():
            for xq in np.array_split(self._normalize(Xq), max(1, len(Xq) // 2048)):
                xe = torch.tensor(xq, dtype=torch.float32, device=self.device)
                xf = torch.cat([xg, xe], 0).unsqueeze(1)
                logits = m((None, xf, yt.unsqueeze(1)), single_eval_pos=len(xg))[:, 0]
                out.append(m.criterion.pi(logits, float(-(tau - mu) / sd)).cpu().numpy())
        return np.concatenate(out)
