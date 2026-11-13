"""Shared RNG and finite-value contracts for reproducible experiments."""
from __future__ import annotations

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_finite(values, name: str) -> None:
    finite = (bool(torch.isfinite(values).all()) if isinstance(values, torch.Tensor)
              else bool(np.isfinite(values).all()))
    if not finite:
        raise FloatingPointError(f"{name} contains NaN or Inf")


