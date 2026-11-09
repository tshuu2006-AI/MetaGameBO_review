"""Evaluation metrics from the requested protocol."""
from __future__ import annotations

import numpy as np


def calculate_normalized_regret(
    y_observed_best: float | np.ndarray, f_min: float, f_max: float
) -> float | np.ndarray:
    """Compute (y_best-f_min)/(f_max-f_min) for minimisation."""
    denominator = float(f_max) - float(f_min)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("f_max must be finite and strictly greater than f_min")
    result = (np.asarray(y_observed_best, dtype=np.float64) - f_min) / denominator
    return float(result) if result.ndim == 0 else result


def calculate_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Upper-tail CVaR; alpha=.95 averages the worst 5% losses."""
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("losses must be a non-empty finite array")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    threshold = np.quantile(values, alpha, method="higher")
    return float(values[values >= threshold].mean())
