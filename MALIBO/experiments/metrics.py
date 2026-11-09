from __future__ import annotations

import math
import numpy as np


def normalized_regret_curve(
    observed_losses: np.ndarray,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Regret curve for minimization, normalized by the task's function range."""
    values = np.asarray(observed_losses, dtype=np.float64).reshape(-1)
    scale = float(f_max) - float(f_min)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("f_max must be finite and strictly greater than f_min.")
    best_so_far = np.minimum.accumulate(values)
    return (best_so_far - float(f_min)) / scale


def cvar_upper_tail(values: np.ndarray, confidence: float = 0.95) -> float:
    """Mean of the worst (1-confidence) fraction of final losses/regrets."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("values must not be empty")
    if not 0.0 <= confidence < 1.0:
        raise ValueError("confidence must be in [0, 1).")
    tail_count = max(1, int(math.ceil((1.0 - confidence) * x.size - 1e-12)))
    return float(np.mean(np.sort(x)[-tail_count:]))


def summarize_curves(curves: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    matrix = np.asarray(curves, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("curves must have shape (n_runs, n_steps) with n_runs >= 1")
    final = matrix[:, -1]
    mean = float(np.mean(final))
    if final.size > 1:
        standard_error = float(np.std(final, ddof=1) / np.sqrt(final.size))
    else:
        standard_error = 0.0
    return {
        "mean_final_regret": mean,
        "standard_error": standard_error,
        "cvar_upper_tail": cvar_upper_tail(final, confidence=confidence),
    }
