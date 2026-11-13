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


def calculate_cvar(losses: np.ndarray, alpha: float = 0.8) -> float:
    """Author-confirmed upper tail, including Python's int/floating semantics."""
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("losses must be a non-empty finite array")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    k = max(1, int(values.size * (1 - alpha)))
    return float(np.sort(values)[-k:].mean())


def summarize_weighted_bce(logits, labels, weights, task_ids, alpha=0.8):
    """Aggregate all rows of each task before computing macro BCE and CVaR.

    Denominators count classifier rows (including duplicated LFBO positives),
    never the sum of weights. Micro BCE weights tasks by their row counts.
    """
    import torch
    from torch.nn import functional as F
    logits, labels, weights = [torch.as_tensor(v).reshape(-1) for v in (logits, labels, weights)]
    ids = torch.as_tensor(task_ids, device=logits.device).reshape(-1)
    if not (len(logits) == len(labels) == len(weights) == len(ids)) or not len(ids):
        raise ValueError("nonempty aligned classifier arrays required")
    if not all(torch.isfinite(v).all() for v in (logits, labels, weights, ids)):
        raise ValueError("classifier arrays must be finite")
    if (weights < 0).any() or ((labels < 0) | (labels > 1)).any():
        raise ValueError("weights must be nonnegative and labels in [0,1]")
    losses = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction="none")
    rows = [{"task_id": int(i), "sample_count": int((ids == i).sum()),
             "weighted_bce": float(losses[ids == i].mean())} for i in torch.unique(ids)]
    task_losses = np.array([r["weighted_bce"] for r in rows])
    return {"macro_bce": float(task_losses.mean()), "micro_bce": float(losses.mean()),
            "cvar_bce": calculate_cvar(task_losses, alpha), "alpha": alpha,
            "tail_k": max(1, int(len(rows) * (1-alpha))), "number_of_tasks": len(rows),
            "sample_count": len(ids), "task_losses": rows}
