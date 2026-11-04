"""Explicit minimization metrics and evaluator-only numerical references."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _finite_vector(values):
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("Expected a nonempty finite one-dimensional vector")
    return data


def best_so_far(y) -> np.ndarray:
    """Return the running minimum of a finite observation vector."""
    return np.minimum.accumulate(_finite_vector(y))


def simple_regret(y_best, f_star: float, tolerance: float = 1e-8):
    """Return y_best-f_star; clamp only negative roundoff within tolerance, never abs."""
    data = np.asarray(y_best, dtype=np.float64)
    if data.ndim > 1 or data.size == 0 or not np.isfinite(data).all() or not np.isfinite(f_star) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Regret inputs and nonnegative tolerance must be finite")
    regret = data - f_star
    if np.any(regret < -tolerance):
        raise ValueError("Observed value is below the reference minimum beyond tolerance")
    result = np.maximum(regret, 0.0)
    return float(result) if result.ndim == 0 else result


def normalized_regret(y_best, reference_min: float, reference_max: float, tolerance: float = 1e-8):
    """Normalize regret by a fixed positive task reference range, never the observed range."""
    scale = reference_max - reference_min
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference_max-reference_min must be finite and strictly positive")
    data = np.asarray(y_best, dtype=np.float64)
    if np.any(data > reference_max + tolerance):
        raise ValueError("Observation exceeds the fixed reference maximum")
    return simple_regret(y_best, reference_min, tolerance) / scale


def empirical_cvar(losses, alpha: float = 0.8) -> float:
    """Exact upper-tail CVaR of equally weighted empirical losses.

    Each loss has mass 1/n. Integrate the worst (1-alpha) mass, including a
    fractional boundary observation; ties have identical values so their order
    cannot change the result. This is not rounded top-k averaging.
    """
    values = np.sort(_finite_vector(losses))[::-1]
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("CVaR alpha must satisfy 0 < alpha < 1")
    mass = len(values) * (1.0 - alpha)
    weights = np.clip(mass - np.arange(len(values)), 0.0, 1.0)
    return float(np.dot(values, weights) / mass)


def aggregate_runs(values) -> dict:
    """Return count, mean, sample std and SE; std/SE are null for one sample."""
    data = _finite_vector(values)
    std = float(np.std(data, ddof=1)) if len(data) > 1 else None
    return {"n": len(data), "mean": float(np.mean(data)), "sample_std": std,
            "standard_error": std / np.sqrt(len(data)) if std is not None else None}


def _estimate_extremum(task, bounds, grid_size, xatol, sign):
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 257:
        raise ValueError("oracle grid_size must be an integer >= 257")
    lo, hi = map(float, bounds)
    if not 0 <= lo < hi <= 1 or not np.isfinite(xatol) or xatol <= 0:
        raise ValueError("Invalid oracle bounds or xatol")
    grid = np.linspace(lo, hi, grid_size)
    scores = sign * task.evaluate(grid[:, None])
    local = np.flatnonzero((scores[1:-1] <= scores[:-2]) & (scores[1:-1] <= scores[2:])) + 1
    candidates = [(float(grid[i]), float(scores[i])) for i in (0, grid_size - 1, int(np.argmin(scores)))]
    for i in local:
        fit = minimize_scalar(lambda x: sign * task.evaluate_scalar(x), bounds=(grid[i-1], grid[i+1]),
                              method="bounded", options={"xatol": xatol})
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError("Oracle local refinement did not converge")
        candidates.append((float(fit.x), float(fit.fun)))
    x, value = min(candidates, key=lambda pair: pair[1])
    return {"x": x, "value": sign * value, "method": "dense_grid_local_refinement_plus_endpoints",
            "grid_size": int(grid_size), "xatol": float(xatol), "exact_global_optimum_guaranteed": False}


def estimate_global_minimum_1d(task, bounds=(0.0, 1.0), grid_size=4097, xatol=1e-12) -> dict:
    """Estimate all grid-bracketed local minima plus endpoints; not a proof of global optimality."""
    return _estimate_extremum(task, bounds, grid_size, xatol, 1)


def estimate_reference_range(task, bounds=(0.0, 1.0), grid_size=4097, xatol=1e-12) -> dict:
    """Estimate fixed task min/max for evaluation only, independent of optimizer histories."""
    low = estimate_global_minimum_1d(task, bounds, grid_size, xatol)
    high = _estimate_extremum(task, bounds, grid_size, xatol, -1)
    if high["value"] <= low["value"]:
        raise ValueError("Oracle reference range is not positive")
    return {"f_star_estimate": low["value"], "reference_min": low["value"], "reference_max": high["value"],
            "argmin_estimate": low["x"], "argmax_estimate": high["x"], "oracle_method": low["method"],
            "grid_size": int(grid_size), "xatol": float(xatol), "exact_global_optimum_guaranteed": False}
