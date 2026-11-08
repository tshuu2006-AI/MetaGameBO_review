"""Independent minimization adapter for the real official PFNs4BO HEBO+ model."""
from __future__ import annotations

import copy
from numbers import Integral, Real
from time import perf_counter

import numpy as np

from models.base_model import BaseOptimizer, validate_points, validate_values

DEFAULT_OPTIONS = dict(model_name="hebo_plus", checkpoint=None, device="cpu", acquisition="ei",
                       n_candidates=128, candidate_batch_size=64, candidate_strategy="uniform",
                       duplicate_atol=1e-8, cpu_threads=2, context_transform="observed_standardize")


def validate_pfn_options(options):
    """Validate config without importing torch, loading weights, or accessing the network."""
    if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS):
        raise ValueError("Unknown PFN model_options; only documented PFN options are accepted")
    values = {**DEFAULT_OPTIONS, **options}
    if values["model_name"] != "hebo_plus" or values["device"] != "cpu" or values["acquisition"] != "ei":
        raise ValueError("Audited PFN baseline requires model_name=hebo_plus, device=cpu, acquisition=ei")
    if values["context_transform"] != "observed_standardize":
        raise ValueError("Only the tested observed_standardize target transform is supported")
    if values["checkpoint"] is not None and (not isinstance(values["checkpoint"], str) or not values["checkpoint"]):
        raise ValueError("checkpoint must be a path string or null (official cache)")
    for key in ("n_candidates", "candidate_batch_size", "cpu_threads"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if values["candidate_strategy"] not in {"uniform", "sobol"}:
        raise ValueError("candidate_strategy must be uniform or sobol")
    tolerance = values["duplicate_atol"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("duplicate_atol must be finite and nonnegative; rtol=0")
    return values


class PFNOptimizer(BaseOptimizer):
    """Pretrained PFN inference, without group meta-training or gradient updates.

    is_meta denotes pretrained prior origin, not permission to train on source
    tasks. Only suggest(1) is supported; asks do not reserve unobserved points.
    `backend` is a Python-only dependency injection seam for isolated tests; YAML
    options cannot set it. CLI always constructs the official real backend.
    """

    is_meta = True

    def __init__(self, bounds, seed=None, model_name="hebo_plus", checkpoint=None, device="cpu", acquisition="ei",
                 n_candidates=128, candidate_batch_size=64, candidate_strategy="uniform", duplicate_atol=1e-8,
                 cpu_threads=2, context_transform="observed_standardize", *, backend=None):
        self.options = validate_pfn_options(dict(model_name=model_name, checkpoint=checkpoint, device=device,
            acquisition=acquisition, n_candidates=n_candidates, candidate_batch_size=candidate_batch_size,
            candidate_strategy=candidate_strategy, duplicate_atol=duplicate_atol, cpu_threads=cpu_threads,
            context_transform=context_transform))
        super().__init__(bounds, seed)
        if self.dim > 18 or not np.isfinite(self.bounds[:, 1] - self.bounds[:, 0]).all():
            raise ValueError("HEBO+ supports up to 18 finite-range features")
        if backend is None:
            from models.pfn_backend import PFNs4BOBackend
            backend = PFNs4BOBackend(checkpoint=checkpoint, model_name=model_name, device=device,
                                    acquisition=acquisition, candidate_batch_size=candidate_batch_size,
                                    cpu_threads=cpu_threads)
        self.backend = backend
        self._reset_task_state()

    def meta_train(self, tasks):
        raise RuntimeError("PFN uses official pretrained weights; group source-task meta_train is disabled")

    def _reset_task_state(self):
        self.context_X = np.empty((0, self.dim))
        self.context_y = np.empty(0)
        self.events = []
        self.inference_seconds = 0.0
        if hasattr(self.backend, "reset_context"):
            self.backend.reset_context()

    def _update(self, X, y):
        self.context_X = validate_points(X, self.bounds)
        self.context_y = validate_values(y, len(X))

    def _normalize(self, points):
        normalized = (points - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        if not np.isfinite(normalized).all() or np.any(normalized < 0) or np.any(normalized > 1):
            raise ValueError("PFN feature normalization must produce finite values in [0,1]")
        return normalized

    def _candidate_pool(self):
        n = int(self.options["n_candidates"])
        if self.options["candidate_strategy"] == "uniform":
            return self._uniform(n)
        from scipy.stats import qmc
        # A fresh scrambled power-of-two pool per ask, then take n entries.
        # This avoids unbalanced Sobol warnings for arbitrary pool sizes.
        sobol_seed = int(self.rng.integers(0, 2**32, dtype=np.uint32))
        unit = qmc.Sobol(self.dim, scramble=True, seed=sobol_seed).random_base2((n - 1).bit_length())[:n]
        return self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])

    def _novel(self, points):
        if len(self.context_X) == 0:
            return np.ones(len(points), dtype=bool)
        return ~np.any(np.all(np.isclose(points[:, None], self.context_X[None], rtol=0,
                                         atol=self.options["duplicate_atol"]), axis=2), axis=1)

    def _bounded_exploration(self, reason):
        event = {"status": "exploration", "reason": reason, "n_context": len(self.context_y),
                 "backend_called": False}
        self.events.append(event)
        for attempt in range(64):
            point = validate_points(self._uniform(1), self.bounds, 1)
            if self._novel(point)[0]:
                event.update(method="seeded_uniform", attempts=attempt + 1)
                return point
        if self.dim == 1:
            lo, hi = self.bounds[0]
            knots = np.unique(np.r_[lo, self.context_X[:, 0], hi])
            widths = np.diff(knots)
            middles = knots[:-1] + widths / 2
            probes = np.r_[middles[np.argsort(-widths, kind="stable")], lo, hi][:, None]
            usable = probes[self._novel(probes)]
            if len(usable):
                event.update(method="deterministic_gap", attempts=64)
                return usable[:1].copy()
        event.update(status="error", method="bounded_failure", attempts=64)
        raise RuntimeError("PFN candidate exhaustion: no novel point after 64 attempts and 1D gap search")

    def suggest(self, n=1):
        if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("n must be a positive integer")
        if n != 1:
            raise NotImplementedError("PFN adapter supports suggest(1) only")
        if not len(self.context_y):
            return self._bounded_exploration("empty_context")
        pool = validate_points(self._candidate_pool(), self.bounds, self.options["n_candidates"])
        candidates = pool[self._novel(pool)]
        if not len(candidates):
            return self._bounded_exploration("candidate_pool_exhausted")
        event = {"status": "scoring", "n_context": len(self.context_y), "n_candidates": len(pool),
                 "n_usable": len(candidates), "acquisition": "ei", "device": "cpu",
                 "y_orientation": "maximize_negative_observed_loss", "apply_power_transform": False,
                 "context_transform": self.options["context_transform"],
                 "checkpoint_id": self.options["model_name"],
                 "checkpoint_sha256": self.backend.metadata.get("checkpoint", {}).get("sha256")}
        self.events.append(event)
        score_seed = int(self.rng.integers(0, 2**32, dtype=np.uint32))
        start = perf_counter()
        try:
            # This is the single minimization -> maximization conversion.
            scores = np.asarray(self.backend.score(self._normalize(self.context_X).copy(),
                -self.context_y.copy(), self._normalize(candidates).copy(), seed=score_seed))
            if (scores.dtype.kind not in "fiu" or scores.shape != (len(candidates),)
                    or not np.isfinite(scores).all() or np.any(scores < 0)):
                raise RuntimeError("PFN EI scores must be a finite nonnegative numeric vector, not a mask/log score")
            index = int(np.argmax(scores))
            result = candidates[index:index+1].copy()
            event.update(status="scored", backend_called=True, selected_index=index,
                         selected_acquisition=float(scores[index]), score_seed=score_seed)
            return validate_points(result, self.bounds, 1)
        except Exception as exc:
            event.update(status="error", exception_type=type(exc).__name__, error=str(exc))
            raise RuntimeError(f"PFN inference failed; no Random substitution: {exc}") from exc
        finally:
            event["inference_seconds"] = perf_counter() - start
            self.inference_seconds += event["inference_seconds"]

    @property
    def diagnostics(self):
        return copy.deepcopy({"method": "pfn_hebo_plus_ei", "model_options": self.options,
            "optimizer_seed": self.seed, "backend": self.backend.metadata,
            "group_meta_train": False, "gradient_updates": 0, "context_size": len(self.context_y),
            "events": self.events, "inference_seconds": self.inference_seconds,
            "model_forward_calls": getattr(self.backend, "forward_calls", None)})
