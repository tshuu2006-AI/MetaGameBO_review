"""Independent LFBO-EI with weighted RF classification of observed target data.

Method reference: Song et al., ICML 2022, https://github.com/lfbo-ml/lfbo
(Record.load_classification_data in model.py). No reference source is vendored.
RF scores are acquisition proxies, not a posterior over objective values.
"""
from __future__ import annotations

import copy
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from models.base_model import BaseOptimizer, validate_points, validate_values


DEFAULT_OPTIONS = dict(gamma=1 / 3, weight_type="ei", classifier="rf", n_estimators=128,
                       n_candidates=1024, duplicate_atol=1e-8, n_jobs=1, min_samples_split=2)


def validate_lfbo_options(options: dict) -> dict:
    """Validate options without fitting a model or advancing any RNG."""
    if not isinstance(options, dict) or set(options) - set(DEFAULT_OPTIONS):
        raise ValueError("Unknown LFBO model_options; see DEFAULT_OPTIONS in models/lfbo.py")
    values = {**DEFAULT_OPTIONS, **options}
    for key in ("gamma", "duplicate_atol"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite real number")
    if not 0 < values["gamma"] < 1:
        raise ValueError("gamma must satisfy 0 < gamma < 1")
    if values["duplicate_atol"] < 0:
        raise ValueError("duplicate_atol must be nonnegative (absolute tolerance; rtol=0)")
    if values["weight_type"] != "ei" or values["classifier"] != "rf":
        raise ValueError("This baseline supports weight_type=ei and classifier=rf only; PI is not implemented")
    for key in ("n_estimators", "n_candidates", "n_jobs", "min_samples_split"):
        value = values[key]
        minimum = 2 if key == "min_samples_split" else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    return values


class LFBOFallback(ValueError):
    """Valid observations cannot currently define a two-class LFBO dataset."""


def prepare_lfbo_classification(X, y, gamma=1 / 3, weight_type="ei"):
    """Return independent (features, labels, mean-one weights) for minimization EI.

    tau is the linear NumPy quantile of ALL observed y, including repeats.
    A good point (y < tau) contributes both a utility-weighted positive and a
    background negative. Every observed point contributes a background negative.
    Raise LFBOFallback for degenerate *finite* data; invalid inputs raise ValueError.
    """
    validate_lfbo_options({"gamma": gamma, "weight_type": weight_type})
    points = validate_points(X)
    values = validate_values(y, len(points))
    count = len(values)
    if count < 2:
        raise LFBOFallback("insufficient_observations")
    if np.all(values == values[0]):
        raise LFBOFallback("all_values_equal")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        threshold = float(np.quantile(values, gamma, method="linear"))
        if not np.isfinite(threshold):
            raise LFBOFallback("invalid_quantile")
        good = values < threshold
        positives = int(good.sum())
        if positives == 0:
            raise LFBOFallback("no_strict_improvement")
        utility = threshold - values[good]
        average = utility.mean()
        if not np.isfinite(utility).all() or np.any(utility <= 0) or not np.isfinite(average) or average <= 0:
            raise LFBOFallback("invalid_utility")
        total = count + positives
        weights = np.concatenate((utility / average * (total / positives),
                                  np.full(count, total / count)))
        weights = weights / weights.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise LFBOFallback("invalid_weights")
    features = np.concatenate((points[good], points), axis=0)
    labels = np.concatenate((np.ones(positives, dtype=int), np.zeros(count, dtype=int)))
    if len(features) != len(labels) or len(labels) != len(weights) or len(np.unique(labels)) != 2:
        raise LFBOFallback("insufficient_classes")
    return features, labels, weights


class LFBO(BaseOptimizer):
    """Non-meta LFBO-EI; refit the full observed history after every observe().

    Only suggest(1) is supported. Proposals are checked against observed history;
    repeated asks without observe do not reserve pending points. Ties choose the
    first maximum in the seeded candidate order. reset retains the RNG stream
    and RF construction seed, while clearing all fitted state and diagnostics.
    """

    is_meta = False

    def __init__(self, bounds, seed=None, gamma=1 / 3, weight_type="ei", classifier="rf",
                 n_estimators=128, n_candidates=1024, duplicate_atol=1e-8, n_jobs=1,
                 min_samples_split=2):
        self.options = validate_lfbo_options(dict(gamma=gamma, weight_type=weight_type, classifier=classifier,
                                                n_estimators=n_estimators, n_candidates=n_candidates,
                                                duplicate_atol=duplicate_atol, n_jobs=n_jobs,
                                                min_samples_split=min_samples_split))
        super().__init__(bounds, seed)
        self.classifier_seed = int(self.rng.integers(0, 2**32, dtype=np.uint32))
        self._reset_task_state()

    def _reset_task_state(self):
        self.model = None
        self.fallback_reason = "cold_start"
        self.fit_count = 0
        self.last_training_set_size = 0
        self.fit_events = []
        self.fit_skips = []
        self.fallback_events = []
        self.acquisition_events = []
        self.fit_seconds = 0.0
        self.acquisition_seconds = 0.0

    def _update(self, X, y):
        points = validate_points(X, self.bounds)
        values = validate_values(y, len(points))
        self.model = None
        self.last_training_set_size = 0
        try:
            features, labels, weights = prepare_lfbo_classification(
                points, values, self.options["gamma"], self.options["weight_type"])
        except LFBOFallback as exc:
            self.fallback_reason = str(exc)
            self.fit_skips.append({"n_observed": len(values), "reason": str(exc)})
            return
        start = perf_counter()
        model = RandomForestClassifier(n_estimators=self.options["n_estimators"],
                                       min_samples_split=self.options["min_samples_split"],
                                       class_weight=None, random_state=self.classifier_seed,
                                       n_jobs=self.options["n_jobs"])
        model.fit(features, labels, sample_weight=weights)
        self.fit_seconds += perf_counter() - start
        if set(model.classes_) != {0, 1}:
            raise RuntimeError("LFBO fitted classifier must contain both classes 0 and 1")
        self.model = model
        self.fallback_reason = None
        self.fit_count += 1
        self.last_training_set_size = len(labels)
        self.fit_events.append({"n_observed": len(values), "n_training_rows": len(labels),
                                "n_positive": int(np.sum(labels)), "classifier_seed": self.classifier_seed})

    def _novel_mask(self, points):
        history, _ = self.history.arrays
        if len(history) == 0:
            return np.ones(len(points), dtype=bool)
        close = np.all(np.isclose(points[:, None, :], history[None, :, :],
                                  atol=self.options["duplicate_atol"], rtol=0), axis=2)
        return ~np.any(close, axis=1)

    def _fallback_point(self, reason):
        event = {"reason": reason, "n_observed": len(self.history.y), "random_attempts": 0}
        self.fallback_events.append(event)
        for attempt in range(64):
            event["random_attempts"] = attempt + 1
            point = validate_points(self._uniform(1), self.bounds, 1)
            if self._novel_mask(point)[0]:
                event["method"] = "seeded_uniform"
                return point
        # In 1D, endpoints and midpoints between consecutive observed points
        # cover every potentially nonempty gap outside the exclusion intervals.
        if self.dim == 1:
            history, _ = self.history.arrays
            lo, hi = self.bounds[0]
            knots = np.unique(np.r_[lo, history[:, 0] if len(history) else [], hi])
            widths = np.diff(knots)
            order = np.argsort(-widths, kind="stable")
            midpoints = knots[:-1] + widths / 2
            points = np.r_[midpoints[order], lo, hi][:, None]
            usable = points[self._novel_mask(points)]
            if len(usable):
                event["method"] = "deterministic_gap"
                return usable[:1].copy()
        event["method"] = "bounded_failure"
        raise RuntimeError("LFBO found no nonduplicate point after 64 random attempts and 1D gap search")

    def suggest(self, n=1):
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError("n must be a positive integer")
        if n != 1:
            raise NotImplementedError("LFBO supports suggest(1) only; batch acquisition is not implemented")
        start = perf_counter()
        try:
            if self.model is None:
                return self._fallback_point(self.fallback_reason)
            classes = np.asarray(self.model.classes_)
            positive_column = np.flatnonzero(classes == 1)
            if len(positive_column) != 1:
                raise RuntimeError("LFBO classifier is missing the unique positive class 1")
            candidates = validate_points(self._uniform(self.options["n_candidates"]), self.bounds,
                                         self.options["n_candidates"])
            usable = candidates[self._novel_mask(candidates)]
            event = {"n_observed": len(self.history.y), "n_sampled": len(candidates), "n_usable": len(usable)}
            self.acquisition_events.append(event)
            if len(usable) == 0:
                return self._fallback_point("candidate_pool_exhausted")
            probabilities = np.asarray(self.model.predict_proba(usable), dtype=float)
            if (probabilities.shape != (len(usable), len(classes)) or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0) or np.any(probabilities > 1)):
                raise RuntimeError("LFBO classifier returned invalid acquisition scores")
            scores = probabilities[:, positive_column[0]]
            selected = int(np.argmax(scores))
            event["selected_score"] = float(scores[selected])
            return usable[selected:selected+1].copy()
        finally:
            self.acquisition_seconds += perf_counter() - start

    @property
    def diagnostics(self):
        """Copied JSON-compatible model state; no evaluator task descriptors/references."""
        counts = {}
        for event in self.fallback_events:
            counts[event["reason"]] = counts.get(event["reason"], 0) + 1
        return copy.deepcopy({"method": "lfbo_ei_rf", "model_options": self.options,
                              "optimizer_seed": self.seed, "classifier_seed": self.classifier_seed,
                              "fit_count": self.fit_count, "last_training_set_size": self.last_training_set_size,
                              "fit_events": self.fit_events, "fit_skips": self.fit_skips,
                              "fallback_events": self.fallback_events, "fallback_counts": counts,
                              "acquisition_events": self.acquisition_events,
                              "fit_seconds": self.fit_seconds, "acquisition_seconds": self.acquisition_seconds,
                              "trained": self.model is not None, "fallback_reason": self.fallback_reason})
