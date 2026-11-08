"""LFBO recipe arithmetic, RF weighting, data validity and bounded acquisition."""
import copy

import numpy as np
import pytest

from models import lfbo
from models.lfbo import LFBO, LFBOFallback, prepare_lfbo_classification


def small_model(**kwargs):
    return LFBO([[0, 1]], seed=7, n_estimators=5, n_candidates=32, **kwargs)


def test_ei_manual_weights_duplicate_good_point_as_background():
    X = np.array([[.1], [.4], [.8]])
    features, labels, weights = prepare_lfbo_classification(X, [0, 1, 2], 1 / 3, "ei")
    np.testing.assert_array_equal(features, [[.1], [.1], [.4], [.8]])
    np.testing.assert_array_equal(labels, [1, 0, 0, 0])
    np.testing.assert_allclose(weights, [2, 2 / 3, 2 / 3, 2 / 3])
    assert weights.mean() == pytest.approx(1)
    # Unequal improvements must survive normalization (two positives).
    features, labels, weights = prepare_lfbo_classification(np.arange(5)[:, None], [-4, -2, 0, 2, 4], .5)
    np.testing.assert_array_equal(labels, [1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(weights, [7 / 3, 7 / 6, .7, .7, .7, .7, .7])
    assert weights[labels == 1].sum() == pytest.approx(weights[labels == 0].sum())


def test_quantile_uses_repeated_values_and_strict_inequality():
    # Quantile over all observations is 0; unique-y quantile would be >0.
    with pytest.raises(LFBOFallback, match="no_strict_improvement"):
        prepare_lfbo_classification(np.arange(4)[:, None], [0, 0, 0, 3], 1 / 3)
    _, labels, _ = prepare_lfbo_classification(np.arange(4)[:, None], [-1, 0, 0, 3], .5)
    assert list(labels) == [1, 0, 0, 0, 0]  # y=tau=0 is never positive


@pytest.mark.parametrize("options", [{"gamma": g} for g in [0, 1, -1, np.nan, np.inf, True, "0.3"]]
    + [{"n_estimators": -1}, {"n_estimators": True}, {"n_estimators": 2.5}, {"n_candidates": 0},
       {"n_jobs": 0}, {"min_samples_split": 1}, {"duplicate_atol": -1}, {"duplicate_atol": np.inf},
       {"weight_type": "pi"}, {"classifier": "mlp"}])
def test_invalid_options_fail(options):
    with pytest.raises(ValueError):
        LFBO([[0, 1]], **options)


@pytest.mark.parametrize("X,y", [([[np.nan]], [0]), ([[.2]], [np.inf]), ([.2], [0]),
    ([[.2]], [0, 1]), ([[.2], [.3]], [[0, 1]])])
def test_helper_rejects_invalid_data_instead_of_fallback(X, y):
    with pytest.raises(ValueError) as exc:
        prepare_lfbo_classification(X, y)
    assert not isinstance(exc.value, LFBOFallback)


def test_nonfinite_or_out_of_bounds_observations_are_rejected_atomically():
    model = small_model()
    for X, y in [([[.2]], [np.nan]), ([[np.inf]], [1]), ([[1.1]], [1])]:
        with pytest.raises(ValueError):
            model.observe(X, y)
        assert not model.history.y and model.fit_count == 0


def test_degenerate_finite_data_never_fits_or_predicts(monkeypatch):
    monkeypatch.setattr(lfbo, "RandomForestClassifier", lambda **kw: pytest.fail("Degenerate fit"))
    model = small_model()
    point = model.suggest()
    assert point.shape == (1, 1) and np.isfinite(point).all()
    assert model.diagnostics["fallback_counts"] == {"cold_start": 1}
    model.observe([[.1]], [3])
    model.suggest()
    assert model.fallback_reason == "insufficient_observations"
    model.observe([[.4], [.8]], [3, 3])
    point = model.suggest()
    assert model.fallback_reason == "all_values_equal"
    assert model.model is None and not np.isclose(point, [[.1], [.4], [.8]], rtol=0, atol=1e-8).any()


def test_incremental_full_history_and_sample_weight_fit_contract(monkeypatch):
    captured = []

    class SpyRF:
        def __init__(self, **options):
            assert options["class_weight"] is None and options["n_jobs"] == 1
            self.classes_ = np.array([0, 1])

        def fit(self, X, labels, sample_weight):
            assert np.isfinite(sample_weight).all() and np.all(sample_weight > 0)
            captured.append((X.copy(), labels.copy(), sample_weight.copy()))

    monkeypatch.setattr(lfbo, "RandomForestClassifier", SpyRF)
    model = small_model()
    model.observe([[.1]], [0])
    assert model.model is None
    model.observe([[.4]], [1])
    assert model.model is not None and model.fit_count == 1
    model.observe([[.8]], [2])
    assert model.fit_count == 2 and model.last_training_set_size == 4
    assert [e["n_observed"] for e in model.fit_events] == [2, 3]
    np.testing.assert_array_equal(captured[-1][0], [[.1], [.1], [.4], [.8]])
    np.testing.assert_allclose(captured[-1][2], [2, 2 / 3, 2 / 3, 2 / 3])
    diagnostics = model.diagnostics
    diagnostics["fit_events"].clear()
    assert len(model.diagnostics["fit_events"]) == 2


def test_acquisition_selects_class_one_column_and_first_tie(monkeypatch):
    model = small_model()
    model.observe([[.1], [.9]], [-2, 0])
    candidates = np.linspace(.2, .8, 32)[:, None]
    monkeypatch.setattr(model, "_uniform", lambda n: candidates.copy())

    class ReversedClasses:
        classes_ = np.array([1, 0])

        def predict_proba(self, X):
            assert len(X) == 32
            scores = np.zeros(32)
            scores[4:6] = .9
            return np.column_stack((scores, 1 - scores))

    model.model = ReversedClasses()
    np.testing.assert_array_equal(model.suggest(), candidates[4:5])


@pytest.mark.parametrize("invalid", ["classes", "nan", "shape"])
def test_invalid_classifier_output_fails_clearly(invalid):
    model = small_model()

    class BrokenClassifier:
        classes_ = np.array([0]) if invalid == "classes" else np.array([0, 1])

        def predict_proba(self, X):
            return np.full((len(X), 1 if invalid == "shape" else 2), np.nan if invalid == "nan" else .5)

    model.model = BrokenClassifier()
    with pytest.raises(RuntimeError, match="class|scores"):
        model.suggest()


def test_duplicate_tolerance_is_absolute_not_relative():
    model = LFBO([[0, 100]], seed=7, n_estimators=2, duplicate_atol=1e-8)
    model.observe([[90]], [0])
    mask = model._novel_mask(np.array([[90], [90 + 5e-9], [90 + 5e-7]]))
    assert mask.tolist() == [False, False, True]


def test_exhausted_candidates_use_bounded_gap_search(monkeypatch):
    model = small_model()
    model.observe([[.25], [.75]], [0, 1])
    monkeypatch.setattr(model, "_uniform", lambda n: np.full((n, 1), .25))
    point = model.suggest()
    np.testing.assert_array_equal(point, [[.5]])
    assert model.diagnostics["fallback_events"][-1] == {
        "reason": "candidate_pool_exhausted", "n_observed": 2, "random_attempts": 64, "method": "deterministic_gap"}


def test_gap_search_tries_endpoints_if_middle_excluded(monkeypatch):
    model = small_model(duplicate_atol=.3)
    model.observe([[.25], [.75]], [0, 1])
    # All of [0,1] is excluded with this tolerance, so failure is explicit.
    monkeypatch.setattr(model, "_uniform", lambda n: np.full((n, 1), .25))
    with pytest.raises(RuntimeError, match="64 random attempts"):
        model.suggest()
    assert model.diagnostics["fallback_events"][-1]["method"] == "bounded_failure"


@pytest.mark.parametrize("n", [0, -1, True, 1.2, 2])
def test_suggest_rejects_invalid_or_batch_requests(n):
    with pytest.raises((ValueError, NotImplementedError)):
        small_model().suggest(n)


def trajectory(seed):
    model = LFBO([[0, 1]], seed=seed, n_estimators=5, n_candidates=64)
    for _ in range(6):
        point = model.suggest()
        assert point.shape == (1, 1) and np.isfinite(point).all() and 0 <= point[0, 0] <= 1
        assert model._novel_mask(point)[0]
        model.observe(point, (point[:, 0] - .63)**2)
    return model


def test_seed_determinism_different_streams_and_reset():
    first, second, different = trajectory(17), trajectory(17), trajectory(18)
    for a, b in zip(first.history.arrays, second.history.arrays):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first.history.arrays[0], different.history.arrays[0])
    old_classifier_seed = first.classifier_seed
    assert first.fit_count == 5
    first.reset()
    assert first.model is None and not first.history.y and first.fit_count == 0
    assert first.last_training_set_size == 0 and first.classifier_seed == old_classifier_seed
    assert first.fit_events == first.fit_skips == first.fallback_events == first.acquisition_events == []
    assert first.fit_seconds == first.acquisition_seconds == 0
    # Like BaseOptimizer, reset retains the RNG stream rather than restarting it.
    np.testing.assert_array_equal(first.suggest(), second._uniform(1))


def test_invalid_utility_does_not_divide_or_train():
    with pytest.raises(LFBOFallback, match="invalid_utility|invalid_quantile"):
        prepare_lfbo_classification([[.1], [.9]], [-1.7e308, 1.7e308], .9)
