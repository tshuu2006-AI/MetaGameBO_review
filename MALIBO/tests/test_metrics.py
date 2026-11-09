import numpy as np

from experiments.metrics import cvar_upper_tail, normalized_regret_curve, summarize_curves


def test_normalized_regret_is_nonincreasing_for_minimization():
    curve = normalized_regret_curve(np.array([8.0, 5.0, 6.0, 2.0]), f_min=0.0, f_max=10.0)
    np.testing.assert_allclose(curve, [0.8, 0.5, 0.5, 0.2])
    assert np.all(np.diff(curve) <= 0.0)


def test_cvar95_selects_worst_five_percent():
    values = np.arange(1.0, 101.0)
    assert cvar_upper_tail(values, confidence=0.95) == 98.0


def test_summary_reports_mean_standard_error_and_cvar():
    curves = np.array([[0.5, 0.4], [0.7, 0.6], [0.3, 0.2]])
    result = summarize_curves(curves, confidence=0.95)
    assert np.isclose(result["mean_final_regret"], 0.4)
    assert np.isclose(result["cvar_upper_tail"], 0.6)
    assert result["standard_error"] > 0.0
