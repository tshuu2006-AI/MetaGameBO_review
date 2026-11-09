import numpy as np
import pytest

from metrics import calculate_cvar, calculate_normalized_regret


def test_normalized_regret_formula():
    assert calculate_normalized_regret(3.0, 1.0, 5.0) == 0.5


def test_normalized_regret_rejects_zero_range():
    with pytest.raises(ValueError):
        calculate_normalized_regret(1.0, 1.0, 1.0)


def test_cvar95_is_worst_five_percent():
    assert calculate_cvar(np.arange(1.0, 101.0), 0.95) == 98.0
