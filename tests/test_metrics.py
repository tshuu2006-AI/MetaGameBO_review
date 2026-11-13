import numpy as np
import pytest

from metrics import calculate_cvar, calculate_normalized_regret
from metrics import summarize_weighted_bce
import torch


def test_normalized_regret_formula():
    assert calculate_normalized_regret(3.0, 1.0, 5.0) == 0.5


def test_normalized_regret_rejects_zero_range():
    with pytest.raises(ValueError):
        calculate_normalized_regret(1.0, 1.0, 1.0)


def test_cvar95_is_worst_five_percent():
    assert calculate_cvar(np.arange(1.0, 101.0), 0.95) == 98.0


@pytest.mark.parametrize('n', [1, 4, 5, 10, 200])
def test_cvar_exact_author_floor_rule(n):
    values = np.arange(n, dtype=float)
    k = max(1, int(n * (1-.8)))
    assert calculate_cvar(values) == np.sort(values)[-k:].mean()


def test_macro_micro_and_denominator_are_distinct():
    logits = torch.tensor([0., 0., 0., 0.], dtype=torch.float64)
    labels = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
    weights = torch.tensor([6., 1., 1., 1.], dtype=torch.float64)
    m = summarize_weighted_bce(logits, labels, weights, [1, 2, 2, 2])
    assert m['macro_bce'] == pytest.approx(3.5*np.log(2))
    assert m['micro_bce'] == pytest.approx(2.25*np.log(2))
    assert m['cvar_bce'] == pytest.approx(6*np.log(2))
    assert m['tail_k'] == 1


def test_split_task_fragments_are_aggregated_before_cvar():
    m = summarize_weighted_bce(torch.zeros(5), torch.zeros(5), torch.tensor([1., 8., 1., 2., 3.]), [1,2,1,3,4])
    assert m['number_of_tasks'] == 4
    assert m['task_losses'][0]['sample_count'] == 2
    assert m['cvar_bce'] == pytest.approx(8*np.log(2))
