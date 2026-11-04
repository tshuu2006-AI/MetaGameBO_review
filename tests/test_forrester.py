"""Formula, shape, data separation and manifest reproducibility checks."""
import copy
import math

import numpy as np
import pytest

from benchmarks.forrester import ForresterTask, forrester, generate_manifest, validate_manifest


@pytest.mark.parametrize("x,base", [(0.0, 4 * math.sin(-4)), (0.5, math.sin(2)), (1.0, 16 * math.sin(8))])
@pytest.mark.parametrize("a,b,c", [(1, 0, 0), (0.2, -5, -5), (3, 15, 5)])
def test_hand_computed_values(x, base, a, b, c):
    expected = a * base + b * (x - 0.5) - c
    assert forrester(x, a, b, c) == pytest.approx(expected)
    values = forrester(np.array([[x]]), a, b, c)
    assert values.shape == (1,) and values.dtype == np.float64
    assert values[0] == pytest.approx(expected)


@pytest.mark.parametrize("X", [[0, 1], np.zeros((2, 2)), [], [[np.nan]], [[np.inf]], [[-0.01]], [[1.01]]])
def test_reject_ambiguous_invalid_inputs(X):
    with pytest.raises(ValueError):
        forrester(X, 1, 0, 0)


def manifest(**kwargs):
    return generate_manifest(**{"num_meta_tasks": 3, "num_target_tasks": 3, "samples_per_task": 8,
                               "task_seed": 44, "observation_seed": 45, **kwargs})


def test_manifest_determinism_and_independent_streams():
    first = manifest()
    assert first == manifest()
    assert first["target_tasks"] != manifest(task_seed=46)["target_tasks"]
    assert first["target_tasks"] == manifest(num_meta_tasks=8)["target_tasks"]
    assert first["meta_tasks"] == manifest(num_target_tasks=8)["meta_tasks"]
    changed_observations = manifest(observation_seed=47)
    assert first["target_tasks"] == changed_observations["target_tasks"]
    assert first["meta_tasks"][0]["X"] != changed_observations["meta_tasks"][0]["X"]
    assert {t['task_id'] for t in first['meta_tasks']}.isdisjoint(t['task_id'] for t in first['target_tasks'])


def test_full_source_sample_count_and_training_adapter():
    data = manifest(num_meta_tasks=200, samples_per_task=32)
    assert len(data["meta_tasks"]) == 200
    for entry in data["meta_tasks"]:
        task = ForresterTask(**{k: entry[k] for k in ("task_id", "a", "b", "c")})
        adapted = task.as_training_task(entry["X"], entry["y"])
        assert adapted.X.shape == (32, 1) and adapted.y.shape == (32,)
        np.testing.assert_allclose(adapted.objective(adapted.X), adapted.y)


@pytest.mark.parametrize("kind", ["id", "parameters", "future_label", "source_label", "range", "generator"])
def test_manifest_leakage_and_corruption_detection(kind):
    data = copy.deepcopy(manifest())
    if kind == "id":
        data["target_tasks"][0]["task_id"] = data["meta_tasks"][0]["task_id"]
    elif kind == "parameters":
        data["target_tasks"][0].update({k: data["meta_tasks"][0][k] for k in ("a", "b", "c")})
    elif kind == "future_label":
        data["target_tasks"][0]["y"] = [1]
    elif kind == "source_label":
        data["meta_tasks"][0]["y"][0] += 1
    elif kind == "range":
        data["target_tasks"][0]["a"] = 99.0
    else:
        data["generation"] = "author_compatible_unverified"
    with pytest.raises(ValueError):
        validate_manifest(data)
