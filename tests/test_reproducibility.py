"""Paired initialization, identical trajectories and safe local resume checks."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.forrester import ForresterTask
from experiments.evaluate import run_experiment, run_single_task
from models.random_search import RandomSearch


def numeric_observations(path):
    rows = list(csv.DictReader((Path(path) / "observations.csv").open()))
    return [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in rows]


def test_repeat_configuration_histories_and_metrics_identical(small_config, tmp_path):
    first = run_experiment(small_config)
    second_config = copy.deepcopy(small_config)
    second_config["output_dir"] = str(tmp_path / "second")
    second = run_experiment(second_config)
    assert numeric_observations(first["output_directory"]) == numeric_observations(second["output_directory"])
    for key in ("metrics", "exploratory_cvar", "manifest_sha256", "config_hash", "total_objective_evaluations"):
        assert first["summary"][key] == second["summary"][key]
    assert (Path(first["output_directory"]) / "task_manifest.json").read_bytes() == (Path(second["output_directory"]) / "task_manifest.json").read_bytes()


def test_initialization_independent_of_optimizer_rng():
    task = ForresterTask("paired", 1, 0, 0)
    a = run_single_task(RandomSearch([[0, 1]], seed=123), task, 6, 2, 44)
    b = run_single_task(RandomSearch([[0, 1]], seed=456), task, 6, 2, 44)
    assert [r['x'] for r in a.observations[:2]] == [r['x'] for r in b.observations[:2]]
    assert [r['x'] for r in a.observations[2:]] != [r['x'] for r in b.observations[2:]]


def test_target_isolation_even_when_reusing_an_instance():
    model = RandomSearch([[0, 1]], seed=44)
    first = run_single_task(model, ForresterTask("one", 1, 0, 0), 4, 2, 0)
    second = run_single_task(model, ForresterTask("two", 2, 3, 4), 4, 2, 0)
    assert len(first.observations) == len(second.observations) == len(model.history.y) == 4
    np.testing.assert_array_equal(model.history.arrays[1], [r['y'] for r in second.observations])


def test_no_silent_overwrite_and_completed_resume(small_config):
    first = run_experiment(small_config)
    root = Path(first["output_directory"])
    caches = {p.name: p.read_bytes() for p in (root / "runs").glob("*.json")}
    with pytest.raises(FileExistsError, match="--resume"):
        run_experiment(small_config)
    second = run_experiment(small_config, resume=True,
                            optimizer_factory=lambda **kw: pytest.fail("Completed runs must not instantiate models"))
    assert second["skipped_completed_runs"] == 4
    assert second["summary"] == first["summary"]
    assert caches == {p.name: p.read_bytes() for p in (root / "runs").glob("*.json")}


def test_partial_resume_runs_only_missing_cache(small_config):
    first = run_experiment(small_config)
    root = Path(first["output_directory"])
    cache = sorted((root / "runs").glob("*.json"))[0]
    # Delete only a test-owned temporary cache to simulate an interrupted run.
    cache.unlink()
    created = []
    def factory(**kwargs):
        created.append(kwargs)
        return RandomSearch(**kwargs)
    second = run_experiment(small_config, resume=True, optimizer_factory=factory)
    assert second["skipped_completed_runs"] == 3 and len(created) == 1
    assert second["summary"]["metrics"] == first["summary"]["metrics"]


@pytest.mark.parametrize("kind", ["manifest", "cache"])
def test_tampered_resume_fails(small_config, kind):
    first = run_experiment(small_config)
    root = Path(first["output_directory"])
    path = root / "task_manifest.json" if kind == "manifest" else next((root / "runs").glob("*.json"))
    data = json.loads(path.read_text())
    if kind == "manifest":
        data["target_tasks"][0]["a"] += 0.01
    else:
        data["result"]["observations"][0]["y"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="modified|Corrupt"):
        run_experiment(small_config, resume=True)
