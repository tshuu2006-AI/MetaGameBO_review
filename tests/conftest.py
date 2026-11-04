"""Small local configurations; no full 100-repeat experiment is run by tests."""
import pytest

from experiments.run_forrester import load_config


@pytest.fixture
def small_config(tmp_path):
    config = load_config("configs/forrester_smoke.yaml")
    config["output_dir"] = str(tmp_path / "results")
    config["benchmark"].update(num_target_tasks=2, num_meta_tasks=2, samples_per_task=3)
    config["evaluation"].update(budget=4, n_init=2, seeds=[0, 1], oracle_grid_size=1025)
    return config
