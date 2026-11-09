"""Forrester task family compatible with the official MALIBO benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForresterTask:
    a: float
    b: float
    c: float

    def evaluate(self, x: np.ndarray | float) -> np.ndarray:
        return forrester_function(x, self)


def forrester_function(x: np.ndarray | float, task: ForresterTask) -> np.ndarray:
    """Return f(x)=a(6x-2)^2 sin(12x-4)+b(x-0.5)-c."""
    values = np.asarray(x, dtype=np.float64)
    high_fidelity = np.square(6.0 * values - 2.0) * np.sin(12.0 * values - 4.0)
    return task.a * high_fidelity + task.b * (values - 0.5) - task.c


def sample_task(seed: int) -> ForresterTask:
    rng = np.random.default_rng(seed)
    return ForresterTask(
        a=float(rng.uniform(0.2, 3.0)),
        b=float(rng.uniform(-5.0, 15.0)),
        c=float(rng.uniform(-5.0, 5.0)),
    )


def task_bounds() -> tuple[tuple[float, float], ...]:
    return ((0.2, 3.0), (-5.0, 15.0), (-5.0, 5.0))


def build_malibo_meta_data(
    n_tasks: int = 256,
    points_per_task: int = 32,
    seed: int = 44,
) -> tuple[Any, dict[int, list[Any]]]:
    """Build MALIBO's expected ``dict[task_id, list[Evaluation]]`` metadata.

    Imports the official MALIBO dependencies lazily so the PFN-only experiment
    and the unit tests do not need blackboxopt/parameterspace installed.
    """
    try:
        import parameterspace as ps
        from blackboxopt import EvaluationSpecification
    except ImportError as exc:
        raise ImportError(
            "Building MALIBO meta-data requires parameterspace and blackboxopt. "
            "Install requirements-malibo.txt first."
        ) from exc

    search_space = ps.ParameterSpace()
    search_space.add(ps.ContinuousParameter(name="x", bounds=[0.0, 1.0]))

    descriptor_space = ps.ParameterSpace()
    descriptor_space.add(ps.ContinuousParameter(name="a", bounds=[0.2, 3.0]))
    descriptor_space.add(ps.ContinuousParameter(name="b", bounds=[-5.0, 15.0]))
    descriptor_space.add(ps.ContinuousParameter(name="c", bounds=[-5.0, 5.0]))

    rng = np.random.default_rng(seed)
    meta_data: dict[int, list[Any]] = {}

    for task_id in range(1, n_tasks + 1):
        descriptor = descriptor_space.sample(rng=rng)
        task = ForresterTask(
            a=float(descriptor["a"]),
            b=float(descriptor["b"]),
            c=float(descriptor["c"]),
        )
        observations = []
        for _ in range(points_per_task):
            configuration = search_space.sample(rng=rng)
            x_value = float(configuration["x"])
            loss = float(forrester_function(x_value, task))
            evaluation_spec = EvaluationSpecification(configuration=configuration)
            evaluation = evaluation_spec.create_evaluation(
                objectives={"loss": loss},
                user_info={"task_uid": task_id},
            )
            observations.append(evaluation)
        meta_data[task_id] = observations

    return search_space, meta_data


def exact_task_bounds(
    task: ForresterTask,
    grid_size: int = 20_001,
) -> tuple[float, float]:
    """Dense-grid approximation to the task's global range on x in [0,1]."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    values = forrester_function(grid, task)
    return float(np.min(values)), float(np.max(values))
