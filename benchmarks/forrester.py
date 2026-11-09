"""Heterogeneous Forrester task family used by the standalone experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import BaseBenchmark, BenchmarkTask


def forrester(x: np.ndarray) -> np.ndarray:
    """Canonical f(x)=(6x-2)^2 sin(12x-4), x in [0,1]."""
    x = np.asarray(x, dtype=np.float64)
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


@dataclass(frozen=True)
class ForresterTask:
    scale: float
    shift: float
    slope: float
    noise_std: float
    tail_event_probability: float
    tail_event_scale: float


class ForresterBenchmark(BaseBenchmark):
    """Forrester tasks with mild shift/scale and rare upper-tail losses."""

    def __init__(self, num_tasks: int = 12, seed: int = 0, grid_size: int = 20001):
        if num_tasks < 2:
            raise ValueError("num_tasks must be at least 2 for leave-one-task-out")
        self.num_tasks, self.seed = num_tasks, seed
        self.bounds = np.array([[0.0, 1.0]], dtype=np.float64)
        rng = np.random.default_rng(seed)
        self.parameters: list[ForresterTask] = []
        for i in range(num_tasks):
            # Khoảng 20% task là tail task: scale/noise/shock lớn hơn.
            hard = i >= max(1, int(np.floor(0.8 * num_tasks)))
            self.parameters.append(ForresterTask(
                scale=float(rng.uniform(0.85, 1.15) * (1.45 if hard else 1.0)),
                shift=float(rng.uniform(-0.035, 0.035)),
                slope=float(rng.uniform(-1.0, 1.0)),
                noise_std=float(rng.uniform(0.01, 0.04) * (2.5 if hard else 1.0)),
                tail_event_probability=0.08 if hard else 0.015,
                tail_event_scale=1.5 if hard else 0.35,
            ))
        grid = np.linspace(0.0, 1.0, grid_size)[:, None]
        bounds = [self._task_bounds(i, grid) for i in range(num_tasks)]
        self.f_min = float(min(v[0] for v in bounds))
        self.f_max = float(max(v[1] for v in bounds))

    def latent(self, X: np.ndarray, task_id: int) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        p = self.parameters[task_id]
        shifted = np.clip(X[:, 0] + p.shift, 0.0, 1.0)
        return p.scale * forrester(shifted) + p.slope * (X[:, 0] - 0.5)

    def _task_bounds(self, task_id: int, grid: np.ndarray) -> tuple[float, float]:
        y = self.latent(grid, task_id)
        return float(y.min()), float(y.max())

    def _objective(self, task_id: int, stream_seed: int) -> Callable[[np.ndarray], np.ndarray]:
        rng = np.random.default_rng(stream_seed)
        p = self.parameters[task_id]

        def evaluate(X: np.ndarray) -> np.ndarray:
            mean = self.latent(X, task_id)
            gaussian = rng.normal(0.0, p.noise_std, mean.shape)
            shock_mask = rng.random(mean.shape) < p.tail_event_probability
            shocks = shock_mask * rng.exponential(p.tail_event_scale, mean.shape)
            return mean + gaussian + shocks

        return evaluate

    def _build_task(self, task_id: int, samples: int | None) -> BenchmarkTask:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id must be in [0, {self.num_tasks})")
        objective = self._objective(task_id, self.seed * 1_000_003 + task_id)
        X = y = None
        if samples is not None:
            rng = np.random.default_rng(self.seed * 10_007 + task_id)
            X = rng.uniform(0.0, 1.0, (samples, 1))
            y = objective(X)
        grid = np.linspace(0.0, 1.0, 20001)[:, None]
        f_min, f_max = self._task_bounds(task_id, grid)
        return BenchmarkTask(task_id, objective, X, y, f_min, f_max)

    def get_meta_train_tasks(
        self, num_tasks: int, num_samples_per_task: int = 512
    ) -> list[BenchmarkTask]:
        if not 1 <= num_tasks <= self.num_tasks:
            raise ValueError("num_tasks is outside the generated task pool")
        return [self._build_task(i, num_samples_per_task) for i in range(num_tasks)]

    def get_target_task(
        self, task_id: int = 0, evaluation_seed: int | None = None
    ) -> BenchmarkTask:
        task = self._build_task(task_id, None)
        run_seed = 0 if evaluation_seed is None else int(evaluation_seed)
        objective = self._objective(
            task_id, self.seed * 2_000_003 + task_id * 10_007 + run_seed
        )
        return BenchmarkTask(task.task_id, objective, f_min=task.f_min, f_max=task.f_max)
