"""Forrester task family used by the public MetaGameBO implementation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseBenchmark, BenchmarkTask


def forrester(x: np.ndarray) -> np.ndarray:
    """Canonical high-fidelity Forrester function."""
    x = np.asarray(x, dtype=np.float64)
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


@dataclass(frozen=True)
class ForresterTask:
    a: float
    b: float
    c: float

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(X, dtype=np.float64))[:, 0]
        return self.a * forrester(x) + self.b * (x - 0.5) - self.c


class ForresterBenchmark(BaseBenchmark):
    """Noiseless source/target tasks matching MetaGameBO's public benchmark.

    a ~ U(0.2, 3), b ~ U(-5, 15), c ~ U(-5, 5). Source observations and
    target tasks use separate deterministic RNG streams to prevent leakage.
    """

    def __init__(self, num_tasks: int = 200, seed: int = 0,
                 grid_size: int = 20001):
        if num_tasks < 1 or grid_size < 3:
            raise ValueError("num_tasks must be positive and grid_size >= 3")
        self.num_tasks, self.seed = num_tasks, seed
        self.bounds = np.array([[0.0, 1.0]])
        self.grid = np.linspace(0.0, 1.0, grid_size)[:, None]
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        self.parameters = [self._draw(rng) for _ in range(num_tasks)]
        extrema = [self._bounds(task) for task in self.parameters]
        self.f_min = min(pair[0] for pair in extrema)
        self.f_max = max(pair[1] for pair in extrema)

    @staticmethod
    def _draw(rng: np.random.Generator) -> ForresterTask:
        return ForresterTask(
            a=float(rng.uniform(0.2, 3.0)),
            b=float(rng.uniform(-5.0, 15.0)),
            c=float(rng.uniform(-5.0, 5.0)),
        )

    def _bounds(self, task: ForresterTask) -> tuple[float, float]:
        values = task.evaluate(self.grid)
        return float(values.min()), float(values.max())

    def get_meta_train_tasks(
        self, num_tasks: int, num_samples_per_task: int = 32
    ) -> list[BenchmarkTask]:
        if not 1 <= num_tasks <= self.num_tasks or num_samples_per_task < 2:
            raise ValueError("invalid source-task or sample count")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1]))
        tasks = []
        for index, parameters in enumerate(self.parameters[:num_tasks]):
            X = rng.uniform(0.0, 1.0, (num_samples_per_task, 1))
            f_min, f_max = self._bounds(parameters)
            tasks.append(BenchmarkTask(
                index, parameters.evaluate, X, parameters.evaluate(X), f_min, f_max
            ))
        return tasks

    def target_parameters(self, seed: int) -> ForresterTask:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 2, int(seed)])
        )
        return self._draw(rng)

    def get_target_task(
        self, task_id: int = 0, evaluation_seed: int | None = None
    ) -> BenchmarkTask:
        run_seed = task_id if evaluation_seed is None else int(evaluation_seed)
        parameters = self.target_parameters(run_seed)
        f_min, f_max = self._bounds(parameters)
        return BenchmarkTask(
            self.num_tasks + run_seed, parameters.evaluate,
            f_min=f_min, f_max=f_max,
        )
