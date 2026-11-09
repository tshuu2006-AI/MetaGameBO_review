"""Interfaces shared by synthetic and external AutoML benchmarks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkTask:
    """A black-box minimisation task and optional offline observations."""

    task_id: int
    objective: Callable[[np.ndarray], np.ndarray]
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    f_min: float | None = None
    f_max: float | None = None


class BaseBenchmark(ABC):
    """Adapter contract for HPOBench, NASBench201, HPO-B, and mocks."""

    bounds: np.ndarray
    f_min: float
    f_max: float

    @abstractmethod
    def get_meta_train_tasks(
        self, num_tasks: int, num_samples_per_task: int = 512
    ) -> list[BenchmarkTask]:
        raise NotImplementedError

    @abstractmethod
    def get_target_task(
        self, task_id: int = 0, evaluation_seed: int | None = None
    ) -> BenchmarkTask:
        raise NotImplementedError


class MockBenchmark(BaseBenchmark):
    """Deterministic stand-in demonstrating the AutoML wrapper API."""

    def __init__(self, dim: int = 3, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self.bounds = np.tile(np.array([[0.0, 1.0]]), (dim, 1))
        self.f_min, self.f_max = 0.0, float(dim)

    def _objective(self, task_id: int) -> Callable[[np.ndarray], np.ndarray]:
        rng = np.random.default_rng(self.seed + task_id)
        optimum = rng.uniform(0.1, 0.9, self.dim)

        def evaluate(X: np.ndarray) -> np.ndarray:
            X = np.atleast_2d(np.asarray(X, dtype=np.float64))
            return np.sum((X - optimum) ** 2, axis=1)

        return evaluate

    def _task(self, task_id: int, samples: int | None) -> BenchmarkTask:
        objective = self._objective(task_id)
        X = y = None
        if samples is not None:
            rng = np.random.default_rng(self.seed * 1009 + task_id)
            X = rng.uniform(0.0, 1.0, (samples, self.dim))
            y = objective(X)
        return BenchmarkTask(task_id, objective, X, y, self.f_min, self.f_max)

    def get_meta_train_tasks(
        self, num_tasks: int, num_samples_per_task: int = 512
    ) -> list[BenchmarkTask]:
        return [self._task(i, num_samples_per_task) for i in range(num_tasks)]

    def get_target_task(
        self, task_id: int = 0, evaluation_seed: int | None = None
    ) -> BenchmarkTask:
        return self._task(task_id, None)
