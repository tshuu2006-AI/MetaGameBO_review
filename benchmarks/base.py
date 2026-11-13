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
