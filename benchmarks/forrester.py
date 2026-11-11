# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
"""Forrester benchmark from the upstream MetaGameBO / MALIBO code (scalar version)."""

import math
import copy
from dataclasses import dataclass
from typing import Optional, Dict, Union, Any, List

import numpy as np
import parameterspace as ps
from blackboxopt import Evaluation, EvaluationSpecification


@dataclass(frozen=True)
class Task:
    """A Forrester task identified by its parameters.

    Attributes:
        uid: Unique task identifier (0 is the target task).
        descriptors: Function parameters ``{"a": ..., "b": ..., "c": ...}``.
    """
    uid: Union[str, int]
    descriptors: Dict[str, Any]


def forrester_function(x: float, a: float, b: float, c: float) -> float:
    """One-dimensional Forrester function.

    ``f(x) = a (6x - 2)^2 sin(12x - 4) + b (x - 0.5) - c``; multimodal, with one
    global minimum, one local minimum and a zero-gradient inflection point.

    Args:
        x: Input in [0, 1].
        a: Amplitude of the high-fidelity term.
        b: Slope of the linear term.
        c: Constant offset.

    Returns:
        Function value at ``x``.
    """
    y_high = math.pow(6 * x - 2, 2) * math.sin(12 * x - 4)
    y = a * y_high + b * (x - 0.5) - c
    return y


class Forrester:
    """Forrester benchmark: one target task plus meta tasks with random parameters.

    Parameters are drawn uniformly from a in [0.2, 3], b in [-5, 15], c in [-5, 5].

    Args:
        n_data_per_task: Number of meta-data points per meta task; its length
            sets the number of meta tasks.
        seed: Seed for task parameters and meta-data sampling.
    """

    def __init__(
        self, n_data_per_task=[4] * 128, seed: Optional[int] = None
    ):
        self.seed = seed
        self._n_data_per_task = n_data_per_task

        descriptors = ps.ParameterSpace()
        descriptors.add(ps.ContinuousParameter(name="a", bounds=[0.2, 3]))
        descriptors.add(ps.ContinuousParameter(name="b", bounds=[-5, 15]))
        descriptors.add(ps.ContinuousParameter(name="c", bounds=[-5, 5]))

        self._search_space = ps.ParameterSpace()
        self._search_space.add(ps.ContinuousParameter(name="x", bounds=[0, 1]))

        self._target_task, self._meta_tasks = self.create_tasks(
            descriptors, len(n_data_per_task), seed
        )

    @property
    def target_task(self) -> Task:
        """The target task (uid 0)."""
        return self._target_task

    @property
    def meta_tasks(self) -> Dict[int, Task]:
        """Meta tasks keyed by uid 1..N."""
        return self._meta_tasks

    @property
    def n_data_per_task(self) -> List:
        """Number of meta-data points per meta task."""
        return self._n_data_per_task

    @property
    def search_space(self) -> ps.ParameterSpace:
        """Search space with a single parameter ``x`` in [0, 1]."""
        return self._search_space

    @staticmethod
    def create_random_task(
        uid,
        descriptors: ps.ParameterSpace,
        seed: Optional[int] = None,
    ):
        """Create a task with random parameters.

        Args:
            uid: Task identifier.
            descriptors: Parameter space of ``a``, ``b``, ``c``.
            seed: Seed or generator for sampling.

        Returns:
            The new task.
        """
        prng = np.random.default_rng(seed)
        return Task(uid, descriptors.sample(rng=prng))

    @staticmethod
    def create_tasks(
        descriptors,
        num_meta_tasks,
        seed: Optional[int] = None,
    ):
        """Create the target task and the meta tasks.

        Args:
            descriptors: Parameter space of ``a``, ``b``, ``c``.
            num_meta_tasks: Number of meta tasks.
            seed: Random seed.

        Returns:
            Tuple ``(target_task, meta_tasks)``, with meta tasks keyed by uid 1..N.
        """
        prng = np.random.default_rng(seed)
        target_task = Forrester.create_random_task(0, descriptors, prng)
        meta_tasks = {
            uid: Forrester.create_random_task(uid, descriptors, prng)
            for uid in range(1, num_meta_tasks + 1)
        }
        return target_task, meta_tasks

    def __call__(
        self,
        eval_spec: EvaluationSpecification,
        task_uid: Optional[int] = None,
    ) -> Evaluation:
        """Evaluate a configuration on one task.

        Args:
            eval_spec: Specification holding the configuration ``{"x": ...}``.
            task_uid: Meta task to evaluate; ``None`` means the target task.

        Returns:
            Evaluation with objective ``"loss"`` and ``task_uid`` in ``user_info``.
        """
        task = self.target_task if task_uid is None else self.meta_tasks[task_uid]

        config = eval_spec.configuration
        objective_values = forrester_function(**config, **task.descriptors)

        return eval_spec.create_evaluation(
            objectives={"loss": objective_values},
            user_info={"task_uid": task_uid}
        )

    def get_meta_data(
        self, seed: Optional[int] = None
    ):
        """Sample meta-data: uniform random evaluations on every meta task.

        Args:
            seed: Sampling seed; defaults to the benchmark seed.

        Returns:
            Dict mapping each meta-task uid to its list of evaluations.
        """
        seed = copy.copy(self.seed) if seed is None else seed
        prng = np.random.default_rng(seed)


        meta_data: Dict[Union[str, int], List[Evaluation]] = dict()

        for (uid, task), n_data in zip(self.meta_tasks.items(), self.n_data_per_task):
            meta_data[uid] = []
            for _ in range(n_data):
                config = self.search_space.sample(rng=prng)
                eval_spec = EvaluationSpecification(configuration=config)
                evaluation = self.__call__(eval_spec, task_uid=uid)
                meta_data[uid].append(evaluation)

        return meta_data