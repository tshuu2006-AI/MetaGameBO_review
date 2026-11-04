"""Vectorized Forrester family and deterministic, disjoint task manifests.

Written independently from the mathematical benchmark specification. This is
not an implementation of the MetaGameBO method or an author-RNG-compatible mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from models.base_model import Task, validate_points, validate_values

DEFAULT_RANGES = {"a": [0.2, 3.0], "b": [-5.0, 15.0], "c": [-5.0, 5.0]}


def derive_seed(seed: int, *labels) -> int:
    """Derive a stable uint32 stream seed from a nonnegative integer and labels."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    payload = json.dumps([int(seed), *labels], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def forrester(X, a: float, b: float, c: float):
    """Evaluate f on scalar x or a nonempty (n,1) batch in [0,1].

    Parameters are fixed finite a>0, b and c. Return float for a scalar input,
    otherwise a float64 vector (n,). Ambiguous flat vectors are rejected.
    """
    params = np.asarray([a, b, c], dtype=np.float64)
    if params.shape != (3,) or not np.isfinite(params).all() or a <= 0:
        raise ValueError("a, b, c must be finite and a must be positive")
    data = np.asarray(X, dtype=np.float64)
    scalar = data.ndim == 0
    batch = np.array([[float(data)]]) if scalar else data
    points = validate_points(batch, np.array([[0.0, 1.0]]))
    x = points[:, 0]
    values = a * np.square(6.0 * x - 2.0) * np.sin(12.0 * x - 4.0) + b * (x - 0.5) - c
    if not np.isfinite(values).all():
        raise ValueError("Forrester evaluation produced nonfinite values")
    return float(values[0]) if scalar else values.astype(np.float64)


@dataclass(frozen=True)
class ForresterTask:
    """Evaluator-side task descriptor, never passed to a target optimizer."""

    task_id: str
    a: float
    b: float
    c: float

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        forrester(0.0, self.a, self.b, self.c)

    def evaluate(self, X) -> np.ndarray:
        """Evaluate a batch (n,1), returning only its n objective values."""
        if np.asarray(X).ndim != 2:
            raise ValueError("evaluate expects a batch (n,1); use evaluate_scalar for a scalar")
        return forrester(X, self.a, self.b, self.c)

    def evaluate_scalar(self, x: float) -> float:
        """Evaluate exactly one scalar coordinate."""
        if np.asarray(x).ndim != 0:
            raise ValueError("evaluate_scalar expects a scalar")
        return forrester(x, self.a, self.b, self.c)

    def as_training_task(self, X, y) -> Task:
        """Adapt source observations to the existing Task interface using copies."""
        points = validate_points(X, np.array([[0.0, 1.0]]))
        values = validate_values(y, len(points))
        return Task(objective=self.evaluate, X=points, y=values)

    def to_dict(self) -> dict:
        """Return stable JSON-compatible task descriptors (no oracle values)."""
        return {"task_id": self.task_id, "a": self.a, "b": self.b, "c": self.c}


def _generate_tasks(count, seed, role, parameter_ranges=None):
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("task count must be a nonnegative integer")
    ranges = DEFAULT_RANGES if parameter_ranges is None else parameter_ranges
    if set(ranges) != {"a", "b", "c"}:
        raise ValueError("parameter_ranges must contain exactly a, b and c")
    for key, value in ranges.items():
        bounds = np.asarray(value, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid parameter range: {key}")
    if ranges["a"][0] <= 0:
        raise ValueError("The a range must be positive")
    derive_seed(seed, role)
    tasks = []
    for index in range(count):
        rng = np.random.default_rng(derive_seed(seed, "task_parameters", role, index))
        parameters = {key: float(rng.uniform(*ranges[key])) for key in ("a", "b", "c")}
        tasks.append(ForresterTask(f"{role}-{index:05d}", **parameters))
    return tasks


def generate_meta_tasks(count: int, seed: int, parameter_ranges=None):
    """Generate source task descriptors, stable when the target count changes."""
    return _generate_tasks(count, seed, "meta", parameter_ranges)


def generate_target_tasks(count: int, seed: int, parameter_ranges=None):
    """Generate target descriptors independently of source count and observations."""
    return _generate_tasks(count, seed, "target", parameter_ranges)


def generate_manifest(num_meta_tasks: int, num_target_tasks: int, samples_per_task: int,
                      task_seed: int, observation_seed: int, parameter_ranges=None,
                      bounds=None) -> dict:
    """Build evaluator-owned task specs and reproducible source data, without target labels."""
    if isinstance(samples_per_task, bool) or not isinstance(samples_per_task, Integral) or samples_per_task <= 0:
        raise ValueError("samples_per_task must be a positive integer")
    domain = np.array([[0.0, 1.0]] if bounds is None else bounds, dtype=float)
    if (domain.shape != (1, 2) or not np.isfinite(domain).all()
            or not 0 <= domain[0, 0] < domain[0, 1] <= 1):
        raise ValueError("Forrester bounds must be a nonempty subinterval of [0,1]")
    derive_seed(observation_seed, "observations")
    meta = generate_meta_tasks(num_meta_tasks, task_seed, parameter_ranges)
    targets = generate_target_tasks(num_target_tasks, task_seed, parameter_ranges)
    if not targets:
        raise ValueError("At least one target task is required")
    manifest = {"schema_version": 1, "generation": "independent_numpy_default_rng_v1",
                "task_seed": int(task_seed), "observation_seed": int(observation_seed),
                "samples_per_task": int(samples_per_task), "bounds": domain.tolist(),
                "parameter_ranges": {k: list(v) for k, v in (DEFAULT_RANGES if parameter_ranges is None else parameter_ranges).items()},
                "meta_tasks": [], "target_tasks": [t.to_dict() for t in targets]}
    for task in meta:
        rng = np.random.default_rng(derive_seed(observation_seed, "source_observations", task.task_id))
        X = rng.uniform(domain[0, 0], domain[0, 1], size=(samples_per_task, 1))
        manifest["meta_tasks"].append({**task.to_dict(), "X": X.tolist(), "y": task.evaluate(X).tolist()})
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    """Reject overlapping tasks, target future labels, and inconsistent source samples."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported task manifest schema")
    if manifest.get("generation") != "independent_numpy_default_rng_v1":
        raise ValueError("Unrecognized manifest generator; author RNG equivalence is not assumed")
    derive_seed(manifest["task_seed"], "manifest")
    derive_seed(manifest["observation_seed"], "manifest")
    _generate_tasks(0, manifest["task_seed"], "validation", manifest["parameter_ranges"])
    domain = np.asarray(manifest["bounds"], dtype=float)
    if domain.shape != (1, 2) or not np.isfinite(domain).all() or not 0 <= domain[0, 0] < domain[0, 1] <= 1:
        raise ValueError("Manifest bounds must lie in [0,1]")
    if not manifest.get("target_tasks"):
        raise ValueError("Manifest must contain target tasks")
    ids, descriptors = set(), set()
    for role in ("meta_tasks", "target_tasks"):
        for entry in manifest[role]:
            allowed = {"task_id", "a", "b", "c"} | ({"X", "y"} if role == "meta_tasks" else set())
            if set(entry) != allowed:
                raise ValueError(f"Unexpected/missing fields in {role}; target labels/oracles are forbidden")
            task = task_from_spec(entry)
            descriptor = (task.a, task.b, task.c)
            if any(not manifest["parameter_ranges"][key][0] <= getattr(task, key) <= manifest["parameter_ranges"][key][1]
                   for key in ("a", "b", "c")):
                raise ValueError("Task descriptors are outside the declared parameter ranges")
            if task.task_id in ids or descriptor in descriptors:
                raise ValueError("Data leakage or duplicate task: IDs and parameter triples must be disjoint")
            ids.add(task.task_id)
            descriptors.add(descriptor)
            if role == "meta_tasks":
                X = validate_points(entry["X"], domain, expected_n=manifest["samples_per_task"])
                y = validate_values(entry["y"], len(X))
                if not np.allclose(y, task.evaluate(X), rtol=1e-12, atol=1e-12):
                    raise ValueError("Source values are inconsistent with their task")


def task_from_spec(spec: dict) -> ForresterTask:
    """Construct an evaluator task from the descriptor fields of a manifest entry."""
    return ForresterTask(**{k: spec[k] for k in ("task_id", "a", "b", "c")})
