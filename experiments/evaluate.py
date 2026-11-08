"""Shared Forrester evaluation engine: evaluator owns tasks, labels and oracle.

Future optimizers receive only bounds, a seed, optional source Task objects,
and completed observations. No target TaskSpec or oracle is passed to them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Optional

import numpy as np
import yaml

from benchmarks.forrester import derive_seed, generate_manifest, task_from_spec, validate_manifest
from experiments.metrics import aggregate_runs, best_so_far, empirical_cvar, estimate_reference_range, normalized_regret, simple_regret
from experiments.tracking import RunTracker
from models.base_model import validate_budget, validate_points, validate_values
from models.random_search import RandomSearch

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_COLUMNS = ["run_id", "query_index", "step", "phase", "bo_step", "x", "y", "best_so_far",
                       "simple_regret", "normalized_regret", "elapsed_seconds", "task_id", "seed", "algorithm"]
RUN_COLUMNS = ["run_id", "task_id", "seed", "optimizer_seed", "initialization_seed", "algorithm", "budget", "n_init",
               "final_best_raw_objective", "final_simple_regret", "final_normalized_regret", "elapsed_seconds", "wandb_url"]


def content_hash(value) -> str:
    """SHA-256 of canonical strict JSON, rejecting NaN and Infinity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    """Atomically replace a generated JSON file with finite, readable JSON."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict:
    """Capture versions, platform, UTC time, git state and a fingerprint of actual code."""
    packages = {}
    for package in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "matplotlib", "PyYAML", "pytest", "wandb",
                    "torch", "pfns4bo", "ConfigSpace", "botorch", "gpytorch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in ("benchmarks", "models", "experiments") for path in sorted((ROOT / folder).glob("*.py"))}
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"), "git_status": _git("status", "--short"),
            "source_sha256": hashes, "source_fingerprint": content_hash(hashes)}


def validate_config(config: dict, allowed_models=("random", "lfbo", "pfn")) -> dict:
    """Validate a complete config, failing on unsupported budgets and unresolved protocols."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    result = copy.deepcopy(config)
    if result.get("schema_version") != 1 or result.get("model") not in allowed_models:
        raise ValueError(f"Unsupported model or schema; allowed models: {', '.join(allowed_models)}")
    required_sections = {"schema_version", "protocol_label", "model", "output_dir", "benchmark", "evaluation", "tracking"}
    if not required_sections <= set(result) or set(result) - required_sections - {"model_options"}:
        raise ValueError("Unsupported or missing top-level config keys")
    if any(not isinstance(result[key], dict) for key in ("benchmark", "evaluation", "tracking")):
        raise ValueError("benchmark, evaluation and tracking must be mappings")
    b, e, tracking = result["benchmark"], result["evaluation"], result["tracking"]
    if not isinstance(result.get("model_options", {}), dict):
        raise ValueError("model_options must be a mapping")
    if {"bounds", "seed"} & set(result.get("model_options", {})):
        raise ValueError("model_options cannot override paired bounds or optimizer seed")
    if set(b) != {"bounds", "parameter_ranges", "num_meta_tasks", "num_target_tasks", "samples_per_task", "task_seed", "observation_seed", "manifest_path"}:
        raise ValueError("Unsupported or missing benchmark config keys")
    if set(e) != {"budget", "budget_mode", "n_init", "seeds", "meta_train", "cvar_alpha", "oracle_grid_size", "oracle_xatol", "regret_tolerance"}:
        raise ValueError("Unsupported or missing evaluation config keys")
    if set(tracking) != {"mode", "project", "entity", "public_project_confirmed"}:
        raise ValueError("Tracking config must contain only mode/project/entity/public_project_confirmed; credentials belong outside YAML")
    if e.get("budget_mode") != "total":
        raise ValueError("Only budget_mode=total is supported; initialization is included")
    validate_budget(e["budget"], e["n_init"])
    seeds = e["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation.seeds must be a nonempty list of unique integers")
    for seed in [*seeds, b["task_seed"], b["observation_seed"]]:
        derive_seed(seed, "validation")
    for name in ("num_meta_tasks", "num_target_tasks", "samples_per_task"):
        value = b[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "num_meta_tasks" else 1):
            raise ValueError(f"Invalid {name}")
    empirical_cvar([1.0], e["cvar_alpha"])
    if not isinstance(e.get("meta_train"), bool):
        raise ValueError("evaluation.meta_train must be a boolean")
    if result["model"] == "lfbo":
        from models.lfbo import validate_lfbo_options
        validate_lfbo_options(result.get("model_options", {}))
        if e["meta_train"]:
            raise ValueError("LFBO is non-meta; evaluation.meta_train must be false")
    if result["model"] == "pfn":
        from models.pfn import validate_pfn_options
        result["model_options"] = validate_pfn_options(result.get("model_options", {}))
        if e["meta_train"]:
            raise ValueError("PFN uses frozen pretrained weights; evaluation.meta_train must be false")
    if not np.isfinite(e["regret_tolerance"]) or e["regret_tolerance"] < 0:
        raise ValueError("regret_tolerance must be finite and nonnegative")
    if isinstance(e["oracle_grid_size"], bool) or not isinstance(e["oracle_grid_size"], int) or e["oracle_grid_size"] < 257:
        raise ValueError("oracle_grid_size must be an integer >=257")
    if not np.isfinite(e["oracle_xatol"]) or e["oracle_xatol"] <= 0:
        raise ValueError("oracle_xatol must be finite and positive")
    if tracking.get("mode") not in {"disabled", "offline", "online"}:
        raise ValueError("Invalid W&B mode")
    if result.get("protocol_label") not in {"development", "smoke_test", "paper_candidate_UNVERIFIED"}:
        raise ValueError("A recognized protocol_label is required")
    # Reuse benchmark validation even when a supplied manifest will be loaded.
    generate_manifest(0, 1, 1, b["task_seed"], b["observation_seed"], b["parameter_ranges"], b["bounds"])
    return result


@dataclass
class RunResult:
    """Completed measured trajectory and evaluator-only references for one task/seed."""
    run_id: str
    task_id: str
    seed: int
    optimizer_seed: Optional[int]
    initialization_seed: int
    algorithm: str
    observations: list
    final_metrics: dict
    reference: dict
    elapsed_seconds: float
    wandb: dict
    started_at_utc: str
    completed_at_utc: str
    optimizer_diagnostics: Optional[dict] = None


def _check_history(optimizer, expected_X, expected_y):
    X, y = optimizer.history.arrays
    if len(expected_y) == 0:
        if X.size or y.size:
            raise ValueError("Optimizer history is not empty after reset (changed, leaked or reordered observations)")
    elif not np.array_equal(X, np.asarray(expected_X)) or not np.array_equal(y, np.asarray(expected_y)):
        raise ValueError("Optimizer changed, leaked or reordered observations outside observe()")


def run_single_task(optimizer, task, budget: int, n_init: int, seed: int, run_metadata: Optional[dict] = None,
                    *, initial_points=None, reference=None, oracle_grid_size=4097, oracle_xatol=1e-12,
                    regret_tolerance=1e-8, tracker=None) -> RunResult:
    """Run exactly budget evaluations, including initialization, without revealing target descriptors.

    Parameters task/reference belong to the evaluator. Only suggest/observe/reset
    and the optimizer's public history/bounds are used. Return a measured RunResult;
    invalid proposals fail before objective evaluation. Timing excludes oracle setup.
    """
    validate_budget(budget, n_init)
    metadata = {} if run_metadata is None else dict(run_metadata)
    algorithm = metadata.get("algorithm", "random")
    optimizer_seed = metadata.get("optimizer_seed", getattr(optimizer, "seed", None))
    init_seed = derive_seed(seed, "initialization", task.task_id)
    run_id = metadata.get("run_id", content_hash([task.task_id, seed, algorithm])[:16])
    if optimizer.bounds.shape != (1, 2):
        raise ValueError("Forrester evaluation requires one-dimensional bounds")
    if np.any(optimizer.bounds < 0) or np.any(optimizer.bounds > 1):
        raise ValueError("Forrester search bounds must lie in [0,1]")
    optimizer.reset()
    _check_history(optimizer, [], [])
    if n_init:
        if initial_points is None:
            initial_points = np.random.default_rng(init_seed).uniform(*optimizer.bounds[0], size=(n_init, 1))
        initial_points = validate_points(initial_points, optimizer.bounds, n_init)
    elif initial_points is not None and np.asarray(initial_points).shape != (0, 1):
        raise ValueError("n_init=0 requires no initial points")
    ref = estimate_reference_range(task, optimizer.bounds[0], oracle_grid_size, oracle_xatol) if reference is None else reference
    expected_X, expected_y, rows = [], [], []
    started_at_utc = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    for index in range(budget):
        _check_history(optimizer, expected_X, expected_y)
        phase = "initial" if index < n_init else "sequential"
        X = initial_points[index:index+1].copy() if index < n_init else optimizer.suggest(1)
        _check_history(optimizer, expected_X, expected_y)
        X = validate_points(X, optimizer.bounds, 1)
        y = validate_values(task.evaluate(X), 1)
        optimizer.observe(X.copy(), y.copy())
        expected_X.append(X[0].copy())
        expected_y.append(float(y[0]))
        _check_history(optimizer, expected_X, expected_y)
        best = float(min(expected_y))
        row = {"run_id": run_id, "query_index": index, "step": index+1, "phase": phase,
               "bo_step": 0 if index < n_init else index-n_init+1, "x": float(X[0, 0]), "y": float(y[0]),
               "best_so_far": best, "simple_regret": simple_regret(best, ref["f_star_estimate"], regret_tolerance),
               "normalized_regret": normalized_regret(best, ref["reference_min"], ref["reference_max"], regret_tolerance),
               "elapsed_seconds": perf_counter()-start, "task_id": task.task_id, "seed": seed, "algorithm": algorithm}
        rows.append(row)
        if tracker is not None:
            tracker.log(row)
    metrics = {"final_best_raw_objective": rows[-1]["best_so_far"], "final_simple_regret": rows[-1]["simple_regret"],
               "final_normalized_regret": rows[-1]["normalized_regret"], "budget": budget, "n_init": n_init}
    return RunResult(run_id, task.task_id, seed, optimizer_seed, init_seed, algorithm, rows, metrics, ref,
                     perf_counter()-start, {"mode": "disabled", "id": None, "url": None, "uploaded": False},
                     started_at_utc, datetime.now(timezone.utc).isoformat(),
                     copy.deepcopy(getattr(optimizer, "diagnostics", None)))


def _write_csv(path, rows, fields):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_curves(results, directory: Path):
    """Draw measured raw-objective/regret means ± sample SE; never draw unrun baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for key, name, label in [("best_so_far", "best_so_far_curve.png", "Best observed raw objective (minimize)"),
                             ("normalized_regret", "regret_curve.png", "Normalized simple regret (fixed task range)")]:
        values = np.array([[row[key] for row in result.observations] for result in results])
        mean = values.mean(axis=0)
        steps = np.arange(1, values.shape[1]+1)
        figure, axes = plt.subplots(figsize=(7.5, 4.5))
        axes.plot(steps, mean, label=f"{results[0].algorithm}; n={len(results)} task-seed pair(s)")
        if len(results) > 1:
            se = values.std(axis=0, ddof=1) / np.sqrt(len(results))
            axes.fill_between(steps, mean-se, mean+se, alpha=0.2, label="±1 descriptive SE")
        axes.set(xlabel="Total objective evaluations (initialization included)", ylabel=label,
                 title=f"{results[0].algorithm} — exploratory Forrester evaluation")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.2)
        figure.tight_layout()
        path = directory / name
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(path)
    return paths


def _read_completed(path, expected, task, evaluation):
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("content_sha256", None)
    if digest != content_hash(payload) or payload.get("status") != "completed":
        raise ValueError(f"Corrupt or incomplete run cache: {path}")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Run cache/config mismatch: {path}")
    result = RunResult(**payload["result"])
    rows = result.observations
    if result.task_id != task.task_id or len(rows) != evaluation["budget"]:
        raise ValueError("Cached trajectory does not match its task/budget")
    artifacts = path.parent / result.run_id
    for filename in ("observations.csv", "measured_result.json", "regret_curve.png", "best_so_far_curve.png"):
        if not (artifacts / filename).is_file():
            raise ValueError(f"Completed run is missing an artifact: {artifacts / filename}")
    if result.optimizer_diagnostics is not None:
        diagnostic_file = artifacts / "optimizer_diagnostics.json"
        if not diagnostic_file.is_file() or json.loads(diagnostic_file.read_text()) != result.optimizer_diagnostics:
            raise ValueError("Completed run has missing or modified optimizer diagnostics")
    values = validate_values([r["y"] for r in rows], len(rows))
    validate_points([[r["x"]] for r in rows], np.asarray(payload["bounds"]), len(rows))
    if not np.array_equal(best_so_far(values), [r["best_so_far"] for r in rows]):
        raise ValueError("Cached best-so-far is inconsistent")
    return result


def run_experiment(config: dict, *, resume=False, optimizer_factory=None) -> dict:
    """Generate/load a paired manifest, run or verify caches, and export real artifacts.

    optimizer_factory(bounds, seed, **model_options) is the extension point. It receives no
    target task metadata; the same runner, initialization and metrics are reused.
    """
    config = validate_config(config, allowed_models=(config["model"],) if optimizer_factory is not None else ("random", "lfbo", "pfn"))
    b, e = config["benchmark"], config["evaluation"]
    env = environment_metadata()
    if config["model"] == "pfn":
        from models.pfn_backend import checkpoint_metadata
        env["pfn_checkpoint"] = checkpoint_metadata(config["model_options"]["checkpoint"], config["model_options"]["model_name"])
    supplied_manifest = b.get("manifest_path")
    if supplied_manifest:
        manifest = json.loads(Path(supplied_manifest).read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if (len(manifest["meta_tasks"]) != b["num_meta_tasks"] or len(manifest["target_tasks"]) != b["num_target_tasks"]
                or manifest["samples_per_task"] != b["samples_per_task"] or manifest["bounds"] != b["bounds"]
                or any(manifest[k] != b[k] for k in ("task_seed", "observation_seed", "parameter_ranges"))):
            raise ValueError("Supplied manifest dimensions/counts/bounds do not match config")
    else:
        manifest = generate_manifest(b["num_meta_tasks"], b["num_target_tasks"], b["samples_per_task"],
                                     b["task_seed"], b["observation_seed"], b["parameter_ranges"], b["bounds"])
    manifest_hash = content_hash(manifest)
    identity = copy.deepcopy({key: value for key, value in config.items() if key != "output_dir"})
    identity["benchmark"].pop("manifest_path", None)
    hash_payload = {"config": identity, "manifest_sha256": manifest_hash,
                    "source_fingerprint": env["source_fingerprint"], "packages": env["packages"]}
    if config["model"] == "pfn":
        hash_payload["checkpoint_sha256"] = env["pfn_checkpoint"]["sha256"]
    config_hash = content_hash(hash_payload)
    directory = Path(config["output_dir"]).resolve() / config_hash[:16]
    if directory.exists():
        if not resume:
            raise FileExistsError(f"Experiment already exists: {directory}. Use --resume to verify/skip completed runs.")
        stored = json.loads((directory / "experiment_identity.json").read_text())
        if stored != {"config_hash": config_hash, "manifest_sha256": manifest_hash}:
            raise ValueError("Existing experiment identity does not match config")
        if content_hash(json.loads((directory / "task_manifest.json").read_text())) != manifest_hash:
            raise ValueError("Existing task manifest was modified")
    else:
        directory.mkdir(parents=True)
        write_json(directory / "experiment_identity.json", {"config_hash": config_hash, "manifest_sha256": manifest_hash})
        write_json(directory / "task_manifest.json", manifest)
        write_json(directory / "environment.json", env)
        (directory / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    run_directory = directory / "runs"
    run_directory.mkdir(exist_ok=True)
    if optimizer_factory is None and config["model"] == "pfn":
        from models.pfn import PFNOptimizer
        factory = PFNOptimizer
    elif optimizer_factory is None and config["model"] == "lfbo":
        from models.lfbo import LFBO
        factory = LFBO
    else:
        factory = RandomSearch if optimizer_factory is None else optimizer_factory
    results, skipped = [], 0
    references = {}
    for spec in manifest["target_tasks"]:
        task = task_from_spec(spec)
        for seed in e["seeds"]:
            run_id = content_hash([config_hash, task.task_id, seed])[:20]
            path = run_directory / f"{run_id}.json"
            expected = {"config_hash": config_hash, "manifest_sha256": manifest_hash, "run_id": run_id}
            if path.exists():
                result = _read_completed(path, expected, task, e)
                results.append(result)
                skipped += 1
                continue
            tracker = RunTracker(config["tracking"], directory / "wandb")
            try:
                optimizer_seed = derive_seed(seed, "optimizer", task.task_id)
                optimizer = factory(bounds=np.array(b["bounds"], dtype=float), seed=optimizer_seed, **config.get("model_options", {}))
                if e["meta_train"]:
                    sources = [task_from_spec(s).as_training_task(s["X"], s["y"]) for s in manifest["meta_tasks"]]
                    optimizer.meta_train(sources)
                info = tracker.start(run_id, {"protocol_label": config["protocol_label"], "algorithm": config["model"],
                                             "task_id": task.task_id, "seed": seed, "optimizer_seed": optimizer_seed,
                                             "config_hash": config_hash, "manifest_sha256": manifest_hash,
                                             "git_commit": env["git_commit"], "source_fingerprint": env["source_fingerprint"],
                                             "versions": env["packages"],
                                             "model_options": config.get("model_options", {}),
                                             "evaluation": e})
                if task.task_id not in references:
                    references[task.task_id] = estimate_reference_range(task, b["bounds"][0], e["oracle_grid_size"], e["oracle_xatol"])
                result = run_single_task(optimizer, task, e["budget"], e["n_init"], seed,
                                         {"run_id": run_id, "algorithm": config["model"], "optimizer_seed": optimizer_seed},
                                         reference=references[task.task_id], regret_tolerance=e["regret_tolerance"], tracker=tracker)
                result.wandb = info
                one_dir = run_directory / run_id
                one_dir.mkdir(exist_ok=True)
                _write_csv(one_dir / "observations.csv", result.observations, OBSERVATION_COLUMNS)
                run_plots = plot_curves([result], one_dir)
                # Stage the measured result as an artifact; completion is saved only after tracking finishes.
                write_json(one_dir / "measured_result.json", asdict(result))
                diagnostic_artifacts = []
                if result.optimizer_diagnostics is not None:
                    diagnostic_file = one_dir / "optimizer_diagnostics.json"
                    write_json(diagnostic_file, result.optimizer_diagnostics)
                    diagnostic_artifacts.append(diagnostic_file)
                tracker.finish(result.final_metrics, [one_dir / "measured_result.json", one_dir / "observations.csv",
                                                      directory / "config_resolved.yaml", directory / "task_manifest.json",
                                                      directory / "environment.json", *run_plots, *diagnostic_artifacts])
                if info["mode"] == "online":
                    result.wandb["uploaded"] = True
                payload = {**expected, "status": "completed", "bounds": b["bounds"], "result": asdict(result),
                           "git_commit": env["git_commit"], "source_fingerprint": env["source_fingerprint"],
                           "environment_file": "environment.json", "config_file": "config_resolved.yaml"}
                write_json(path, {**payload, "content_sha256": content_hash(payload)})
                results.append(result)
            except Exception as exc:
                try:
                    tracker.fail()
                except Exception:
                    pass
                write_json(run_directory / f"{run_id}.failed.json", {**expected, "status": "failed", "task_id": task.task_id,
                                                                  "seed": seed, "exception_type": type(exc).__name__, "error": str(exc)})
                raise RuntimeError(f"Run failed: task={task.task_id}, seed={seed}, config={config_hash}, output={directory}: {exc}") from exc
    rows = [row for result in results for row in result.observations]
    run_rows = [{"run_id": r.run_id, "task_id": r.task_id, "seed": r.seed, "optimizer_seed": r.optimizer_seed,
                 "initialization_seed": r.initialization_seed, "algorithm": r.algorithm, **r.final_metrics,
                 "elapsed_seconds": r.elapsed_seconds, "wandb_url": r.wandb["url"]} for r in results]
    _write_csv(directory / "observations.csv", rows, OBSERVATION_COLUMNS)
    _write_csv(directory / "runs.csv", run_rows, RUN_COLUMNS)
    plot_curves(results, directory)
    summary = {"schema_version": 1, "status": "completed", "protocol_label": config["protocol_label"], "algorithm": config["model"],
               "table1_reproduction": False, "config_hash": config_hash, "manifest_sha256": manifest_hash,
               "n_target_tasks": len(manifest["target_tasks"]), "n_seeds": len(e["seeds"]), "n_runs": len(results),
               "budget_per_run": e["budget"], "n_init_included": e["n_init"], "total_objective_evaluations": len(rows),
               "sampling_unit": "target_task_seed_pair", "metrics": {},
               "runtime_seconds": aggregate_runs([r.elapsed_seconds for r in results]),
               "metric_notes": {"normalization": "fixed_task_dense_grid_refined_min_max",
                                "oracle": "numerical_estimate_not_exact_guarantee", "standard_error": "sample_std_ddof1/sqrt(n); null at n=1",
                                "dependence": "Task-seed pairs reuse tasks; pooled SE is descriptive, not an independent-task confidence interval",
                                "table1_average_loss": "UNVERIFIED; not equated with raw objective, BCE or regret"}}
    for key in ("final_best_raw_objective", "final_simple_regret", "final_normalized_regret"):
        summary["metrics"][key] = aggregate_runs([r.final_metrics[key] for r in results])
    summary["exploratory_cvar"] = {"alpha": e["cvar_alpha"], "metric": "final_normalized_regret", "sampling_unit": "target_task_seed_pair",
                                   "convention": "exact_equal_mass_fractional_upper_tail", "table1_equivalence": "UNVERIFIED",
                                   "value": empirical_cvar([r.final_metrics["final_normalized_regret"] for r in results], e["cvar_alpha"]),
                                   "per_seed_across_tasks": [{"seed": seed, "n_tasks": len(manifest["target_tasks"]),
                                       "value": empirical_cvar([r.final_metrics["final_normalized_regret"] for r in results if r.seed == seed], e["cvar_alpha"])} for seed in e["seeds"]]}
    write_json(directory / "summary.json", summary)
    tracking_runs = [r.wandb for r in results if r.wandb["id"] is not None]
    if tracking_runs:
        write_json(directory / "wandb_run_ids.json", tracking_runs)
    write_json(directory / "completion.json", {"config_hash": config_hash, "n_runs": len(results), "status": "completed"})
    return {"output_directory": str(directory), "summary": summary, "skipped_completed_runs": skipped}
