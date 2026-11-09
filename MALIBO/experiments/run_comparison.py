from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from benchmarks.forrester import (
    ForresterTask,
    exact_task_bounds,
    forrester_function,
    sample_task,
    build_malibo_meta_data,
)
from configs.experiment import ExperimentConfig
from experiments.metrics import normalized_regret_curve, summarize_curves
from models.malibo_adapter import MALIBOFactory
from models.pfn_adapter import PFNFactory

LOGGER = logging.getLogger("baseline_experiment")


def _candidate_pool(seed: int, count: int, X_obs: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(0.0, 1.0, size=(count, 1)).astype(np.float64)
    if X_obs.size:
        distances = np.min(
            np.abs(candidates[:, None, :] - X_obs[None, :, :]), axis=(1, 2)
        )
        candidates = candidates[distances > 1e-10]
    if candidates.shape[0] == 0:
        raise RuntimeError("Candidate pool is empty after removing evaluated points.")
    return candidates


def _evaluate_one_run(
    method_name: str,
    method_factory,
    task: ForresterTask,
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    candidate_seeds: list[int],
    config: ExperimentConfig,
    run_seed: int,
) -> np.ndarray:
    X_obs = np.asarray(initial_x, dtype=np.float64).copy()
    y_obs = np.asarray(initial_y, dtype=np.float64).reshape(-1).copy()
    f_min, f_max = exact_task_bounds(task, config.grid_size_for_bounds)

    if method_name == "malibo":
        optimizer = method_factory.new_task_optimizer(seed=run_seed)
        suggest: Callable = lambda X, y, pending, step: method_factory.suggest(
            optimizer, X, y, pending, seed=run_seed * 10_007 + step
        )
    else:
        optimizer = method_factory.new_task_optimizer()
        suggest = lambda X, y, pending, step: method_factory.suggest(
            optimizer, X, y, pending
        )

    history = list(normalized_regret_curve(y_obs, f_min, f_max))
    for step, candidate_seed in enumerate(candidate_seeds):
        X_pending = _candidate_pool(
            seed=candidate_seed,
            count=config.candidate_count,
            X_obs=X_obs,
        )
        next_index = suggest(X_obs, y_obs, X_pending, step)
        if not 0 <= next_index < len(X_pending):
            raise RuntimeError(
                f"{method_name} returned invalid candidate index {next_index}."
            )
        x_next = X_pending[next_index : next_index + 1]
        y_next = np.asarray(forrester_function(x_next[:, 0], task)).reshape(-1)
        X_obs = np.vstack((X_obs, x_next))
        y_obs = np.concatenate((y_obs, y_next))
        history.append(float(normalized_regret_curve(y_obs, f_min, f_max)[-1]))

    return np.asarray(history, dtype=np.float64)


def run_experiment(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        seed=args.seed,
        runs=args.runs,
        initial_points=args.initial_points,
        bo_steps=args.steps,
        candidate_count=args.candidates,
        pfn_model=args.pfn_model,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    factories = {}
    if "malibo" in args.methods:
        LOGGER.info("Building MALIBO meta-training data...")
        search_space, meta_data = build_malibo_meta_data(
            n_tasks=config.meta_tasks,
            points_per_task=config.meta_points_per_task,
            seed=config.meta_seed,
        )
        malibo_factory = MALIBOFactory(
            search_space=search_space,
            config=config,
            repo_path=args.malibo_repo,
        )
        LOGGER.info("Meta-training official MALIBO model...")
        malibo_factory.meta_train(meta_data)
        factories["malibo"] = malibo_factory

    if "pfn" in args.methods:
        LOGGER.info("Loading PFNs4BO pretrained model (%s)...", config.pfn_model)
        factories["pfn"] = PFNFactory(
            model_name=config.pfn_model,
            device=config.pfn_device,
        )

    all_curves: dict[str, list[np.ndarray]] = {name: [] for name in args.methods}
    for run_index in range(config.runs):
        run_seed = config.seed + run_index
        task = sample_task(seed=10_000_000 + run_seed)
        rng = np.random.default_rng(run_seed)
        initial_x = rng.uniform(0.0, 1.0, size=(config.initial_points, 1))
        initial_y = np.asarray(forrester_function(initial_x[:, 0], task)).reshape(-1)
        candidate_seeds = [run_seed * 1_000_003 + step for step in range(config.bo_steps)]

        for method_name in args.methods:
            LOGGER.info("Run %d/%d - %s", run_index + 1, config.runs, method_name)
            curve = _evaluate_one_run(
                method_name=method_name,
                method_factory=factories[method_name],
                task=task,
                initial_x=initial_x,
                initial_y=initial_y,
                candidate_seeds=candidate_seeds,
                config=config,
                run_seed=run_seed,
            )
            all_curves[method_name].append(curve)

    summary_rows = []
    for method_name, curves_list in all_curves.items():
        matrix = np.vstack(curves_list)
        np.savez_compressed(output_dir / f"{method_name}_curves.npz", regrets=matrix)
        summary = summarize_curves(matrix, confidence=config.cvar_confidence)
        summary_rows.append({"method": method_name, **summary})

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {**config.to_dict(), "methods": args.methods, "assumptions": {
                "forrester_task_family": "official MALIBO Forrester parameterization",
                "initial_points": config.initial_points,
                "bo_steps": config.bo_steps,
                "objective_noise_std": 0.0,
                "candidate_pool_size": config.candidate_count,
            }},
            handle,
            indent=2,
        )

    LOGGER.info("Finished. Results saved to %s", output_dir.resolve())
    for row in summary_rows:
        LOGGER.info(
            "%s: mean=%.6f, SE=%.6f, CVaR95=%.6f",
            row["method"], row["mean_final_regret"],
            row["standard_error"], row["cvar_upper_tail"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare official MALIBO and PFNs4BO on the Forrester task family."
    )
    parser.add_argument("--methods", nargs="+", choices=["malibo", "pfn"],
                        default=["malibo", "pfn"])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--initial-points", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pfn-model", choices=["hebo_plus", "bnn"], default="hebo_plus")
    parser.add_argument(
        "--malibo-repo",
        default="third_party/meta-learning-likelihood-free-bayesian-optimization",
    )
    parser.add_argument("--output", default="results/forrester")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args()
    if args.runs < 1 or args.steps < 1 or args.initial_points < 1 or args.candidates < 2:
        raise SystemExit("runs, steps, initial-points must be >=1; candidates must be >=2")
    run_experiment(args)


if __name__ == "__main__":
    main()
