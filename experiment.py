"""Leave-one-task-out experiment runner and result reporting."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks import ForresterBenchmark
from metrics import calculate_cvar, calculate_normalized_regret
from models import MALIBOAgent, MetaGameBOAgent, PFNAgent


def candidate_pool(rng: np.random.Generator, count: int,
                   observed: np.ndarray) -> np.ndarray:
    candidates = rng.uniform(0.0, 1.0, (count, observed.shape[1]))
    if len(observed):
        distances = np.linalg.norm(
            candidates[:, None] - observed[None], axis=2
        )
        candidates = candidates[np.min(distances, axis=1) > 1e-9]
    return candidates


def run_bo(agent, task, steps: int, initial_points: int, candidates: int,
           seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (initial_points, 1))
    y = np.asarray(task.objective(X), dtype=np.float64)
    curve = [calculate_normalized_regret(np.min(y), task.f_min, task.f_max)]
    for _ in range(steps):
        pool = candidate_pool(rng, candidates, X)
        x_next = np.atleast_2d(agent.suggest(X, y, pool))
        y_next = np.asarray(task.objective(x_next)).reshape(-1)
        X, y = np.vstack((X, x_next)), np.concatenate((y, y_next))
        curve.append(calculate_normalized_regret(
            np.min(y), task.f_min, task.f_max
        ))
    return np.asarray(curve)


def make_agents(bounds: np.ndarray, seed: int, epochs: int, device: str):
    shared = dict(bounds=bounds, seed=seed, meta_epochs=epochs, device=device)
    return {
        "MALIBO": MALIBOAgent(**shared),
        "PNF-PFNs4BO": PFNAgent(**shared),
        "MetaGameBO": MetaGameBOAgent(**shared),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, np.ndarray]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    benchmark = ForresterBenchmark(args.tasks, args.seed)
    offline = benchmark.get_meta_train_tasks(
        args.tasks, args.samples_per_task
    )
    method_names = ("MALIBO", "PNF-PFNs4BO", "MetaGameBO")
    all_curves: dict[str, list[np.ndarray]] = {name: [] for name in method_names}
    # Huấn luyện mỗi fold một lần; target observations không đi vào meta-training.
    folds = min(args.folds, args.tasks)
    trained: dict[int, dict] = {}
    for fold in range(folds):
        source_tasks = [task for task in offline if task.task_id != fold]
        agents = make_agents(
            benchmark.bounds, args.seed + fold, args.epochs, args.device
        )
        print(f"[fold {fold + 1}/{folds}] meta-training on {len(source_tasks)} tasks")
        for name, agent in agents.items():
            print(f"  - {name}")
            agent.meta_train(source_tasks)
        trained[fold] = agents

    for run in range(args.runs):
        fold = run % folds
        for method, template in trained[fold].items():
            # Mỗi method nhận một objective stream khởi tạo giống hệt nhau.
            task = benchmark.get_target_task(fold, evaluation_seed=run)
            curve = run_bo(
                copy.deepcopy(template), task, args.steps,
                args.initial_points, args.candidates, seed=run,
            )
            all_curves[method].append(curve)
        print(f"[run {run + 1:03d}/{args.runs}] target task={fold}")
    return {name: np.vstack(curves) for name, curves in all_curves.items()}


def save_results(curves: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, values in curves.items():
        final = values[:, -1]
        rows.append({
            "method": method,
            "average_normalized_regret": float(final.mean()),
            "std": float(final.std(ddof=1)) if len(final) > 1 else 0.0,
            "cvar": calculate_cvar(final, args.cvar_alpha),
        })
        np.save(output / f"{method}_curves.npy", values)
    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for method, values in curves.items():
            mean, std = values.mean(0), values.std(0)
            x = np.arange(len(mean))
            plt.plot(x, mean, label=method)
            plt.fill_between(x, mean - std, mean + std, alpha=.15)
        plt.xlabel("BO step (step 0 is the initial design)")
        plt.ylabel("Normalized regret")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "convergence.png", dpi=180)
        plt.close()
    except ImportError:
        print("matplotlib not installed; skipped convergence.png")
    print("\nMethod                 Avg. normalized regret +/- std    CVaR")
    print("-" * 69)
    for row in rows:
        print(
            f"{row['method']:<22} {row['average_normalized_regret']:.6f} +/- "
            f"{row['std']:.6f}          {row['cvar']:.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--folds", type=int, default=12)
    parser.add_argument("--samples-per-task", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--initial-points", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--cvar-alpha", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/forrester")
    parser.add_argument("--quick", action="store_true", help="small smoke experiment")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.runs, args.tasks, args.folds = 2, 5, 2
        args.samples_per_task, args.steps = 64, 8
        args.candidates, args.epochs, args.initial_points = 128, 3, 2
    counts = (
        args.runs, args.tasks, args.folds, args.samples_per_task, args.steps,
        args.initial_points, args.candidates, args.epochs,
    )
    if min(counts) < 1:
        raise SystemExit("all count arguments must be positive")
    curves = run_experiment(args)
    save_results(curves, args)


if __name__ == "__main__":
    main()
