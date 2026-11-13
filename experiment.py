"""Run MALIBO and PFN on the paper's Forrester task family."""
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
from models import MALIBOAgent, PFNAgent
from models.numerics import require_finite, seed_everything


def candidate_pool(rng: np.random.Generator, count: int,
                   observed: np.ndarray) -> np.ndarray:
    candidates = rng.uniform(0.0, 1.0, (count, observed.shape[1]))
    distances = np.linalg.norm(candidates[:, None] - observed[None], axis=2)
    candidates = candidates[np.min(distances, axis=1) > 1e-9]
    if not len(candidates):
        raise RuntimeError("candidate pool is empty")
    return candidates


def run_bo(agent, task, steps: int, initial_points: int,
           candidates: int, seed: int) -> np.ndarray:
    """`steps` counts ALL objective calls, including the initial design."""
    if not 1 <= initial_points <= steps:
        raise ValueError("initial_points must be within total BO evaluation budget")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    agent.rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (initial_points, 1))
    y = np.asarray(task.objective(X), dtype=np.float64).reshape(-1)
    curve = list(calculate_normalized_regret(np.minimum.accumulate(y), task.f_min, task.f_max))
    for _ in range(steps - initial_points):
        pool = candidate_pool(rng, candidates, X)
        x_next = np.atleast_2d(agent.suggest(X, y, pool))
        y_next = np.asarray(task.objective(x_next), dtype=np.float64).reshape(-1)
        X, y = np.vstack((X, x_next)), np.concatenate((y, y_next))
        curve.append(calculate_normalized_regret(y.min(), task.f_min, task.f_max))
    require_finite(curve, "regret curve")
    agent.bo_trace = {"X": X.tolist(), "y": y.tolist(), "objective_evaluations": len(y)}
    return np.asarray(curve)


def make_agents(args, bounds: np.ndarray) -> dict:
    shared = dict(
        bounds=bounds, seed=args.seed, meta_epochs=args.epochs,
        task_batch_size=args.task_batch_size, device=args.device,
    )
    agents = {
        "MALIBO": MALIBOAgent(**shared),
        "PFN": PFNAgent(
            bounds=bounds, seed=args.seed, meta_epochs=args.epochs,
            device=args.device, task_batch_size=args.task_batch_size,
        ),
    }
    return {name: agents[name] for name in args.methods}


def run_experiment(args) -> dict[str, np.ndarray]:
    seed_everything(args.seed)
    benchmark = ForresterBenchmark(args.tasks, args.seed)
    source_tasks = benchmark.get_meta_train_tasks(
        args.tasks, args.samples_per_task
    )
    agents = make_agents(args, benchmark.bounds)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'meta_dataset.npz',
                        X=np.stack([t.X for t in source_tasks]),
                        y=np.stack([t.y for t in source_tasks]))
    bce_rows, objective_rows, task_rows = [], [], []
    for name, agent in agents.items():
        print(f"[train] {name}", flush=True)
        seed_everything(args.seed)
        agent.meta_train(source_tasks)
        with (output / f"{name}_training.json").open('w', encoding='utf-8') as stream:
            json.dump(agent.training_diagnostics, stream, indent=2)
        torch.save((agent.network if name == 'PFN' else agent.model).state_dict(), output / f'{name}_checkpoint.pt')
        if name == 'MALIBO':
            for split in ('train', 'validation'):
                m = agent.classifier_metrics[split]
                bce_rows.append({'method': name, 'split': split, **{k:v for k,v in m.items()
                                if k not in ('task_losses', 'regularization', 'objective_with_regularization')}})
                objective_rows.append({'method': name, 'split': split,
                                       'regularization': m['regularization'],
                                       'objective_with_regularization': m['objective_with_regularization']})
                task_rows.extend({'method': name, 'split': split, **r} for r in m['task_losses'])
    if bce_rows:
        _write_csv(output / 'classifier_bce.csv', bce_rows)
        _write_csv(output / 'classifier_objective.csv', objective_rows)
        _write_csv(output / 'classifier_task_losses.csv', task_rows)

    curves = {name: [] for name in agents}
    traces = []
    for run in range(args.runs):
        target = benchmark.get_target_task(evaluation_seed=run)
        for name, template in agents.items():
            agent = copy.deepcopy(template)
            curves[name].append(run_bo(
                agent, target, args.steps,
                args.initial_points, args.candidates, args.seed + run,
            ))
            traces.append({'method': name, 'run': run, 'f_min': target.f_min,
                           'f_max': target.f_max, **agent.bo_trace})
        print(f"[run {run + 1:03d}/{args.runs}]", flush=True)
    (output/'bo_traces.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    return {name: np.vstack(values) for name, values in curves.items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_results(curves: dict[str, np.ndarray], args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, values in curves.items():
        final = values[:, -1]
        std = float(final.std(ddof=1)) if len(final) > 1 else 0.0
        rows.append({
            "method": method,
            "average_normalized_regret": float(final.mean()),
            "standard_error": std / np.sqrt(len(final)),
            "cvar_normalized_regret": calculate_cvar(final, args.cvar_alpha),
        })
    _write_csv(output / "bo_regret.csv", rows)
    np.savez_compressed(output / "curves.npz", **curves)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for method, values in curves.items():
        mean = values.mean(axis=0)
        se = values.std(axis=0) / np.sqrt(len(values))
        x = np.arange(1, len(mean)+1)
        plt.plot(x, mean, label=method)
        plt.fill_between(x, mean - se, mean + se, alpha=0.15)
    plt.xlabel("Total objective evaluations (including initial design)")
    plt.ylabel("Normalized regret (mean +/- SE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "convergence.png", dpi=180)
    plt.close()

    print("\nMethod        Average regret +/- SE    CVaR", flush=True)
    for row in rows:
        print(
            f"{row['method']:<14}{row['average_normalized_regret']:.6f} +/- "
            f"{row['standard_error']:.6f}    {row['cvar_normalized_regret']:.6f}", flush=True,
        )
    print(f"Saved to {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--samples-per-task", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--task-batch-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--initial-points", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--cvar-alpha", type=float, choices=(0.8,), default=0.8)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--methods", nargs="+", choices=("MALIBO", "PFN"),
        default=("MALIBO", "PFN"),
    )
    parser.add_argument("--output", default="results/author_reply")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.quick:
        args.runs, args.tasks, args.samples_per_task = 4, 8, 24
        args.epochs, args.steps, args.candidates = 8, 8, 128
        if args.output == "results/author_reply":
            args.output = "results/author_reply_smoke"
    counts = (
        args.runs, args.tasks, args.samples_per_task, args.epochs,
        args.task_batch_size, args.steps, args.initial_points,
        args.candidates, args.threads,
    )
    if min(counts) < 1:
        parser.error("all count arguments must be positive")
    if args.initial_points > args.steps or args.samples_per_task < 2:
        parser.error("initial points cannot exceed total evaluations; samples per task must be >=2")
    if not 0 <= args.cvar_alpha < 1:
        parser.error("alpha values must be in [0, 1)")
    torch.set_num_threads(args.threads)
    save_results(run_experiment(args), args)


if __name__ == "__main__":
    main()
