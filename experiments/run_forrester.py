"""CPU-only CLI for Phase 1. Run from the repository root with python -m."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.evaluate import ROOT, run_experiment, validate_config


def load_config(path) -> dict:
    """Load a YAML mapping; resolve relative output/manifest paths against the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for key in ("output_dir",):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
    manifest_path = config["benchmark"].get("manifest_path")
    if manifest_path:
        value = Path(manifest_path)
        config["benchmark"]["manifest_path"] = str(value if value.is_absolute() else ROOT / value)
    return config


def main(argv=None) -> int:
    """Apply explicit CLI overrides and run only the requested protocol; return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/forrester.yaml")
    parser.add_argument("--model", choices=["random"])
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--mode", choices=["smoke", "full"])
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--public-project-confirmed", action="store_true")
    parser.add_argument("--output-dir", help="Parent output directory; a content-hash subdirectory is created")
    parser.add_argument("--manifest", help="Existing evaluator-only task manifest")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--n-init", type=int)
    parser.add_argument("--cvar-alpha", type=float)
    parser.add_argument("--resume", action="store_true", help="Verify and skip completed runs; execute only missing runs")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        for key, value in [("model", args.model), ("output_dir", args.output_dir)]:
            if value is not None:
                config[key] = value
        for key, value in [("budget", args.budget), ("n_init", args.n_init), ("cvar_alpha", args.cvar_alpha)]:
            if value is not None:
                config["evaluation"][key] = value
        if args.seed is not None or args.seeds is not None:
            config["evaluation"]["seeds"] = [args.seed] if args.seed is not None else args.seeds
        if args.manifest:
            config["benchmark"]["manifest_path"] = str(Path(args.manifest).resolve())
        for key, value in [("mode", args.wandb_mode), ("project", args.wandb_project), ("entity", args.wandb_entity)]:
            if value is not None:
                config["tracking"][key] = value
        if args.public_project_confirmed:
            config["tracking"]["public_project_confirmed"] = True
        config = validate_config(config)
        b, e = config["benchmark"], config["evaluation"]
        if args.mode == "smoke" and (b["num_target_tasks"] > 3 or b["num_meta_tasks"] > 10
                                      or e["budget"] > 16 or len(e["seeds"]) > 3):
            raise ValueError("Smoke mode refuses heavy configurations; use configs/forrester_smoke.yaml")
        if config["protocol_label"] == "paper_candidate_UNVERIFIED" and args.mode != "full":
            raise ValueError("The unverified paper candidate requires explicit --mode full")
        if args.mode == "full" and config["protocol_label"] != "paper_candidate_UNVERIFIED":
            raise ValueError("Full mode requires configs/forrester_paper_candidate.yaml; no silent preset expansion")
        print(f"Protocol={config['protocol_label']}; algorithm={config['model']}; total budget/run={e['budget']} including n_init={e['n_init']}")
        if args.resume:
            print("Resume requested: existing completed runs will be verified before skipping.")
        outcome = run_experiment(config, resume=args.resume)
        print(json.dumps(outcome, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Experiment failed ({args.config}): {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
