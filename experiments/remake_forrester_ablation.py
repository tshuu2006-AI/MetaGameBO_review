"""Forrester ablation of the re-implementation (paper "Ablation Study"), Table-1 losses only.

MALIBO and MetaGameBO (remake) with each level of the game switched on/off, for
both Eq. (3) labelling schemes of ``models.lfbo.weighting``:

* ``lfbo``:  class 0 = all points (LFBO / MALIBO / authors' code); the default everywhere else.
* ``paper``: class 0 = points with y >= threshold only (Eq. 3 as written in the paper).

All variants share the meta-training loop (2000 full-batch Adam steps), the data
(200 tasks x 32 points) and, per seed, the 80/20 split by sample. The losses of
the two schemes are not comparable with each other (different labels and floor).

    python -m experiments.remake_forrester_ablation               # seeds 0 1 2
    python -m experiments.remake_forrester_ablation --seeds 0 1 2 3 4 --workers 8

Outputs: ``results/forrester_remake_ablation.md``; JSON per run in ``results/raw/forrester_remake``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.common import RESULTS, _init_worker, mean_se

VARIANTS = {"malibo": "MALIBO (all tasks, all samples)",
            "remake_task": "+ CVaR task level only",
            "remake_sample": "+ k-DPP sample level only (Eq. 6)",
            "remake_full": "MetaGameBO remake (both levels)"}
SCHEMES = {"lfbo": "LFBO labels (class 0 = all points)", "paper": "Eq. (3) as written (class 0 = y >= threshold)"}


def run_job(job):
    path, name, seed = job
    from benchmarks.forrester import BOUNDS, make_meta_tasks
    from models import build
    from models.malibo.training import MetaTrainConfig
    from models.metagamebo_remake.model import GameConfig
    base, scheme = name.rsplit("_", 1)
    cfg = MetaTrainConfig(seed=seed, weighting=scheme)
    if base == "malibo":
        opt = build("MALIBO", BOUNDS, seed, cfg=cfg)
    else:
        game = {"remake_full": GameConfig(), "remake_task": GameConfig(sample_level=None),
                "remake_sample": GameConfig(task_level="erm")}[base]
        opt = build("MetaGameBO (remake)", BOUNDS, seed, cfg=cfg, game=game)
    t0 = time.time()
    opt.meta_train(make_meta_tasks(200, 32, seed))
    d = opt.log.__dict__
    i = int(np.argmin(d["val_avg_task_loss"]))
    res = dict(name=name, seed=seed, seconds=time.time() - t0,
               final={k: v for k, v in d["final"].items() if k != "val_task_losses"},
               best_val_iter=int(d["iteration"][i]), best_val_avg=float(d["val_avg_task_loss"][i]),
               val_cvar_at_best=float(d["val_cvar_task_loss"][i]), hard_frac=float(np.mean(d["hard_frac"])))
    Path(path).write_text(json.dumps(res))
    return name, seed, time.time() - t0


def table(results, seeds) -> str:
    rows = ["## Forrester: ablation of the re-implementation (Table-1 losses)", "",
            f"200 tasks x 32 points, 2000 full-batch steps; seeds {sorted(seeds)}; mean ± standard error. "
            "Held-out = random 20% of the pooled rows (same split for all variants of a seed).", ""]
    for scheme, title in SCHEMES.items():
        rows += [f"### {title}", "",
                 "| Variant | seeds | tasks per step | train Avg / CVaR | held-out Avg | held-out CVaR | "
                 "held-out Avg / CVaR at the best iteration | best iteration |", "|---|---|---|---|---|---|---|---|"]
        ref = None
        for base, label in VARIANTS.items():
            rs = [r for r in results if r["name"] == f"{base}_{scheme}"]
            if not rs:
                continue
            f = lambda k: mean_se(np.array([r["final"][k] for r in rs]))
            (va, vase), (vc, vcse) = f("val_avg"), f("val_cvar")
            if base == "malibo":
                ref = (va, vc)
            rel = "" if ref is None or base == "malibo" else f" ({100 * (va / ref[0] - 1):+.0f}%)"
            relc = "" if ref is None or base == "malibo" else f" ({100 * (vc / ref[1] - 1):+.0f}%)"
            rows.append(
                f"| {label} | {len(rs)} | {100 * np.mean([r['hard_frac'] for r in rs]):.0f}% | "
                f"{f('train_avg')[0]:.3f} / {f('train_cvar')[0]:.3f} | {va:.3f} ± {vase:.3f}{rel} | "
                f"{vc:.3f} ± {vcse:.3f}{relc} | {np.mean([r['best_val_avg'] for r in rs]):.3f} / "
                f"{np.mean([r['val_cvar_at_best'] for r in rs]):.3f} | {np.mean([r['best_val_iter'] for r in rs]):.0f} |")
        rows.append("")
    rows += ["Notes:",
             "- Percentages: relative to the MALIBO row of the same labelling scheme (negative = lower loss).",
             "- Paper (ablation text, relative to MALIBO): task level only −21.7% average loss / −28.1% CVaR; "
             "sample level only −71.8% / −70.5%; both −95.7% / −88.6%.",
             "- \"Best iteration\": iteration with the lowest held-out average loss (logged every 50 steps), i.e. "
             "what early stopping on the held-out loss would keep.",
             "- Train loss ln 2 = 0.693 means the classifier outputs 0.5 everywhere."]
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    raw = RESULTS / "raw" / "forrester_remake"
    raw.mkdir(parents=True, exist_ok=True)
    names = [f"{b}_{s}" for s in SCHEMES for b in VARIANTS]
    jobs = [(str(raw / f"{n}__s{s}.json"), n, s) for n in names for s in a.seeds]
    jobs = [j for j in jobs if a.rerun or not Path(j[0]).exists()]
    print(f"[forrester_remake] {len(jobs)} runs to do", flush=True)
    if jobs:
        with get_context("spawn").Pool(a.workers, initializer=_init_worker) as pool:
            for i, (n, s, dt) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f"[forrester_remake] {i}/{len(jobs)} {n} seed {s}: {dt:.0f}s", flush=True)
    results = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
    results = [r for r in results if r["seed"] in a.seeds]
    md = table(results, a.seeds)
    (RESULTS / "forrester_remake_ablation.md").write_text(md)
    print(md)
    print("wrote", RESULTS / "forrester_remake_ablation.md")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
