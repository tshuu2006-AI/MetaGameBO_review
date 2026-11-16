"""Forrester ablation with the authors' released code (paper Table 1 and "Ablation Study").

Each run does what the authors' ``run_forrester.py`` does up to the end of
meta-training (200 tasks x 32 points, 100 epochs, batch 10, CPU), for one flag
combination, and then re-evaluates the trained classifier on the authors' own
held-out split:

* with the sample weights their trainer uses (their "Val Loss" / "Val CVaR"), and
* with the unmodified Eq. (3) (LFBO) weights on the same rows.

Runs:
  script_standard   no flags                      (the script's MALIBO baseline)
  script_dro        --use_dro                     (task level only)
  script_kdpp       --use_kdpp                    (sample level only)
  script_full       --use_dro --use_kdpp          (MetaGameBO)
  true_standard     no flags, k-DPP really off    (constructor ``use_kdpp=False``)
  true_dro          --use_dro, k-DPP really off

Why the last two: ``MetaGameBO.__init__`` has ``use_kdpp=True, alpha=0.8`` as
defaults and ``run_forrester.py`` only overrides them when ``--use_kdpp`` is
given, so without that flag the k-DPP weight adjustment (``_apply_kdpp_weights``)
still runs, with k = 0.8 n instead of 0.4 n.

    python -m experiments.author_forrester_ablation                # seeds 44 45
    python -m experiments.author_forrester_ablation --seeds 44 --workers 6

Outputs: ``results/forrester_author_ablation.md``; one JSON per run in
``results/raw/forrester_author``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["script_standard", "script_dro", "script_kdpp", "script_full", "true_standard", "true_dro"]
# paper Table 1 (MALIBO, MetaGameBO) and the ablation text (reductions relative to MALIBO)
PAPER = {"script_standard": (0.4277, 0.8195), "script_dro": (0.4277 * (1 - 0.217), 0.8195 * (1 - 0.281)),
         "script_kdpp": (0.4277 * (1 - 0.718), 0.8195 * (1 - 0.705)), "script_full": (0.0183, 0.0934)}
LABEL = {"script_standard": "no flags (MALIBO row)", "script_dro": "`--use_dro` (task level only)",
         "script_kdpp": "`--use_kdpp` (sample level only)", "script_full": "`--use_dro --use_kdpp` (MetaGameBO)",
         "true_standard": "no flags, k-DPP really off", "true_dro": "`--use_dro`, k-DPP really off"}


def job(mode: str, seed: int, out: str) -> None:
    """Worker (own process, started with this file as the script so that ``benchmarks`` is the authors')."""
    import contextlib, io, json, tempfile, time
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "models" / "metagamebo_original" / "author"))
    os.chdir(tempfile.mkdtemp())  # the authors' trainer writes results/<name>.txt in the cwd
    from benchmarks.forrester import Forrester
    import MetaGameBO as M
    import meta_learning.meta_classifier as MC

    use_dro = mode in ("script_dro", "script_full", "true_dro")
    kdpp_flag = mode in ("script_kdpp", "script_full")
    # ---- as in run_forrester.py (device cpu; its `MALIBO(...)` is a NameError, the class is MetaGameBO) ----
    torch.manual_seed(seed)
    classifier_config = {"num_layers": 5, "num_features": 50, "num_hidden_units": 64, "device": "cpu", "seed": seed}
    train_config = {"num_epochs": 100, "batch_size": 10, "use_dro": use_dro, "alpha": 0.8, "seed": seed}
    if kdpp_flag:
        classifier_config.update(use_kdpp=True, alpha=0.4, kdpp_lr=0.1)
        train_config.update(use_kdpp=True, kdpp_alpha=0.4, kdpp_lr=0.1)
    if mode.startswith("true_"):
        classifier_config["use_kdpp"] = False

    rec = {"w_lfbo": [], "good0": [], "s1": []}
    orig_rw = M.MetaGameBO._reweight_samples

    def rw(X, y, gamma, weight_type="ei"):  # record the unmodified Eq. (3) weights of every task
        x, z, w = orig_rw(X, y, gamma, weight_type)
        yy = np.asarray(y, float).reshape(-1)
        good = yy < np.quantile(np.unique(yy), q=gamma)
        rec["w_lfbo"].append(np.asarray(w, float).reshape(-1).copy())
        rec["good0"].append(good)
        rec["s1"].append(int(good.sum()))
        return x, z, w

    M.MetaGameBO._reweight_samples = staticmethod(rw)
    orig_train = MC.train

    def train(*a, **kw):  # record the pooled training data and the authors' random train/validation split
        rec["data"] = [np.asarray(v) for v in kw["data"]]
        orig_split = torch.utils.data.random_split

        def split(ds, lengths, *aa, **kk):
            parts = orig_split(ds, lengths, *aa, **kk)
            rec["train_idx"], rec["val_idx"] = np.array(parts[0].indices), np.array(parts[1].indices)
            return parts

        torch.utils.data.random_split = split
        try:
            return orig_train(*a, **kw)
        finally:
            torch.utils.data.random_split = orig_split

    MC.train = train
    benchmark = Forrester(n_data_per_task=[32] * 200, seed=seed)
    optimizer = M.MetaGameBO(benchmark.search_space, **classifier_config)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        optimizer.meta_fit(benchmark.get_meta_data(), **train_config)
    seconds = time.time() - t0

    reported = {}
    for f in os.listdir("results"):
        for line in open(os.path.join("results", f)):
            for k in ("Final Train Loss", "Final Train CVaR", "Final Val Loss", "Final Val CVaR"):
                if line.startswith(k + ":"):
                    reported[k] = float(line.split(":")[1])

    X, tid, lab, w_used = rec["data"]
    lab, w_used = lab.reshape(-1).astype(float), w_used.reshape(-1).astype(float)
    w_lfbo = np.concatenate(rec["w_lfbo"])
    model = optimizer.classifier.meta_model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float64), torch.tensor(tid, dtype=torch.long))[1].reshape(-1).numpy()
    bce = np.logaddexp(0, logit) - lab * logit
    p = 1 / (1 + np.exp(-logit))
    # row kinds per task: label-1 copies of good points (1), label-0 copies of good points (2), bad points (0)
    kind = np.concatenate([np.concatenate([np.full(s, 1), np.where(g, 2, 0)]) for s, g in zip(rec["s1"], rec["good0"])])

    def cvar(v, alpha=0.8):
        v = np.sort(np.asarray(v))[::-1]
        return float(v[:max(1, int(len(v) * (1 - alpha)))].mean())

    def task_losses(idx, w):
        return np.array([np.mean(w[idx][tid[idx] == t] * bce[idx][tid[idx] == t]) for t in np.unique(tid[idx])])

    res = dict(mode=mode, seed=seed, seconds=seconds, reported=reported,
               use_kdpp_effective=bool(optimizer.use_kdpp), kdpp_k_fraction=float(optimizer.alpha),
               weights_changed=bool(not np.allclose(w_lfbo, w_used)),
               class1_weight_share_lfbo=float(w_lfbo[lab == 1].sum() / w_lfbo.sum()),
               class1_weight_share_used=float(w_used[lab == 1].sum() / w_used.sum()))
    for name, idx in (("val", rec["val_idx"]), ("train", rec["train_idx"])):
        Lu, Ll = task_losses(idx, w_used), task_losses(idx, w_lfbo)
        res[name] = dict(n_rows=int(len(idx)), n_tasks=int(len(Lu)),
                         used_pooled_mean=float(np.mean(w_used[idx] * bce[idx])),
                         used_task_mean=float(Lu.mean()), used_cvar=cvar(Lu),
                         lfbo_task_mean=float(Ll.mean()), lfbo_cvar=cvar(Ll),
                         p_good_pos_copy=float(p[idx][kind[idx] == 1].mean()),
                         p_good_neg_copy=float(p[idx][kind[idx] == 2].mean()),
                         p_bad=float(p[idx][kind[idx] == 0].mean()),
                         acc_good=float((p[idx][kind[idx] == 1] > 0.5).mean()),
                         acc_bad=float((p[idx][kind[idx] == 0] < 0.5).mean()))
    Path(out).write_text(json.dumps(res, indent=1))


def table(results) -> str:
    import numpy as np
    seeds = sorted({r["seed"] for r in results})
    rows = ["## Forrester: ablation with the authors' code (paper Table 1 and Ablation Study)", "",
            f"Authors' `run_forrester.py` logic, meta-training only (200 tasks x 32 points, 100 epochs, batch 10, CPU); "
            f"seeds {seeds}; mean over seeds. All numbers are on the authors' own held-out split (random 20% of the rows).",
            "", "| Run (flags of run_forrester.py) | seeds | k-DPP reweighting applied | class-1 weight share | "
            "Val Loss / Val CVaR reported by the authors' trainer | paper | same rows, unmodified Eq. (3) weights: "
            "Avg / CVaR | mean P(good) on good / bad points |", "|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        rs = [r for r in results if r["mode"] == m]
        if not rs:
            continue
        mean = lambda f: float(np.mean([f(r) for r in rs]))
        kd = f"yes, k = {rs[0]['kdpp_k_fraction']:g} n" if rs[0]["weights_changed"] else "no"
        paper = "{:.4f} / {:.4f}".format(*PAPER[m]) if m in PAPER else "–"
        rows.append(
            f"| {LABEL[m]} | {len(rs)} | {kd} | {mean(lambda r: r['class1_weight_share_used']):.3f} | "
            f"{mean(lambda r: r['reported']['Final Val Loss']):.4f} / {mean(lambda r: r['reported']['Final Val CVaR']):.4f} | "
            f"{paper} | {mean(lambda r: r['val']['lfbo_task_mean']):.3f} / {mean(lambda r: r['val']['lfbo_cvar']):.3f} | "
            f"{mean(lambda r: r['val']['p_good_pos_copy']):.3f} / {mean(lambda r: r['val']['p_bad']):.3f} |")
    rows += ["", "Notes:",
             "- Paper column: Table 1 for MALIBO and MetaGameBO; for the two ablation rows, the values implied by the "
             "ablation text (average loss −21.7% / CVaR −28.1% for task level only, −71.8% / −70.5% for sample level only, "
             "relative to MALIBO).",
             "- Class-1 weight share: fraction of the total sample weight carried by the label-1 rows (good points). "
             "Eq. (3) gives 0.5; `_apply_kdpp_weights` changes it before the train/validation split, so the authors' "
             "validation loss is computed with the changed weights.",
             "- Without `--use_kdpp` the script still applies `_apply_kdpp_weights` with k = 0.8 n "
             "(constructor defaults `use_kdpp=True, alpha=0.8`); the two *really off* rows pass `use_kdpp=False`.",
             "- The authors' trainer computes its validation numbers in training mode (embedding noise on) and its "
             "\"Val Loss\" adds 0.1 x the embedding regulariser; the re-evaluation uses eval mode, hence small differences.",
             "- P(good): mean predicted probability of the good class on the held-out label-1 rows / on the rows of bad points."]
    return "\n".join(rows) + "\n"


def main():
    import argparse, json, subprocess, tempfile
    from concurrent.futures import ThreadPoolExecutor
    from experiments.common import RESULTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[44, 45], help="44 is the default of run_forrester.py")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    raw = RESULTS / "raw" / "forrester_author"
    raw.mkdir(parents=True, exist_ok=True)
    todo = [(m, s, raw / f"{m}__s{s}.json") for m in a.modes for s in a.seeds]
    todo = [t for t in todo if a.rerun or not t[2].exists()]
    print(f"[forrester_author] {len(todo)} runs to do (about 4-7 minutes each)", flush=True)

    def run(t):
        m, s, out = t
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--job", m, str(s), str(out)],
                           cwd=tempfile.mkdtemp(), capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))
        print(f"[forrester_author] {m} seed {s}: {'ok' if p.returncode == 0 else 'ERROR ' + p.stderr[-400:]}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    results = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
    results = [r for r in results if r["mode"] in a.modes and r["seed"] in a.seeds]
    md = table(results)
    (RESULTS / "forrester_author_ablation.md").write_text(md)
    print(md)
    print("wrote", RESULTS / "forrester_author_ablation.md")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--job":
        job(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        main()
