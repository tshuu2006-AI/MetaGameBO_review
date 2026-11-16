"""Run the benchmarks on Kaggle (or any machine): install, download, analyze, save.

Steps (each skipped when already done):
  1. pip-install the packages Kaggle lacks (author code: parameterspace, blackboxopt).
  2. Fetch the code that is not in this repository: the authors' MetaGameBO
     (github.com/Allen0497/MetaGameBO -> models/metagamebo_original/author) and
     PFNs4BO with its pre-trained weights (github.com/automl/PFNs4BO -> models/pfn/PFNs4BO).
  3. Download the data of the selected benchmarks into ``--data-dir`` (outside
     the Kaggle output) and link it as ``benchmarks/data``; build the caches.
  4. Run ``experiments.analyze_<benchmark>`` for each selected benchmark, within
     an optional time budget (the run is stopped, then the table/figure are
     rebuilt from the finished jobs).
  5. Results go straight to ``--output`` (default ``/kaggle/working/results``):
     every finished job is a JSON file there at once, and every running job
     keeps a checkpoint (``*.ckpt.pkl``: state after meta-training, then every
     ``--ckpt-minutes`` of BO), so a stopped or crashed run loses at most a few
     minutes per running job. Run the same command again (``--resume`` for a new
     session) and the jobs continue from their checkpoints.

Kaggle notebook (Settings: Internet on; GPU optional, used by PFN):

    !rm -rf /tmp/repo && git clone -b kaggle https://github.com/tshuu2006-AI/MetaGameBO_review.git /tmp/repo
    !cd /tmp/repo && python kaggle_run.py --benchmarks nasbench201 hpob --hours 11

Resume in a later session: add the previous notebook's output as an input and pass
``--resume /kaggle/input/<name>/results``; finished jobs are not run again and
unfinished ones continue from their checkpoints.
Anything after ``--`` is passed to every analyze script, e.g. ``-- --seeds 20``.
``--noisy`` runs hpobench/nasbench201 with noisy observations (results ``<benchmark>_noisy``).
``--discrete`` runs hpob on the tabulated test evaluations, as the official code does
(results ``hpob_discrete``); with ``-- --budget 105`` this is 5 initial points + 100 steps.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ON_KAGGLE = Path("/kaggle/working").exists()
BENCHMARKS = ["forrester", "hpobench", "nasbench201", "hpob"]

HF = "https://huggingface.co/datasets/synetune/blackbox-repository/resolve/main"
HPOB = "https://media.githubusercontent.com/media/sebastianpinedaar/hpo-data/refs/heads/main"
SYNETUNE_FILES = ["configspace.json", "fidelities_values.npy", "fidelityspace.json",
                  "hyperparameters.parquet", "metadata.json", "objectives_evaluations.npy"]
AUTHOR_REPO = "https://github.com/Allen0497/MetaGameBO.git"
PFN_REPO = "https://github.com/automl/PFNs4BO.git"
PFN_WEIGHTS = ["model_hebo_morebudget_9_unused_features_3.pt"]  # the model used by models/pfn
# the author pinned blackboxopt==5.0.4, which needs Python < 3.12 (Kaggle has 3.13);
# the author code also runs with blackboxopt 5.3.8 / parameterspace 0.10.2
PIP = ["parameterspace>=0.10.0", "blackboxopt>=5.0.4", "pyarrow"]


def log(msg: str) -> None:
    print(f"[kaggle_run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (skipped if ``dest`` exists and is not empty)."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "metagamebo-repro"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dest)


# ---------------------------------------------------------------- setup steps
def install_packages() -> None:
    missing = []
    for spec in PIP:
        mod = spec.split("=")[0].split(">")[0].split("<")[0]
        try:
            __import__(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        log(f"pip install {' '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def fetch_code() -> None:
    author = ROOT / "models" / "metagamebo_original" / "author"
    if not (author / "MetaGameBO.py").exists():
        log("cloning the authors' MetaGameBO code")
        subprocess.run(["git", "clone", "--depth", "1", AUTHOR_REPO, str(author)], check=True)
    pfn = ROOT / "models" / "pfn" / "PFNs4BO"
    if not (pfn / "pfns4bo" / "__init__.py").exists():
        log("cloning PFNs4BO")
        subprocess.run(["git", "clone", "--depth", "1", PFN_REPO, str(pfn)], check=True)
    models = pfn / "pfns4bo" / "final_models"
    for name in PFN_WEIGHTS:
        pt, gz = models / name, models / (name + ".gz")
        if pt.exists():
            continue
        if not gz.exists() or gz.stat().st_size < 1_000_000:  # missing or a git-lfs pointer
            download(f"https://github.com/automl/PFNs4BO/raw/main/pfns4bo/final_models/{name}.gz", gz)
        log(f"unzipping {gz.name}")
        with gzip.open(gz, "rb") as fi, open(pt, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def link_data(data_dir: Path) -> None:
    """Make ``benchmarks/data`` point to ``data_dir`` (keeps the data out of the Kaggle output)."""
    link = ROOT / "benchmarks" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or not link.exists():
        if link.is_symlink() and link.resolve() == data_dir.resolve():
            return
        if link.is_symlink():
            link.unlink()
        link.symlink_to(data_dir, target_is_directory=True)
    elif link.resolve() != data_dir.resolve():
        log(f"benchmarks/data already exists as a folder; using it instead of {data_dir}")


def fetch_data(benchmarks) -> None:
    data = ROOT / "benchmarks" / "data"
    for bench, folder in (("hpobench", "fcnet"), ("nasbench201", "nasbench201")):
        if bench in benchmarks:
            for f in SYNETUNE_FILES:
                download(f"{HF}/{folder}/{f}", data / folder / f)
    if "hpob" in benchmarks:
        for name in ("hpob-data", "saved-surrogates"):
            if not (data / name).is_dir():
                zp = data / f"{name}.zip"
                download(f"{HPOB}/{name}.zip", zp)
                log(f"unzipping {zp.name}")
                with zipfile.ZipFile(zp) as z:
                    z.extractall(data)
                zp.unlink()
    # compact caches (built once, small)
    code = []
    if "hpobench" in benchmarks:
        code.append("from benchmarks import hpobench; hpobench.load(); hpobench.load_seeds()")
    if "nasbench201" in benchmarks:
        code.append("from benchmarks import nasbench201; nasbench201.load(); nasbench201.load_seeds()")
    if "hpob" in benchmarks:
        code.append("from benchmarks import hpob\n"
                    "if not hpob.CACHE.exists() or not any(hpob.CACHE.glob('*.npz')): hpob.prepare()")
    for c in code:
        log(f"building cache: {c.splitlines()[0]}")
        subprocess.run([sys.executable, "-c", c], cwd=ROOT, check=True)


# ---------------------------------------------------------------- running
def run_module(args, timeout):
    """Run ``python -m <args>`` in its own process group; stop the whole group on timeout."""
    p = subprocess.Popen([sys.executable, "-m", *args], cwd=ROOT, start_new_session=True)
    try:
        p.wait(timeout=timeout)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        log("time budget reached: stopping the run (finished jobs are kept)")
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        return False


def report_forrester() -> None:
    """Rebuild the Forrester table and figure from the cached jobs."""
    code = ("import json\nfrom experiments.analyze_forrester import RESULTS, METHODS, table, plot\n"
            "raw = RESULTS / 'raw' / 'forrester'\n"
            "rs = [json.loads(p.read_text()) for p in sorted(raw.glob('*.json'))]\n"
            "rs = [r for r in rs if 'error' not in r]\n"
            "(RESULTS / 'forrester_benchmark.md').write_text(table(rs, METHODS))\n"
            "plot(rs, RESULTS / 'forrester_plot.png')\n")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT)


def save_results(output: Path) -> None:
    """Results are written into ``output`` directly; only report what is there."""
    raw = output / "raw"
    n = len(list(raw.glob("*/*.json"))) if raw.exists() else 0
    c = len(list(raw.glob("*/*.ckpt.pkl"))) if raw.exists() else 0
    log(f"{output}: {n} finished jobs, {c} checkpoints of unfinished jobs")


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    ap.add_argument("--hours", type=float, default=None,
                    help="total time budget (Kaggle sessions stop after 12 h); default: no limit")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--hpob-groups", nargs="+", default=None, help="subset of HPO-B search spaces")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/tmp/metagamebo_data") if ON_KAGGLE else ROOT / "benchmarks" / "data")
    ap.add_argument("--output", type=Path,
                    default=Path("/kaggle/working/results") if ON_KAGGLE else ROOT / "results")
    ap.add_argument("--resume", type=Path, default=None,
                    help="a previous results/ folder (e.g. a Kaggle input) whose raw/ jobs are reused")
    ap.add_argument("--ckpt-minutes", type=float, default=5, help="minutes between BO checkpoints of a job")
    ap.add_argument("--discrete", action="store_true",
                    help="hpob: tabulated test evaluations instead of surrogates (results under hpob_discrete)")
    ap.add_argument("--noisy", action="store_true",
                    help="hpobench/nasbench201: noisy protocol (results under <benchmark>_noisy)")
    ap.add_argument("--setup-only", action="store_true", help="install and download, do not run")
    a = ap.parse_args(argv)
    t_end = time.time() + a.hours * 3600 if a.hours else None

    a.output.mkdir(parents=True, exist_ok=True)
    os.environ["METAGAMEBO_RESULTS"] = str(a.output.resolve())  # inherited by the analyze runs
    os.environ["METAGAMEBO_CKPT_SECONDS"] = str(60 * a.ckpt_minutes)
    install_packages()
    fetch_code()
    link_data(a.data_dir)
    fetch_data(a.benchmarks)
    if a.resume:
        src = a.resume / "raw" if (a.resume / "raw").exists() else a.resume
        if src.resolve() != (a.output / "raw").resolve():
            shutil.copytree(src, a.output / "raw", dirs_exist_ok=True)
        log(f"resumed cached jobs from {src}")
    if a.setup_only:
        log("setup done")
        return

    for bench in a.benchmarks:
        left = None if t_end is None else t_end - time.time() - 600  # keep 10 min to save
        if left is not None and left <= 0:
            log(f"no time left for {bench}; skipped")
            continue
        args = [f"experiments.analyze_{bench}", "--workers", str(a.workers), *extra]
        if bench == "hpob" and a.hpob_groups:
            args += ["--groups", *a.hpob_groups]
        if a.noisy and bench in ("hpobench", "nasbench201"):
            args.append("--noisy")
        if a.discrete and bench == "hpob":
            args.append("--discrete")
        log(f"running {' '.join(args)}" + (f" (budget {left / 3600:.1f} h)" if left else ""))
        ok = run_module(args, left)
        if not ok:  # stopped or failed: rebuild table/figure from the finished jobs
            if bench == "forrester":
                report_forrester()
            else:
                run_module([*args, "--report-only"], None)
        save_results(a.output)
    log("all done")


if __name__ == "__main__":
    main()
