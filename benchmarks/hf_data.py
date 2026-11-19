"""Benchmark data on a public Hugging Face dataset repo: upload once, download on Kaggle.

The repo mirrors ``benchmarks/data`` folder by folder:

  fcnet/            HPOBench FC-Net tables (from synetune/blackbox-repository) + compact caches
  nasbench201/      NAS-Bench-201 tables (from synetune/blackbox-repository) + compact caches
  hpob-data/        HPO-B v3 JSON files (github.com/releaunifreiburg/HPO-B, MIT)
  saved-surrogates/ HPO-B XGBoost surrogates (same source)
  hpob_cache/       compact per-search-space ``.npz`` built by ``benchmarks.hpob.prepare``

The loaders read the compact caches first (``compact.npz``, ``compact_seeds.npz``,
``hpob_cache/*.npz``), so a run only needs those plus, for HPO-B with surrogates,
``saved-surrogates``; the raw tables are kept in the repo for rebuilding the caches.

Upload (after ``hf auth login``)::

    python -m benchmarks.hf_data upload --repo Huu-ai/metagamebo-benchmarks

Download (what ``kaggle_run.py`` calls)::

    python -m benchmarks.hf_data download --benchmarks hpobench nasbench201 hpob

The repo is public, so downloads need no token; a token (only for faster downloads, and
needed to upload) is read from ``HF_TOKEN`` (environment), else the Kaggle secret
``HF_TOKEN``, else the local ``hf auth login`` cache.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

DATA = Path(__file__).resolve().parent / "data"
REPO = os.environ.get("HF_DATA_REPO", "Huu-ai/metagamebo-benchmarks")  # public
FOLDERS = ["fcnet", "nasbench201", "hpob-data", "saved-surrogates", "hpob_cache"]
CACHES = {"hpobench": ["fcnet/compact.npz", "fcnet/compact_seeds.npz"],
          "nasbench201": ["nasbench201/compact.npz", "nasbench201/compact_seeds.npz"],
          "hpob": ["hpob_cache/*.npz"]}
RAW = {"hpobench": ["fcnet/*"], "nasbench201": ["nasbench201/*"], "hpob": ["hpob-data/*"]}

CARD = """---
license: other
pretty_name: MetaGameBO reproduction benchmark data
tags: [bayesian-optimization, hyperparameter-optimization, meta-learning, benchmark]
---

# MetaGameBO reproduction: benchmark data

Mirror of the data used to reproduce MetaGameBO (AAAI-26) on HPOBench, NAS-Bench-201 and
HPO-B, laid out as the project's `benchmarks/data` folder. Nothing here is new data.

| Folder | Content | Source and licence |
|---|---|---|
| `fcnet/` | HPOBench FC-Net tables (Klein and Hutter, 2019) | [synetune/blackbox-repository](https://huggingface.co/datasets/synetune/blackbox-repository), see its card |
| `nasbench201/` | NAS-Bench-201 tables (Dong and Yang, 2020) | same |
| `hpob-data/` | HPO-B v3 meta-datasets (Pineda Arango et al., 2021) | [HPO-B](https://github.com/releaunifreiburg/HPO-B), MIT |
| `saved-surrogates/` | HPO-B XGBoost surrogates | same |
| `hpob_cache/`, `*/compact*.npz` | compact caches built by this project's loaders | derived from the above |

Load with `python -m benchmarks.hf_data download --benchmarks hpobench nasbench201 hpob`.
"""


def get_token() -> Optional[str]:
    """HF token from ``HF_TOKEN``, else the Kaggle secret ``HF_TOKEN``, else ``None`` (local login cache).

    Returns:
        The token, or ``None`` to let ``huggingface_hub`` use its own cache.
    """
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient  # only on Kaggle
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def upload(repo: str = REPO, private: bool = True, folders: Iterable[str] = FOLDERS) -> None:
    """Create the dataset repo (if needed) and upload the data folders, resumably.

    Args:
        repo: Dataset repo id ``<user>/<name>``.
        private: Create the repo as private.
        folders: Sub-folders of ``benchmarks/data`` to upload.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=get_token())
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                    commit_message="Dataset card")
    missing = [f for f in folders if not (DATA / f).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing folders in {DATA}: {missing}")
    # one commit per folder; if it stops, run again: files already on the Hub are not re-sent
    for f in folders:
        print(f"uploading {f}/", flush=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=DATA / f, path_in_repo=f,
                          commit_message=f"Add {f}/", ignore_patterns=[".cache/*", "*.zip"])


def patterns(benchmarks: Iterable[str], surrogates: bool = True, raw: bool = False) -> List[str]:
    """File patterns a set of benchmarks needs.

    Args:
        benchmarks: Benchmark names (``hpobench``, ``nasbench201``, ``hpob``; others are ignored).
        surrogates: HPO-B also needs its XGBoost surrogates (not for the discrete protocol).
        raw: Also download the raw tables (only needed to rebuild the caches).

    Returns:
        ``allow_patterns`` for ``snapshot_download``.
    """
    out = []
    for b in benchmarks:
        out += CACHES.get(b, []) + (RAW.get(b, []) if raw else [])
    if "hpob" in benchmarks and surrogates:
        out.append("saved-surrogates/*")
    return out


def download(benchmarks: Iterable[str], repo: str = REPO, surrogates: bool = True, raw: bool = False,
             dest: Path = DATA) -> Path:
    """Download the files the benchmarks need into ``dest`` (default ``benchmarks/data``).

    Args:
        benchmarks: Benchmark names.
        repo: Dataset repo id.
        surrogates: HPO-B also needs its XGBoost surrogates.
        raw: Also download the raw tables.
        dest: Local data folder; files keep their repo paths below it.

    Returns:
        ``dest``.
    """
    from huggingface_hub import snapshot_download
    allow = patterns(benchmarks, surrogates, raw)
    if allow:
        snapshot_download(repo, repo_type="dataset", local_dir=dest, allow_patterns=allow, token=get_token())
    return dest


def main(argv=None) -> None:
    """Command line: ``upload`` or ``download``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload", help="upload benchmarks/data to the dataset repo")
    up.add_argument("--repo", default=REPO)
    up.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    up.add_argument("--folders", nargs="+", default=FOLDERS, choices=FOLDERS)
    down = sub.add_parser("download", help="download what the benchmarks need")
    down.add_argument("--repo", default=REPO)
    down.add_argument("--benchmarks", nargs="+", default=["hpobench", "nasbench201", "hpob"])
    down.add_argument("--no-surrogates", action="store_true", help="HPO-B discrete protocol")
    down.add_argument("--raw", action="store_true", help="also download the raw tables")
    a = ap.parse_args(argv)
    if "<user>" in a.repo:
        ap.error("set the repo: --repo <user>/metagamebo-benchmarks (or HF_DATA_REPO)")
    if a.cmd == "upload":
        upload(a.repo, private=not a.public, folders=a.folders)
    else:
        print(download(a.benchmarks, a.repo, surrogates=not a.no_surrogates, raw=a.raw))


if __name__ == "__main__":
    main()
