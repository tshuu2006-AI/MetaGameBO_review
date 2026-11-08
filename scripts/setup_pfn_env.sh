#!/usr/bin/env bash
# Separate, tested Python 3.11 CPU environment. Never installs into .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv-pfn/bin/python ]]; then
  pfn_python="${PFN_PYTHON:-python3.11}"
  "$pfn_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
  "$pfn_python" -m venv .venv-pfn
fi
.venv-pfn/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "PFN environment must use Python 3.11"'
.venv-pfn/bin/python -m pip install -r requirements-pfn.txt
.venv-pfn/bin/python -m pip check
.venv-pfn/bin/python -m scripts.fetch_pfn_checkpoint
