"""Verify archived runs with their immutable original code, without fitting."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .common import require


def is_historical(folder):
    path = Path(folder) / 'config.json'
    return path.is_file() and json.loads(path.read_text()).get('schema_version', 1) < 2


def show(folder, mode):
    folder = Path(folder).resolve()
    require(mode in ('verify', 'show', 'compare'), 'Archived runs are read-only')
    snapshot = folder / 'source_snapshot'
    require((snapshot / 'experiments/lfbo_gp_unified/verification.py').is_file(),
            'Archived verifier snapshot missing; original run cannot be verified')
    # Separate processes avoid mixing current modules with archived ones.
    code = '''import pathlib, sys
snapshot, folder, mode = sys.argv[1:]
sys.path.insert(0, snapshot)
from experiments.lfbo_gp_unified.verification import verify
summary, _ = verify(folder)
if mode == 'verify':
    print('PASS: archived raw results and immutable original source snapshot; no model imports or fits.')
else:
    print((pathlib.Path(folder) / 'comparison.md').read_text())
'''
    return subprocess.call([sys.executable, '-I', '-S', '-B', '-c', code,
                            str(snapshot), str(folder), mode])
