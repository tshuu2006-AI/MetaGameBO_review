"""Execute the checksum-verified base-EI preparation component unchanged."""
from __future__ import annotations

import ast
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'configs/lfbo_gp_unified/source_contract.json'


def source_check():
    contract = json.loads(CONTRACT.read_text())
    for entry in contract['sources']:
        raw = (ROOT / entry['local_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Author source changed: ' + entry['local_path'])
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if blob != entry['git_blob_sha']:
            raise ValueError('GitHub blob identity mismatch: ' + entry['local_path'])
    return contract


@functools.lru_cache(maxsize=1)
def reweight_function():
    contract = source_check()
    entry = next(e for e in contract['sources'] if e['path'] == 'MetaGameBO.py')
    path = ROOT / entry['local_path']
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MetaGameBO')
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_reweight_samples')
    node.decorator_list = []
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_reweight_samples']


def prepare_ei_rows(X, y, gamma=1 / 3):
    """Exact source values/order; does not choose a train/validation protocol."""
    source_check()
    points = np.asarray(X, dtype=np.float64)
    values = np.asarray(y, dtype=np.float64)
    if (points.ndim != 2 or values.ndim != 1 or len(points) != len(values)
            or len(values) < 2 or not np.isfinite(points).all() or not np.isfinite(values).all()):
        raise ValueError('Need finite aligned raw observations')
    if not 0 < gamma < 1:
        raise ValueError('gamma must be between zero and one')
    tau = float(np.quantile(np.unique(values), gamma))
    good = np.flatnonzero(values < tau)
    if not len(good):
        raise ValueError('Degenerate source EI utility; do not replace with PI weights')
    features, labels, weights = reweight_function()(points, values, gamma, weight_type='ei')
    ids = np.r_[good, np.arange(len(values))]
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise FloatingPointError('Invalid original EI weights')
    np.testing.assert_array_equal(features, points[ids])
    return dict(X=features, labels=labels, weights=weights, raw_indices=ids,
                roles=['positive_improvement'] * len(good) + ['background_negative'] * len(values),
                tau=tau, gamma=gamma, scope='EXACT_BASE_EI_COMPONENT_NOT_TABLE1_EVALUATOR')
