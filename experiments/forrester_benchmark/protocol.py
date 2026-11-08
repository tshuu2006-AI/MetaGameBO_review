"""Load the measured design and identify executions of the distributable runner.

The historical protocol and run IDs are retained verbatim. Historical source
paths in that protocol are provenance, not imports of local development phases.
New executions additionally bind the current runner source fingerprint, so they
cannot silently reuse historical caches or caches from different code.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.forrester import derive_seed
from experiments.evaluate import content_hash
from experiments.forrester_benchmark.objective import AUTHOR_SHA, ROOT, AuthorTarget
from models.lfbo import validate_lfbo_options
from models.pfn import validate_pfn_options
from models.pfn_backend import CHECKPOINT_SHA256

PROTOCOL_ID = '5ca7ade2c95861f2cd1d700471907f203aaa387a60e8ef36d644ff3b61e7b7f6'
DEFAULT_PROTOCOL = ROOT / 'configs/forrester_benchmark/protocol.json'
MODEL_OPTIONS = {
    'random': {},
    'lfbo': validate_lfbo_options(dict(classifier='rf', gamma=1/3, weight_type='ei',
        n_estimators=300, n_candidates=5000, n_jobs=1)),
    'pfn': validate_pfn_options(dict(model_name='hebo_plus', acquisition='ei',
        n_candidates=256, candidate_batch_size=64, device='cpu', cpu_threads=2,
        candidate_strategy='uniform', context_transform='observed_standardize')),
}
EVALUATION = dict(budget=16, n_init=2, budget_mode='total', seeds=[0], meta_train=False,
    cvar_alpha=.8, oracle_grid_size=4097, oracle_xatol=1e-12, regret_tolerance=1e-8)


def checkpoint_identity(metadata):
    return {k: v for k, v in metadata.items() if k != 'path'}


def environment_identity(metadata):
    return dict(python=metadata['python'].split()[0], packages=metadata['packages'])


def source_identity():
    """Only distributable runtime files; local audit/pilot files do not affect it."""
    files = list((ROOT/'experiments/forrester_benchmark').glob('*.py'))
    for folder in ('models', 'benchmarks'):
        files.extend((ROOT/folder).glob('*.py'))
    files.extend(ROOT/'experiments'/name for name in ('__init__.py','evaluate.py','metrics.py','tracking.py'))
    files.append(ROOT/'configs/forrester_benchmark/author_snapshot/forrester.py')
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    return dict(source_sha256=hashes, source_fingerprint=content_hash(hashes))


def validate(lock, mode=None):
    if set(lock) != {'protocol_id','payload'}:
        raise ValueError('Malformed benchmark protocol')
    p = lock['payload']
    if lock['protocol_id'] != PROTOCOL_ID or content_hash(p) != PROTOCOL_ID:
        raise ValueError('Measured protocol identity changed')
    if p['mode'] != 'full' or mode not in (None,'full'):
        raise ValueError('Only the completed full benchmark is distributed')
    if p['table1_reproduction'] is not False or p['paper_equivalence_status'] != 'UNVERIFIED':
        raise ValueError('Unsupported paper claim')
    if p['evaluation'] != EVALUATION or p['model_options'] != MODEL_OPTIONS:
        raise ValueError('Measured settings changed')
    if p['expected'] != dict(unique_targets=100, runs_per_algorithm=100, algorithms=3,
            total_runs=300, online_calls_per_algorithm=1600, total_online_calls=4800):
        raise ValueError('Wrong benchmark size')
    for key in ('manifest','initial_design','references'):
        if content_hash(p[key]) != p[key+'_sha256']:
            raise ValueError('Changed evaluator component: '+key)
    specs = p['manifest']['target_tasks']
    if len(specs)!=100 or [s['source_author_seed'] for s in specs]!=list(range(10000,10100)):
        raise ValueError('Wrong author seed map')
    ids = [s['task_id'] for s in specs]
    if len(set(ids))!=100 or len({tuple(s[k] for k in ('a','b','c')) for s in specs})!=100:
        raise ValueError('Repeated target')
    if len(p['initial_design'])!=100 or set(p['references'])!=set(ids):
        raise ValueError('Missing evaluator inputs')
    for spec,d in zip(specs,p['initial_design']):
        if spec['source_hash']!=AUTHOR_SHA or spec['author_uid']!=0 or d['task_id']!=spec['task_id'] or d['seed']!=0:
            raise ValueError('Wrong target provenance')
        init_seed=derive_seed(0,'initialization',spec['task_id'])
        X=np.random.default_rng(init_seed).uniform(0,1,(2,1))
        if d['initialization_seed']!=init_seed or d['optimizer_seed']!=derive_seed(0,'optimizer',spec['task_id']):
            raise ValueError('Wrong RNG stream')
        if X.tolist()!=d['initial_X'] or AuthorTarget(spec).evaluate(X).tolist()!=d['initial_y']:
            raise ValueError('Initialization differs from measured design')
    if p['checkpoint']['sha256']!=CHECKPOINT_SHA256:
        raise ValueError('Unexpected checkpoint')
    # Preserve compatibility with the actually measured core, without importing
    # or distributing any of the private phase-specific orchestration modules.
    for relative,digest in p['source_sha256'].items():
        path=Path(relative)
        if path.parts[0] in ('models','benchmarks') or relative in (
                'experiments/__init__.py','experiments/evaluate.py','experiments/metrics.py','experiments/tracking.py'):
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Measured core changed: '+relative)
    return copy.deepcopy(p)


def load(path=DEFAULT_PROTOCOL, mode=None):
    path=Path(path)
    if not path.is_absolute(): path=ROOT/path
    lock=json.loads(path.read_text())
    validate(lock,mode)
    return lock


def load_config(path):
    path=Path(path)
    if not path.is_absolute(): path=ROOT/path
    config=yaml.safe_load(path.read_text())
    expected={'protocol_path','protocol_id','algorithm','model_options','evaluation','table1_reproduction','output_root'}
    if not isinstance(config,dict) or set(config)!=expected:
        raise ValueError('Malformed benchmark YAML')
    lock=load(config['protocol_path'])
    method=config['algorithm']
    if method not in MODEL_OPTIONS or config['protocol_id']!=lock['protocol_id']:
        raise ValueError('Algorithm/protocol mismatch')
    if config['model_options']!=MODEL_OPTIONS[method] or config['evaluation']!=EVALUATION or config['table1_reproduction'] is not False:
        raise ValueError('YAML differs from measured protocol')
    return config,lock
