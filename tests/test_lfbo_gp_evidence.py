"""Evidence is measured, immutable and replayable without refitting."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.lfbo_gp_unified.protocol import config, sha, read, unseal, ROOT
from experiments.lfbo_gp_unified.runner import run, fixture_config
from experiments.lfbo_gp_unified.verification import verify
from experiments.lfbo_gp_unified.evidence import export_evidence, verify_evidence, replay_checkpoints


def hashes(folder):
    return {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}


def fixture_run(folder, policy='representative'):
    c = fixture_config(config())
    run(folder, 'smoke', 'B', 'classifier', c, fixture=True, checkpoint_policy=policy)
    return c


def deny_fit(monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.gaussian_process import GaussianProcessRegressor
    from models.gp import GP
    from experiments.lfbo_gp_unified import data
    def fail(*args, **kwargs):
        raise AssertionError('No fits or new objective data are permitted')
    for owner, name in [(RandomForestClassifier, 'fit'), (GaussianProcessRegressor, 'fit'),
                        (GP, 'fit'), (data, 'family_manifest')]:
        monkeypatch.setattr(owner, name, fail)


def test_checkpoint_export_replay_and_resume_do_not_refit_or_modify(monkeypatch, tmp_path):
    source, bundle = tmp_path/'run', tmp_path/'evidence'
    c = fixture_run(source)
    assert len(list(source.rglob('*.joblib'))) == 2
    before = hashes(source)
    deny_fit(monkeypatch)
    report = export_evidence(source, bundle)
    assert report['task_count'] == 3 and report['observations'] == 96
    assert report['checkpoint_status'] == 'SAVED_DURING_FIT'
    assert verify_evidence(bundle)['checkpoint_count'] == 2
    assert replay_checkpoints(bundle)['checkpoints_replayed'] == 2
    run(source, 'smoke', 'B', 'classifier', c, fixture=True, resume=True)
    assert before == hashes(source)
    sealed = hashes(bundle)
    verify_evidence(bundle); replay_checkpoints(bundle)
    assert sealed == hashes(bundle)
    with pytest.raises(ValueError, match='destination exists'):
        export_evidence(source, bundle)
    assert sealed == hashes(bundle)


def test_checkpoint_capture_does_not_change_predictions(tmp_path):
    plain, captured = tmp_path/'plain', tmp_path/'captured'
    fixture_run(plain, 'none'); fixture_run(captured)
    for p in (plain/'families/000/units').rglob('*.json'):
        a = unseal(p)['result']; b = unseal(captured/p.relative_to(plain))['result']
        for key in ('method', 'model_seed', 'training_X', 'predictions', 'task_loss'):
            assert a[key] == b[key]


def test_original_run_without_models_is_never_refitted(monkeypatch, tmp_path):
    source, bundle = tmp_path/'historical', tmp_path/'evidence'
    fixture_run(source, 'none')
    deny_fit(monkeypatch)
    before = hashes(source)
    export_evidence(source, bundle)
    manifest = read(bundle/'evidence_manifest.json')
    assert manifest['checkpoint_status'] == 'NOT_SAVED_DURING_ORIGINAL_RUN'
    assert not list(bundle.rglob('*.joblib'))
    with pytest.raises(ValueError, match='No fitted model checkpoints'):
        replay_checkpoints(bundle)
    assert before == hashes(source)


def test_verification_checks_model_bytes_without_loading(tmp_path, monkeypatch):
    source = tmp_path/'run'; fixture_run(source)
    import joblib
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Ordinary verification must not deserialize estimators'))
    verify(source)
    p = next(source.rglob('*.joblib')); p.write_bytes(p.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='checkpoint missing or changed'):
        verify(source)


def test_compact_verifier_rejects_loss_tampering_even_with_updated_hash(tmp_path):
    source, bundle = tmp_path/'run', tmp_path/'evidence'
    fixture_run(source); export_evidence(source, bundle)
    p = bundle/'classifier_predictions.npz'
    with np.load(p, allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    data['weighted_BCE'][0] += 1
    np.savez_compressed(p, **data)
    manifest = read(bundle/'evidence_manifest.json')
    manifest['files'][p.name] = sha(p)
    (bundle/'evidence_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        verify_evidence(bundle)


def test_compact_verification_imports_no_models_or_pickle(tmp_path):
    source, bundle = tmp_path/'run', tmp_path/'evidence'
    fixture_run(source); export_evidence(source, bundle)
    code = '''import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in ('models', 'sklearn', 'scipy', 'torch', 'joblib'):
        raise RuntimeError('Forbidden model import: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments.lfbo_gp_unified.evidence import verify_evidence
'''+f'assert verify_evidence({str(bundle)!r})["model_fits"] == 0\n'
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_missing_export_and_cli_guards_do_not_create_data(tmp_path):
    from experiments.lfbo_gp_unified.cli import main
    bundle = tmp_path/'new'
    with pytest.raises((ValueError, FileNotFoundError)):
        export_evidence(tmp_path/'missing', bundle)
    assert not bundle.exists()
    assert main(['--mode', 'export-evidence', '--run-id', 'missing']) == 1
    assert main(['--mode', 'verify-evidence']) == 1


def test_npz_arrays_are_decompressed_only_once_per_verification(tmp_path, monkeypatch):
    source, bundle = tmp_path/'run', tmp_path/'evidence'
    fixture_run(source); export_evidence(source, bundle)
    original = np.load
    opened = []
    class SingleRead:
        def __init__(self, *args, **kwargs):
            self.archive = original(*args, **kwargs)
            self.files = self.archive.files
            self.reads = {}
            opened.append(self)
        def __enter__(self): return self
        def __exit__(self, *exc): self.archive.close()
        def __getitem__(self, key):
            self.reads[key] = self.reads.get(key, 0)+1
            assert self.reads[key] == 1, 'Repeated decompression scales with tasks/rows'
            return self.archive[key]
    monkeypatch.setattr(np, 'load', SingleRead)
    verify_evidence(bundle)
    assert len(opened) == 2
    assert all(set(a.reads) == set(a.files) for a in opened)
