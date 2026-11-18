"""Public runtime closure, CLI safety, archived isolation and scientific invariants."""
from pathlib import Path
import copy
import json
import subprocess
import sys

import pytest

from experiments.lfbo_gp_unified.protocol import ROOT,config,source_identity,save,sha
from experiments.lfbo_gp_unified.runner import fixture_config,run
from experiments.lfbo_gp_unified import cli,historical


def test_public_source_identity_contains_no_local_audit_evidence():
    c=config();sources=source_identity(c)
    assert c['schema_version']==2
    assert not any(n.startswith(('output/','reports/','results/')) for n in sources)
    assert all((ROOT/n).is_file() for n in sources)
    assert not any('author_email' in n for n in sources)
    assert 'vendor/metagamebo/MetaGameBO.py' in sources
    assert 'vendor/lfbo/model.py' in sources
    requirements=(ROOT/'requirements.txt').read_text().splitlines()
    assert all(f'{name}=={version}' in requirements for name,version in c['runtime']['packages'].items())


def test_entrypoint_default_is_track_B_and_does_not_run_without_mode():
    p=subprocess.run([sys.executable,'-B','main.py','--help'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0 and '--mode' in p.stdout
    assert '--experiment' not in p.stdout and '--evaluation' not in p.stdout
    p=subprocess.run([sys.executable,'-B','main.py'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode!=0 and '--mode' in p.stderr
    p=subprocess.run([sys.executable,'-B','main.py','--mode','compare'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode!=0 and '--run-id required' in p.stderr


def test_archived_reader_uses_snapshot_and_preserves_files(tmp_path,monkeypatch):
    folder=tmp_path/'old-run';c=fixture_config(config());c['schema_version']=1
    run(folder,'smoke','B','classifier',c,fixture=True)
    before={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    assert historical.is_historical(folder)
    assert historical.show(folder,'verify')==0
    monkeypatch.setattr(cli,'OUTPUT',tmp_path)
    assert cli.main(['--mode','show','--run-id',folder.name])==0
    assert before=={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    assert cli.main(['--mode','resume','--run-id',folder.name])==1


def test_archived_corruption_is_not_bypassed(tmp_path):
    folder=tmp_path/'old-corrupted';c=fixture_config(config());c['schema_version']=1
    run(folder,'smoke','B','classifier',c,fixture=True)
    source=folder/'source_snapshot/models/gp.py';source.write_text(source.read_text()+'\n')
    assert historical.show(folder,'verify')!=0


def test_missing_data_does_not_create_results(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'OUTPUT',tmp_path)
    for mode in ['show','verify','compare']:
        assert cli.main(['--mode',mode,'--run-id','missing'])==1
    assert not (tmp_path/'missing').exists()
