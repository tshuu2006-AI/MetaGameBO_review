"""Distributable benchmark tests; no private phase modules or large reruns."""
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiments.forrester_benchmark.objective import ROOT, AuthorTarget
from experiments.forrester_benchmark.protocol import DEFAULT_PROTOCOL, load, load_config, source_identity, validate
from experiments.forrester_benchmark.records import independent_cvar, numeric_rows, csv_rows
from experiments.forrester_benchmark.run import CountedTarget, read_completed, result_path, rid, run
from experiments.forrester_benchmark.verify import derive, stats, verify_published


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope='module')
def lock():
    return load()


def byte_map(folder):
    return {str(f.relative_to(folder)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in folder.rglob('*') if f.is_file()}


def test_public_source_has_no_private_phase_imports():
    for path in (ROOT/'experiments/forrester_benchmark').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('experiments.phase')
            elif isinstance(node, ast.Import):
                assert all(not n.name.startswith('experiments.phase') for n in node.names)
    assert not any('phase4_' in p for p in source_identity()['source_sha256'])


def test_fixed_author_dataset_and_shared_initialization(lock):
    p=validate(lock)
    assert len(p['manifest']['target_tasks'])==100
    assert [s['source_author_seed'] for s in p['manifest']['target_tasks']]==list(range(10000,10100))
    assert len({tuple(s[k] for k in ('a','b','c')) for s in p['manifest']['target_tasks']})==100
    for spec,design in zip(p['manifest']['target_tasks'],p['initial_design']):
        assert AuthorTarget(spec).evaluate(design['initial_X']).tolist()==design['initial_y']
    assert p['table1_reproduction'] is False


@pytest.mark.parametrize('method',['random','lfbo','pfn'])
def test_yaml_matches_measured_settings(lock,method):
    config,actual=load_config(f'configs/forrester_benchmark/{method}.yaml')
    assert actual==lock and config['algorithm']==method
    assert config['evaluation']['budget']==16 and config['evaluation']['n_init']==2


def test_changed_protocol_and_yaml_rejected(lock,tmp_path):
    changed=copy.deepcopy(lock);changed['payload']['evaluation']['budget']=17
    with pytest.raises(ValueError,match='identity'):validate(changed)
    config=(ROOT/'configs/forrester_benchmark/lfbo.yaml').read_text().replace('n_estimators: 300','n_estimators: 1000')
    path=tmp_path/'changed.yaml';path.write_text(config)
    with pytest.raises(ValueError,match='differs'):load_config(path)


@pytest.mark.parametrize('points',[[[1.01]],[[float('nan')]],[[float('inf')]],[[.1,.2]],[.5]])
def test_invalid_queries_do_not_spend_online_budget(lock,points):
    target=CountedTarget(lock['payload']['manifest']['target_tasks'][0])
    with pytest.raises(ValueError):target.evaluate(points)
    assert target.calls==0


def test_online_budget_cap(lock):
    target=CountedTarget(lock['payload']['manifest']['target_tasks'][0])
    target.evaluate(np.linspace(0,1,16)[:,None])
    assert target.calls==16
    with pytest.raises(RuntimeError,match='budget'):target.evaluate([[.5]])
    assert target.calls==16


def test_new_execution_cannot_write_published_artifacts(lock):
    with pytest.raises(ValueError,match='outside published'):
        run(lock,'random','full',output_root=ROOT/'artifacts/forrester_benchmark/raw')


def test_independent_statistics_and_fractional_tail():
    assert stats([0.,1.,2.,3.])['standard_error']==pytest.approx(np.std([0.,1.,2.,3.],ddof=1)/2)
    assert independent_cvar([0.,1.,2.],.5)==pytest.approx(2.5/1.5)
    records={m:[dict(task_id=str(i),final_metrics=dict(final_normalized_regret=v,
        final_simple_regret=v,final_best_raw_objective=v),elapsed_seconds=1.)
        for i,v in enumerate(values)] for m,values in {
            'random':[.2,.4,.6],'lfbo':[.1,.3,.5],'pfn':[.3,.1,.6]}.items()}
    paired=derive(records,{'tie_absolute_tolerance':1e-12})['paired_comparisons']
    assert [paired['pfn_minus_random'][k] for k in ('wins','ties','losses')]==[1,1,1]
    assert paired['lfbo_minus_random']['mean']==pytest.approx(-.1)


def test_published_results_all_300_runs_and_analysis():
    report,_=verify_published()
    assert report['status']=='PASS' and report['actual_completed_runs']==300
    assert report['actual_online_calls']==4800 and report['actual_unique_targets']==100
    assert report['published_checksums']=='PASS' and report['published_analysis']=='PASS'


def test_corrupted_completed_cache_fails_closed(lock,tmp_path):
    source=ROOT/'artifacts/forrester_benchmark/raw'/lock['protocol_id']/'random'
    identity=read(source/'experiment_identity.json')
    run_id=rid(lock['protocol_id'],'random',lock['payload']['initial_design'][0]['task_id'])
    shutil.copytree(source/'runs'/run_id,tmp_path/'runs'/run_id)
    assert read_completed(tmp_path,run_id,identity,lock['payload'])['task_id']==lock['payload']['initial_design'][0]['task_id']
    file=tmp_path/'runs'/run_id/'online_calls.jsonl';file.write_text(file.read_text()+'{}\n')
    with pytest.raises(ValueError,match='Modified'):read_completed(tmp_path,run_id,identity,lock['payload'])


def test_random_replay_public_interface_exact_and_resume_zero_calls(lock,tmp_path):
    from models.random_search import RandomSearch
    constructors=[]
    def public_only(**kwargs):
        assert set(kwargs)=={'bounds','seed'}
        model=RandomSearch(**kwargs);constructors.append(model);return model
    outcome=run(lock,'random','full',audit=True,output_root=tmp_path,optimizer_factory=public_only)
    assert outcome['new_online_objective_calls']==16 and len(constructors)==1
    folder=Path(outcome['output_directory'])
    original=ROOT/'artifacts/forrester_benchmark/raw'/lock['protocol_id']/'random'
    target=lock['payload']['initial_design'][0]['task_id']
    expected=[r for r in csv_rows(original/'observations.csv') if r['task_id']==target]
    assert numeric_rows(csv_rows(folder/'observations.csv'))==numeric_rows(expected)
    identity=read(folder/'experiment_identity.json')
    assert identity['source_fingerprint']==source_identity()['source_fingerprint']
    assert identity['historical_source_fingerprint']==lock['payload']['source_fingerprint']
    before=byte_map(folder)
    with patch.object(CountedTarget,'evaluate',side_effect=AssertionError('query on resume')):
        resumed=run(lock,'random','full',audit=True,resume=True,output_root=tmp_path,
            optimizer_factory=lambda **kwargs:pytest.fail('constructor on resume'))
    assert resumed['new_online_objective_calls']==0 and resumed['skipped_completed_runs']==1
    assert byte_map(folder)==before


def test_partial_replay_failure_preserved_and_resume(lock,tmp_path):
    original=CountedTarget.evaluate
    def interrupt(target,X):
        if target.calls==4:raise KeyboardInterrupt('TEST_ONLY_INTERRUPTION')
        return original(target,X)
    with patch.object(CountedTarget,'evaluate',interrupt):
        with pytest.raises(KeyboardInterrupt):run(lock,'random','full',audit=True,output_root=tmp_path)
    folder=result_path(lock,'random',root=tmp_path)
    assert not list((folder/'runs').iterdir())
    failures=list((folder/'failures').glob('*.failed.json'))
    assert len(failures)==1 and read(failures[0])['online_calls_before_failure']==4
    resumed=run(lock,'random','full',audit=True,resume=True,output_root=tmp_path)
    assert resumed['new_online_objective_calls']==16
    assert sum(len(f.read_text().splitlines()) for f in folder.rglob('online_calls.jsonl'))==20


def check_replay(lock,tmp_path,method):
    outcome=run(lock,method,'full',audit=True,output_root=tmp_path)
    folder=Path(outcome['output_directory'])
    target=lock['payload']['initial_design'][0]['task_id']
    original=ROOT/'artifacts/forrester_benchmark/raw'/lock['protocol_id']/method
    expected=[r for r in csv_rows(original/'observations.csv') if r['task_id']==target]
    assert numeric_rows(csv_rows(folder/'observations.csv'))==numeric_rows(expected)
    assert outcome['new_online_objective_calls']==16


def test_lfbo_replay_matches_original_measurement(lock,tmp_path):
    check_replay(lock,tmp_path,'lfbo')


@pytest.mark.pfn_real
def test_pfn_real_replay_matches_original_measurement(lock,tmp_path):
    check_replay(lock,tmp_path,'pfn')
