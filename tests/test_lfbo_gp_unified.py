"""Small synthetic fixtures; never execute production smoke/family/full."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import log_ndtr

from experiments.lfbo_gp_unified.protocol import *
from experiments.lfbo_gp_unified.data import prepare_task,family_manifest
from experiments.lfbo_gp_unified.probability import gaussian,logcdf
from experiments.lfbo_gp_unified.metrics import sample,task_summary,aggregate,calibration
from experiments.lfbo_gp_unified.runner import fixture_config,fixture_data,run
from experiments.lfbo_gp_unified.lfbo_runner import fit as rf_fit
from experiments.lfbo_gp_unified.gp_runner import fit as gp_fit
from experiments.lfbo_gp_unified.verification import audit_task,verify


@pytest.fixture(scope='module')
def c():return fixture_config(config())


@pytest.fixture(scope='module')
def task(c):return fixture_data(44,3,c)['tasks'][0]


def test_registered_20000_partitions_and_independent_streams():
    c=config();s=c['dataset']['split'];p=ROOT/s['registered_manifest'];assert sha(p)==s['manifest_sha256']
    values=read(p)['splits'];assert len(values)==20000 and len({(v['family_seed'],v['task_id']) for v in values})==20000
    for r in values:
        assert r['seed']==seed(r['family_seed'],'raw-split',r['task_id'])
        assert len(r['train'])==26 and len(r['validation'])==6 and not set(r['train'])&set(r['validation'])
        assert set(r['train'])|set(r['validation'])==set(range(32))
    assert len({seed(44,s,'meta0001') for s in c['dataset']['rng']['streams']})==len(c['dataset']['rng']['streams'])


def test_source_generation_reproducibility_and_UID0_excluded(c):
    # Generator fixture only, 64 observations; not a production benchmark run.
    a=family_manifest(44,2,c);b=family_manifest(44,2,c)
    assert a==b and [t['uid'] for t in a['tasks']]==[1,2]
    assert a['RNG_before']!=a['RNG_after'] and a['excluded_target_uid']==0


def test_same_raw_and_no_leakage(task,c):
    assert audit_task(task,44,c)
    tr=task['split']['train'];va=task['split']['validation'];assert len(tr)==26 and len(va)==6
    assert not {tuple(task['raw']['X'][i]) for i in tr}&{tuple(task['raw']['X'][i]) for i in va}
    assert [r['point_index'] for r in task['tracks']['A']['validation']]==va
    assert [r['point_index'] for r in task['tracks']['B']['validation'] if r['role']=='background_negative']==va


def test_training_only_tau_weights_and_GP_predictions(task,c):
    changed=copy.deepcopy(task['raw'])
    for rid in task['split']['validation']:changed['y'][rid]-=100
    new=prepare_task(task['descriptor'],changed,44,c)
    assert new['constants']==task['constants'] and new['tracks']['B']['training']==task['tracks']['B']['training']
    a=gp_fit(task,'A',c);b=gp_fit(new,'A',c)
    assert [r['probability'] for r in a['predictions']]==[r['probability'] for r in b['predictions']]
    bnew=gp_fit(new,'B',c);before=gp_fit(task,'B',c)
    lookup={r['point_index']:r['probability'] for r in before['predictions']}
    assert all(r['probability']==lookup[r['point_index']] for r in bnew['predictions'])
    for track in ['A','B']:
        original=rf_fit(task,track,c);altered=rf_fit(new,track,c)
        lookup={r['point_index']:r['probability'] for r in original['predictions']}
        assert all(r['probability']==lookup[r['point_index']] for r in altered['predictions'])


def test_degenerate_training_and_coordinate_overlap_rejected(task,c):
    raw=copy.deepcopy(task['raw']);raw['y']=[1.]*32
    with pytest.raises(ValueError,match='Degenerate'):prepare_task(task['descriptor'],raw,44,c)
    raw=copy.deepcopy(task['raw']);raw['X'][task['split']['validation'][0]]=raw['X'][task['split']['train'][0]]
    with pytest.raises(ValueError,match='coordinate overlap'):prepare_task(task['descriptor'],raw,44,c)


def test_frozen_source_weights_and_augmentation_order(task,c):
    rows=task['tracks']['B']['training'];n=task['constants']['n_good'];tr=task['split']['train'];y=task['raw']['y'];tau=task['constants']['tau']
    assert [r['point_index'] for r in rows[:n]]==[i for i in tr if y[i]<tau]
    assert [r['point_index'] for r in rows[n:]]==tr
    assert np.mean([r['weight'] for r in rows])==pytest.approx(1)
    val=task['tracks']['B']['validation'];assert not np.allclose([r['weight'] for r in val],1)
    assert all(r['weight']==task['constants']['c'] for r in val if r['label']==0)


@pytest.mark.parametrize('track',['A','B'])
def test_native_RF_reproducibility_class_mapping_and_no_overrides(task,c,track):
    a=rf_fit(task,track,c);b=rf_fit(task,track,c)
    assert a['predictions']==b['predictions'] and a['classes']==[0,1] and a['positive_column']==1
    assert a['RF_parameters']==dict(c['RF']['parameters'],random_state=a['classifier_seed'])
    assert len(a['estimator_random_states'])==c['fixture']['RF_trees']
    assert a['training_weights']==[r['weight'] for r in task['tracks'][track]['training']]


def test_RF1000_production_profile_on_single_synthetic_fixture(task):
    c=config();r=rf_fit(task,'B',c)
    assert r['RF_parameters']['n_estimators']==1000 and len(r['estimator_random_states'])==1000


@pytest.mark.parametrize('z',[-1000.,-100.,-20.,-8.1,-8.,-3.,0.,3.,8.1,20.,1000.])
def test_stable_logcdf(z,c):
    assert logcdf(z,c)==pytest.approx(log_ndtr(z),rel=2e-12,abs=2e-12)
    s=gaussian(-z,1.,0.,.7,c)
    assert all(math.isfinite(v) for v in s.values()) and 0<=s['augmented_probability']<=1


@pytest.mark.parametrize('z',[-12.,-8.1,-5.,-1.,0.,1.,6.])
def test_EI_integral_and_derived_probability(z,c):
    # Stable independent integral for negative tails; no cancellation in reference.
    ref=quad(lambda t:t*math.exp(-.5*(t-z)**2)/math.sqrt(2*math.pi),0,np.inf,epsabs=1e-40,epsrel=1e-12)[0]
    s=gaussian(-z,1.,0.,.7,c)
    assert s['EI']==pytest.approx(ref,rel=2e-10,abs=1e-40)
    assert s['augmented_probability']==pytest.approx(ref/(ref+.7),rel=2e-10,abs=1e-40)


def test_Gaussian_zero_variance_failure_explicit(c):
    for variance in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):gaussian(0.,variance,1.,.4,c)


def test_posterior_Monte_Carlo_and_weighted_risk_minimizer(c):
    rng=np.random.default_rng(44);mu=.1;sigma=.8;tau=.4;U=.35;balance=.7
    draws=rng.normal(mu,sigma,700000);utility=np.maximum(tau-draws,0.)
    analytical=gaussian(mu,sigma*sigma,tau,U,c);mc=utility.mean()/(utility.mean()+U)
    assert mc==pytest.approx(analytical['augmented_probability'],rel=.005)
    q=analytical['augmented_probability'];A=balance*analytical['EI']/U;B=balance
    risk=lambda p:-A*math.log(p)-B*math.log1p(-p)
    assert risk(q)<risk(q-.05) and risk(q)<risk(q+.05)
    assert -A/q+B/(1-q)==pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize('N,k',[(3,1),(200,39)])
def test_literal_task_CVaR_and_family_SE(N,k):
    values=np.linspace(.1,3,N).tolist();r=task_summary(values,.8)
    assert r['N']==N and r['k']==k and r['CVaR']==pytest.approx(np.mean(sorted(values,reverse=True)[:k]))
    assert aggregate([1.])['SE'] is None and aggregate([1.,2.,3.])['SE']==pytest.approx(1/math.sqrt(3))


def test_stable_row_denominator_and_calibration(c):
    r=sample(-1000.,0.,1,3.)
    assert r['BCE']==1000 and r['weighted_BCE']==3000
    assert calibration([dict(probability=.2,label=0)]*19,c)['status']=='INSUFFICIENT_ROWS'
    assert calibration([dict(probability=.2,label=0)]*20,c)['Brier']==pytest.approx(.04)


@pytest.mark.parametrize('track',['A','B'])
def test_end_to_end_synthetic_fixture_resume_readonly_and_corruption(tmp_path,c,track,monkeypatch):
    folder=tmp_path/('fixture-'+track);run(folder,'smoke',track,'classifier',c,fixture=True)
    summary,_=verify(folder);assert summary['status']==c['fixture']['status'] and summary['families']==1
    assert all(m['mean_BCE']['SE'] is None for m in summary['methods'].values())
    paths=list((folder/'families/000/units').rglob('*.json'));before={str(p):sha(p) for p in paths}
    run(folder,'smoke',track,'classifier',c,resume=True,fixture=True)
    assert before=={str(p):sha(p) for p in paths}
    code="import builtins\nf=builtins.__import__\ndef guard(n,*a,**kw):\n if n.split('.')[0] in ('models','numpy','scipy','sklearn','torch','wandb'):raise RuntimeError(n)\n return f(n,*a,**kw)\nbuiltins.__import__=guard\nfrom experiments.lfbo_gp_unified.verification import verify\n"
    code+=f"verify({str(folder)!r})"
    subprocess.run([sys.executable,'-S','-B','-c',code],cwd=ROOT,check=True)
    from experiments.lfbo_gp_unified import cli
    monkeypatch.setattr(cli,'OUTPUT',tmp_path)
    for mode in ['show','verify','compare']:assert cli.main(['--mode',mode,'--run-id',folder.name])==0
    p=paths[0];item=read(p);item['payload']['result']['predictions'][0]['weighted_BCE']+=1;item['sha256']=canonical(item['payload']);save(p,item,replace=True)
    with pytest.raises(ValueError,match='Loss weighted_BCE'):verify(folder)


def test_config_source_and_cli_guards(tmp_path,c,monkeypatch):
    from experiments.lfbo_gp_unified.cli import main,preflight
    assert preflight(config())['status']=='PASS'
    assert main(['--mode','full','--run-id','MUST_NOT_CREATE'])==1
    assert not (OUTPUT/'MUST_NOT_CREATE').exists()
    assert main(['--mode','show','--run-id','MUST_NOT_CREATE'])==1
    p=tmp_path/'fixture';run(p,'smoke','A','classifier',c,fixture=True)
    saved=read(p/'config.json');saved['classification']['gamma']=.2;save(p/'config.json',saved,replace=True)
    with pytest.raises(ValueError,match='Locked file'):verify(p)


def test_source_snapshot_corruption(tmp_path,c):
    p=tmp_path/'fixture';run(p,'smoke','A','classifier',c,fixture=True)
    source=p/'source_snapshot/models/gp.py';source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='Source snapshot'):verify(p)


def test_optional_paired_BO_fixture(tmp_path,c):
    p=tmp_path/'fixture-bo';run(p,'smoke','A','bo',c,fixture=True)
    summary,_=verify(p);assert summary['evaluation']=='bo' and summary['track'] is None
    data=unseal(p/'families/000/dataset.json')['data']['tasks'][0]
    for method in ['LFBO_EI','GP_EI']:
        r=unseal(p/'families/000/units'/(method+'.json'))['result']
        assert r['online_objective_calls']==c['BO']['budget']==4
        assert [[v['x']] for v in r['observations'][:2]]==data['initial_X']
    before={str(f):sha(f) for f in (p/'families/000/units').glob('*.json')}
    run(p,'smoke','A','bo',c,fixture=True,resume=True)
    assert before=={str(f):sha(f) for f in (p/'families/000/units').glob('*.json')}


def test_failed_fit_recorded_and_partial_resume_reuses_completed_units(tmp_path,c,monkeypatch):
    from experiments.lfbo_gp_unified import gp_runner
    p=tmp_path/'interrupted';native=gp_runner.fit
    def fail(t,track,c):
        if t['task_id']=='meta0002':raise RuntimeError('injected GP failure')
        return native(t,track,c)
    monkeypatch.setattr(gp_runner,'fit',fail)
    with pytest.raises(RuntimeError,match='injected'):run(p,'smoke','B','classifier',c,fixture=True)
    assert len(list((p/'failures').glob('*.json')))==1 and not (p/'summary.json').exists()
    paths=list((p/'families/000/units').rglob('*.json'));assert len(paths)==3
    before={str(f):sha(f) for f in paths}
    monkeypatch.setattr(gp_runner,'fit',native)
    run(p,'smoke','B','classifier',c,fixture=True,resume=True)
    assert before=={str(f):sha(f) for f in paths};assert verify(p)[0]['tasks_per_family']==3


def test_parallel_two_family_fixture_has_family_level_SE(tmp_path,c):
    p=tmp_path/'two-family-fixture';c=copy.deepcopy(c);c['fixture']['seeds']=[44,45]
    run(p,'full','A','classifier',c,workers=2,fixture=True)
    summary,stats=verify(p);assert summary['families']==2
    for method,r in summary['methods'].items():
        expected=aggregate([family['methods'][method]['mean_BCE'] for family in stats])
        assert r['mean_BCE']==expected and r['mean_BCE']['SE'] is not None


def test_preflight_import_guard_no_models_or_numeric_engines():
    code="import builtins\nf=builtins.__import__\ndef guard(n,*a,**kw):\n if n.split('.')[0] in ('models','numpy','scipy','sklearn','torch','wandb','pfns4bo'):raise RuntimeError(n)\n return f(n,*a,**kw)\nbuiltins.__import__=guard\nfrom experiments.lfbo_gp_unified.cli import preflight\nfrom experiments.lfbo_gp_unified.protocol import config\nassert preflight(config())['model_fits']==0\n"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_full_CLI_dispatch_is_only_mocked_after_acknowledgements(monkeypatch):
    from experiments.lfbo_gp_unified import cli
    calls=[];monkeypatch.setattr(cli.subprocess,'call',lambda args,**kw:calls.append(args) or 0)
    assert cli.main(['--mode','full','--track','B','--run-id','MUST_NOT_CREATE','--confirm-full','--acknowledge-independent-protocol'])==0
    assert len(calls)==1 and not (OUTPUT/'MUST_NOT_CREATE').exists()
