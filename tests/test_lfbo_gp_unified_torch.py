"""PyTorch and original source AST parity; arithmetic and saved synthetic fixtures only."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from source_reference import original_validation_metric
from experiments.lfbo_gp_unified.metrics import sample,task_summary,aggregate

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('N',[3,200])
@pytest.mark.parametrize('unit',[True,False])
def test_BCE_PyTorch_source_loop_and_literal_CVaR(N,unit):
    rng=np.random.default_rng(44);ids=np.repeat(np.arange(1,N+1),rng.integers(1,9,N));logits=rng.normal(0,10,len(ids))
    labels=rng.integers(0,2,len(ids));weights=np.ones(len(ids)) if unit else rng.lognormal(0,1,len(ids))
    lp=-np.logaddexp(0,-logits);lm=-np.logaddexp(0,logits)
    rows=[sample(float(p),float(q),int(z),float(w))['weighted_BCE'] for p,q,z,w in zip(lp,lm,labels,weights)]
    reference=torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits),torch.tensor(labels,dtype=torch.float64),weight=torch.tensor(weights),reduction='none').numpy()
    np.testing.assert_allclose(rows,reference,rtol=1e-13,atol=1e-13)
    values,k,risk=original_validation_metric()(torch.tensor(ids),torch.tensor(labels,dtype=torch.float64),torch.tensor(logits),torch.tensor(weights),.8)
    tasks=[float(np.mean(np.array(rows)[ids==i])) for i in np.unique(ids)];actual=task_summary(tasks,.8)
    assert k==actual['k']==(1 if N==3 else 39)
    np.testing.assert_allclose(values,tasks,rtol=1e-13,atol=1e-13);assert actual['CVaR']==pytest.approx(risk)


def test_saved_real_model_predictions_on_synthetic_fixtures():
    path=ROOT/'tests/fixtures/lfbo_gp/fixture_rows_for_torch.json'
    assert path.is_file(), 'Committed numerical parity fixture is missing'
    for record in json.loads(path.read_text())['fixtures']:
        rows=record['rows'];ids=[r['uid'] for r in rows]
        z=torch.tensor([r['label'] for r in rows],dtype=torch.float64);w=torch.tensor([r['weight'] for r in rows],dtype=torch.float64);logits=torch.tensor([r['logit'] for r in rows],dtype=torch.float64)
        measured=torch.nn.functional.binary_cross_entropy_with_logits(logits,z,weight=w,reduction='none').numpy()
        np.testing.assert_allclose(measured,[r['weighted_BCE'] for r in rows],rtol=1e-11,atol=1e-12)
        values,k,risk=original_validation_metric()(torch.tensor(ids),z,logits,w,.8)
        assert k==record['metrics']['k']==1
        np.testing.assert_allclose([np.mean(values),risk],[record['metrics']['mean_BCE'],record['metrics']['CVaR']],rtol=1e-11,atol=1e-12)


def test_100_family_aggregation_not_20000_independent_runs():
    means=[];risks=[];pooled=[]
    for i in range(100):
        losses=(np.linspace(0,1,200)+(i%3)*.5).tolist();r=task_summary(losses,.8);means.append(r['mean_BCE']);risks.append(r['CVaR']);pooled.extend(losses)
    a=aggregate(means);b=aggregate(risks)
    assert a['runs']==100 and a['SE']==pytest.approx(np.std(means,ddof=1)/10)
    assert b['mean']!=pytest.approx(task_summary(pooled,.8)['CVaR'])
