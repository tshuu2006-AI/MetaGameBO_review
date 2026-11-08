"""Raw-record checks: formula, budget, initialization, model diagnostics and metrics."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-11)

def independent_cvar(values,alpha=.8):
    sorted_values=sorted(map(float,values),reverse=True);mass=len(sorted_values)*(1-alpha);left=mass;total=0.
    for value in sorted_values:
        weight=min(1.,left);total+=weight*value;left-=weight
        if left<=0:break
    return total/mass

def verify_record(record,spec,design,reference,method):
    obs=record['observations']
    assert len(obs)==16 and record['task_id']==spec['task_id'] and record['seed']==design['seed']
    assert record['optimizer_seed']==design['optimizer_seed'] and record['initialization_seed']==design['initialization_seed']
    assert record['algorithm']==method and record['reference']==reference
    assert [o['step'] for o in obs]==list(range(1,17))
    assert [o['query_index'] for o in obs]==list(range(16))
    assert [o['phase'] for o in obs]==['initial']*2+['sequential']*14
    assert [o['bo_step'] for o in obs]==[0,0,*range(1,15)]
    assert all(o['task_id']==spec['task_id'] and o['seed']==design['seed'] and o['run_id']==record['run_id'] and o['algorithm']==method for o in obs)
    a=np.array([[o[k] for k in ('x','y','best_so_far','simple_regret','normalized_regret','elapsed_seconds')] for o in obs])
    assert np.isfinite(a).all() and np.all((a[:,0]>=0)&(a[:,0]<=1))
    x=a[:,0];y=spec['a']*(6*x-2)**2*np.sin(12*x-4)+spec['b']*(x-.5)-spec['c'];close(y,a[:,1])
    best=np.minimum.accumulate(a[:,1]);simple=np.maximum(best-reference['f_star_estimate'],0)
    norm=np.maximum(best-reference['reference_min'],0)/(reference['reference_max']-reference['reference_min'])
    assert np.all(best-reference['reference_min']>=-1e-8)
    assert np.all(np.diff(a[:,2])<=0) and np.all(a[:,3:]>=0) and np.all(np.diff(a[:,5])>=0)
    close(best,a[:,2]);close(simple,a[:,3]);close(norm,a[:,4])
    assert [(o['x'],o['y']) for o in obs[:2]]==list(zip(np.asarray(design['initial_X'])[:,0],design['initial_y']))
    duplicates=0
    for index in range(2,16):
        count=int(np.isclose(x[index],x[:index],atol=1e-8,rtol=0).sum());duplicates+=count
        if method!='random':assert count==0
    for metric,value in [('final_best_raw_objective',best[-1]),('final_simple_regret',simple[-1]),('final_normalized_regret',norm[-1])]:
        close(value,record['final_metrics'][metric])
    assert record['final_metrics']['budget']==16 and record['final_metrics']['n_init']==2
    assert np.isfinite(record['elapsed_seconds']) and record['elapsed_seconds']>=a[-1,5]
    diag=record['optimizer_diagnostics']
    if method=='pfn':
        assert diag['backend']['backend']=='official_pfns4bo_TransformerBOMethod'
        assert diag['backend']['pretrained_weights_frozen'] and diag['backend']['model_eval']
        assert diag['backend']['checkpoint']['sha256']=='cc7f6f4b9b06e59987e42845b7b0d31ffa5b414b9eddfe14d88b25120e3cd4f8'
        assert diag['gradient_updates']==0 and diag['group_meta_train'] is False and diag['context_size']==16
        assert len(diag['events'])==14 and diag['model_forward_calls']==70
        assert [e['n_context'] for e in diag['events']]==list(range(2,16))
        assert all(e['status']=='scored' and e['backend_called'] and np.isfinite(e['selected_acquisition']) and e['selected_acquisition']>=0 for e in diag['events'])
    elif method=='lfbo':
        assert diag['fit_count']==15 and len(diag['acquisition_events'])==14 and not diag['fallback_events']
        assert diag['model_options']['n_estimators']==300 and diag['model_options']['n_candidates']==5000
    else:assert diag is None
    assert record['wandb']==dict(mode='disabled',id=None,url=None,uploaded=False)
    return dict(online_calls=16,initial_calls=2,sequential_calls=14,duplicate_sequential_proposals=duplicates,
                max_formula_abs_difference=float(np.max(np.abs(y-a[:,1]))))

def numeric_rows(table):
    fields=('task_id','seed','query_index','step','phase','bo_step','algorithm','x','y','best_so_far','simple_regret','normalized_regret')
    return [{k:r[k] for k in fields} for r in table]
