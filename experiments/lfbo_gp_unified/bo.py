"""Optional paired BO. Classifier probability evaluators are never BO scores."""
from __future__ import annotations
import math
import statistics
from .protocol import *


def prepare(family,c,fixture=False):
    import numpy as np
    from .generator import plan
    from .objective import AuthorTarget
    from experiments.metrics import estimate_reference_range
    spec=plan(family,0)['target'] if not fixture else dict(uid=0,task_id='target0000',a=1.,b=0.,c=0.)
    rng=np.random.default_rng(seed(family,'bo-initial',spec['task_id']));before=rng.bit_generator.state
    initial=rng.uniform(0,1,(c['BO']['n_init'],1)).tolist();pools=[]
    for step in range(c['BO']['n_init'],c['BO']['budget']):
        s=seed(family,'bo-candidates',f"{spec['task_id']}:{step}");r=np.random.default_rng(s)
        pools.append(dict(step=step,seed=s,RNG_before=r.bit_generator.state,X=r.uniform(0,1,(c['BO']['candidates'],1)).tolist(),RNG_after=r.bit_generator.state))
    target=AuthorTarget(spec);refs=estimate_reference_range(target,grid_size=c['BO']['reference_grid'],xatol=c['BO']['reference_xatol'])
    return dict(family_seed=family,excluded_target_uid=0,tasks=[dict(descriptor=spec,initial_X=initial,candidate_pools=pools,references=refs,
        reference_offline_evaluations=target.evaluations,initial_RNG_before=before,initial_RNG_after=rng.bit_generator.state)],
        objective_observations=0,status='FIXTURE_NOT_BENCHMARK' if fixture else 'PAIRED_BO_INDEPENDENT')


def execute(t,family,method,c):
    import numpy as np
    from models.gp import GP
    from models.lfbo import LFBO
    from .objective import AuthorTarget
    from experiments.metrics import normalized_regret
    spec=t['descriptor'];stream='bo-lfbo' if method=='LFBO_EI' else 'bo-gp';s=seed(family,stream,spec['task_id'])
    model=LFBO(c['dataset']['domain'],seed=s,**c['RF']['owner_options']) if method=='LFBO_EI' else GP(c['dataset']['domain'],seed=s,**c['GP'])
    target=AuthorTarget(spec);X=np.array(t['initial_X']);y=target.evaluate(X);model.observe(X,y);observations=[dict(index=i+1,x=float(x[0]),y=float(v)) for i,(x,v) in enumerate(zip(X,y))]
    proposals=[];fits=[dict(training_X=X.tolist(),training_y=y.tolist(),diagnostics=model.diagnostics)]
    for pool in t['candidate_pools']:
        candidates=np.array(pool['X']);history=np.array([r['x'] for r in observations])
        mask=~np.any(np.isclose(candidates[:,0,None],history[None,:],rtol=0,atol=c['RF']['owner_options']['duplicate_atol']),axis=1)
        available=candidates[mask];require(len(available)>0,'Empty shared BO candidate pool')
        if method=='LFBO_EI':
            require(model.model is not None,'LFBO BO fallback forbidden')
            column=np.flatnonzero(model.model.classes_==1);require(len(column)==1,'LFBO BO positive class')
            scores=model.model.predict_proba(available)[:,column[0]]
        else:scores=model.expected_improvement(available)
        selected=int(np.argmax(scores));point=available[[selected]];value=target.evaluate(point);model.observe(point,value)
        proposals.append(dict(step=pool['step'],candidate_pool_sha256=canonical(pool),selected_original_index=int(np.flatnonzero(mask)[selected]),score=float(scores[selected]),score_semantics='native LFBO augmented-class score' if method=='LFBO_EI' else 'native minimization Expected Improvement',training_rows_before=len(observations)))
        observations.append(dict(index=len(observations)+1,x=float(point[0,0]),y=float(value[0])))
        hx,hy=model.history.arrays;fits.append(dict(training_X=hx.tolist(),training_y=hy.tolist(),diagnostics=model.diagnostics))
    ref=t['references'];best=np.minimum.accumulate([r['y'] for r in observations]);regret=normalized_regret(best,ref['reference_min'],ref['reference_max'],c['BO']['regret_tolerance']).tolist()
    require(target.evaluations==c['BO']['budget'],'Online BO budget')
    return dict(method=method,model_seed=s,online_objective_calls=target.evaluations,observations=observations,proposals=proposals,
        fits=fits,model_fits=len(fits),regret_curve=regret,final_normalized_regret=regret[-1],reference=ref)


def verify_family(folder,index,d,c,job,pid):
    directory=folder/'families'/f'{index:03d}';t=d['tasks'][0];spec=t['descriptor'];stats={};units=[]
    require(spec['uid']==0 and len(t['initial_X'])==c['BO']['n_init'],'BO target/init identity')
    for method in ('LFBO_EI','GP_EI'):
        item=unseal(directory/'units'/(method+'.json'));r=item['result'];require(item['identity']==dict(protocol_id=pid,dataset_sha256=sha(directory/'dataset.json'),family_index=index,method=method),'BO unit identity')
        require(r['method']==method and len(r['observations'])==c['BO']['budget']==r['online_objective_calls'],'BO budget/cache corruption')
        require([[o['x']] for o in r['observations'][:c['BO']['n_init']]]==t['initial_X'],'Shared BO initialization')
        best=math.inf;ref=t['references'];curve=[]
        for row in r['observations']:
            x=row['x'];expected=spec['a']*(6*x-2)**2*math.sin(12*x-4)+spec['b']*(x-.5)-spec['c']
            require(math.isclose(expected,row['y'],rel_tol=1e-12,abs_tol=1e-12),'BO raw equation')
            best=min(best,row['y']);require(best>=ref['reference_min']-c['BO']['regret_tolerance'] and best<=ref['reference_max']+c['BO']['regret_tolerance'],'Invalid BO reference range')
            value=max(0.,best-ref['reference_min'])/(ref['reference_max']-ref['reference_min']);curve.append(value)
        require(len(r['proposals'])==len(t['candidate_pools']),'BO missing proposal')
        for proposal,pool,observation in zip(r['proposals'],t['candidate_pools'],r['observations'][c['BO']['n_init']:]):
            require(proposal['candidate_pool_sha256']==canonical(pool) and pool['X'][proposal['selected_original_index']]==[observation['x']],'BO candidate trace')
        require(all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(curve,r['regret_curve'])),'BO regret curve')
        require(math.isclose(curve[-1],r['final_normalized_regret'],rel_tol=1e-12,abs_tol=1e-12),'BO final regret')
        stats[method]=dict(final_normalized_regret=curve[-1]);units.append((spec['task_id'],r))
    return dict(family_index=index,seed=d['family_seed'],methods=stats),units,d


def aggregate_bo(stats,c):
    result={};alpha=c['classification']['alpha']
    for method in ('LFBO_EI','GP_EI'):
        values=sorted((r['methods'][method]['final_normalized_regret'] for r in stats),reverse=True);mass=len(values)*(1-alpha)
        result[method]=dict(median=statistics.median(values),CVaR_regret=math.fsum(v*max(0.,min(1.,mass-i)) for i,v in enumerate(values))/mass)
    result['paired_LFBO_minus_GP']=[dict(seed=r['seed'],difference=r['methods']['LFBO_EI']['final_normalized_regret']-r['methods']['GP_EI']['final_normalized_regret']) for r in stats]
    return result
