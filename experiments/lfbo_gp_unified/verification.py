"""Read-only raw-data/metric verifier; no numpy, model, fit or objective API."""
from __future__ import annotations
import math
import functools
from pathlib import Path
import statistics
from .protocol import *
from .probability import gaussian,rf_scores
from .metrics import sample,task_loss,task_summary,aggregate


def close(a,b,message):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,rel_tol=2e-11,abs_tol=1e-12),message)


def quantile_unique(y,gamma):
    values=sorted(set(y));q=(len(values)-1)*gamma;lo=int(q)
    return values[lo]+(q-lo)*(values[min(lo+1,len(values)-1)]-values[lo])


@functools.lru_cache(maxsize=4)
def registered_splits(path,digest):
    require(sha(path)==digest,'Registered split artifact changed')
    return {(r['family_seed'],r['task_id']):r for r in read(path)['splits']}


def audit_task(t,family,c,manifest_root=ROOT):
    n=c['dataset']['observations'];raw=t['raw'];X,y=raw['X'],raw['y'];tr=t['split']['train'];va=t['split']['validation']
    require(len(X)==len(y)==n and all(len(x)==1 and math.isfinite(x[0]) and 0<=x[0]<=1 for x in X) and all(math.isfinite(v) for v in y),'Invalid raw data')
    require(len(tr)==c['dataset']['split']['train_count'] and len(va)==c['dataset']['split']['validation_count'],'Wrong split counts')
    require(len(set(tr))==len(tr) and len(set(va))==len(va) and not set(tr)&set(va) and set(tr)|set(va)==set(range(n)),'Raw indices overlap/partition')
    require(not {tuple(X[i]) for i in tr}&{tuple(X[i]) for i in va},'Raw X leakage')
    require(t['uid']>0 and t['descriptor']['uid']==t['uid'] and t['task_id']==f"meta{t['uid']:04d}",'Target/meta mix')
    if all(k in t['descriptor'] for k in ('a','b','c')):
        d=t['descriptor']
        for point,value in zip(X,y):
            x=point[0];close(value,d['a']*(6*x-2)**2*math.sin(12*x-4)+d['b']*(x-.5)-d['c'],'Source equation mismatch')
    require(t['split']['seed']==seed(family,'raw-split',t['task_id']),'Wrong split seed')
    split=c['dataset']['split'];registered=registered_splits(str(Path(manifest_root)/split['registered_manifest']),split['manifest_sha256'])[(family,t['task_id'])]
    require(registered['train']==tr and registered['validation']==va,'Partition differs from registered indices')
    require(t['streams']=={s:seed(family,s,t['task_id']) for s in c['dataset']['rng']['streams']},'Wrong RNG streams')
    tau=quantile_unique([y[i] for i in tr],c['classification']['gamma']);constants=t['constants'];close(constants['tau'],tau,'Training-only tau mismatch')
    m=sum(y[i]<tau for i in tr);U=statistics.mean(max(tau-y[i],0) for i in tr);bal=(len(tr)+m)/(2*len(tr))
    require(U>0 and 0<m<len(tr),'Degenerate training')
    close(constants['U_train'],U,'Training U mismatch');close(constants['c'],bal,'Training balance mismatch')
    close(constants['positive_coefficient'],bal/U,'Training positive coefficient');close(constants['background_weight'],bal,'Background constant')
    require(constants['normalization_source_indices']==tr and constants['n_good']==m and constants['n_train']==len(tr),'Normalization indices/counts')
    for a in ('A','B'):
        for side,indices in [('training',tr),('validation',va)]:
            good=[i for i in indices if y[i]<tau] if a=='B' else []
            expected=[dict(point_index=i,label=1,weight=bal*(tau-y[i])/U,role='positive_improvement') for i in good]
            expected += [dict(point_index=i,label=0 if a=='B' else int(y[i]<tau),weight=bal if a=='B' else 1.,role='background_negative' if a=='B' else 'raw_event') for i in indices]
            actual=t['tracks'][a][side];require(len(actual)==len(expected),'Augmented count mismatch')
            for r,e in zip(actual,expected):
                require(all(r[k]==e[k] for k in ('point_index','label','role')),'Label/order/member mismatch')
                close(r['weight'],e['weight'],'Frozen training weight mismatch')
    return True


def audit_unit(unit,t,track,c,identity):
    require(unit['identity']==identity,'Unit identity/dataset/source mismatch');r=unit['result'];rows=r['predictions']
    require(r['model_fits']==1 and r['method'] in methods(c,track),'Wrong method or invented fit')
    require(len(rows)==len(t['tracks'][track]['validation']),'Validation denominator')
    gp=r['method']==methods(c,track)[1];training=t['split']['train'] if gp else [v['point_index'] for v in t['tracks'][track]['training']]
    require(r['training_X']==[t['raw']['X'][i] for i in training],'Training coordinates differ')
    if gp:
        require(r['training_y']==[t['raw']['y'][i] for i in training] and r['model_seed']==t['streams']['gp'] and not r['include_observation_noise'],'GP training leakage/noise/seed')
        require(canonical(r['GP_options'])==canonical(c['GP']),'GP option override')
    else:
        stream='rf-event' if track=='A' else 'lfbo-ei'
        require(r['model_seed']==t['streams'][stream] and r['classes']==[0,1] and r['positive_column']==1,'RF seed/class mapping')
        expected=dict(c['RF']['parameters'],random_state=r['classifier_seed']);require(r['RF_parameters']==expected,'RF silent hyperparameter change')
        require(len(r['estimator_random_states'])==c['RF']['parameters']['n_estimators'],'RF tree count')
        require(r['training_labels']==[v['label'] for v in t['tracks'][track]['training']] and r['training_weights']==[v['weight'] for v in t['tracks'][track]['training']],'RF labels/weights changed')
    for actual,expected in zip(rows,t['tracks'][track]['validation']):
        require(all(actual[k]==expected[k] for k in ('point_index','label','role','weight')),'Validation row changed')
        rid=expected['point_index'];require(actual['x']==t['raw']['X'][rid][0] and actual['y']==t['raw']['y'][rid] and actual['tau']==t['constants']['tau'] and actual['U_train']==t['constants']['U_train'],'Prediction raw data/constants')
        if gp:
            score=gaussian(actual['posterior_mean'],actual['latent_variance'],t['constants']['tau'],t['constants']['U_train'],c)
            for key,value in score.items():close(actual[key],value,'Gaussian '+key)
            prefix='event' if track=='A' else 'augmented'
            probability=score[prefix+'_probability'];lp,lm=score[prefix+'_log_p'],score[prefix+'_log_not_p']
        else:
            score=rf_scores(actual['probability'],c['classification']['RF_epsilon']);probability=score['probability'];lp,lm=score['log_p'],score['log_not_p']
            close(actual['p_used'],score['p_used'],'RF clipping')
        close(actual['probability'],probability,'Probability semantics')
        for key,value in dict(log_p=lp,log_not_p=lm,**sample(lp,lm,expected['label'],expected['weight'])).items():close(actual[key],value,'Loss '+key)
    close(r['task_loss'],task_loss(rows),'Task row-count BCE')
    return r


def family(folder,index,complete=True):
    folder=Path(folder);pid=check_lock(folder);c=read(folder/'config.json');job=read(folder/'job.json');directory=folder/'families'/f'{index:03d}'
    dataset=unseal(directory/'dataset.json');require(dataset['protocol_id']==pid,'Dataset protocol')
    d=dataset['data'];family_seed=job['seeds'][index];require(d['family_seed']==family_seed and len(d['tasks'])==job['tasks'] and d['excluded_target_uid']==0,'Wrong family population')
    if job['evaluation']=='bo':
        from .bo import verify_family
        return verify_family(folder,index,d,c,job,pid)
    require(d['version']==('synthetic-fixture-v1' if job['fixture'] else c['dataset']['version']),'Dataset generation version')
    losses={m:[] for m in methods(c,job['track'])};units=[]
    from .checkpoints import selected,model_path,check
    for task_index,t in enumerate(d['tasks']):
        audit_task(t,family_seed,c,folder/'source_snapshot')
        for method in losses:
            path=directory/'units'/t['task_id']/(method+'.json')
            if not path.exists():
                require(not complete,'Incomplete task: '+str(path));continue
            identity=dict(protocol_id=pid,dataset_sha256=sha(directory/'dataset.json'),family_index=index,task_id=t['task_id'],track=job['track'],method=method)
            r=audit_unit(unseal(path),t,job['track'],c,identity)
            if selected(job.get('checkpoint_policy','none'),index,task_index):
                require('model_checkpoint' in r,'Expected fitted model checkpoint missing')
            if 'model_checkpoint' in r:check(model_path(directory,t['task_id'],method),r['model_checkpoint'])
            losses[method].append(r['task_loss']);units.append((t['task_id'],r))
    stats={m:task_summary(v,c['classification']['alpha']) for m,v in losses.items()}
    require(all(s['N']==job['tasks'] for s in stats.values()),'Task dropped from family')
    return dict(family_index=index,seed=family_seed,methods=stats),units,d


def verify(folder,reports=True):
    folder=Path(folder);check_lock(folder);c=read(folder/'config.json');job=read(folder/'job.json')
    require(job['seeds']==c['dataset']['seeds'][:len(job['seeds'])],'Seed replacement')
    stats=[]
    for index in range(len(job['seeds'])):
        result,_,_=family(folder,index);receipt=unseal(folder/'families'/f'{index:03d}'/'completed.json')
        require(receipt==result,'Family completed metrics changed');stats.append(result)
    names=methods(c,job['track']) if job['evaluation']=='classifier' else ['LFBO_EI','GP_EI']
    fields=['mean_BCE','CVaR'] if job['evaluation']=='classifier' else ['final_normalized_regret']
    summary=dict(run_id=job['run_id'],protocol_id=check_lock(folder),config_sha256=canonical(c),status=job['status'],evaluation=job['evaluation'],track=job['track'] if job['evaluation']=='classifier' else None,
        track_status=c['classification']['track_'+job['track']]['status'] if job['evaluation']=='classifier' else 'PAIRED_BO_DIAGNOSTIC',
        paper_equivalence=c['paper_equivalence'],families=len(stats),tasks_per_family=job['tasks'],
        methods={m:{field:aggregate([r['methods'][m][field] for r in stats]) for field in fields} for m in names})
    if job['evaluation']=='bo':
        from .bo import aggregate_bo
        summary['regret_details']=aggregate_bo(stats,c)
    if reports:
        require(read(folder/'summary.json')==summary,'Summary differs from raw')
        for name,digest in read(folder/'report_manifest.json').items():require(sha(safe_child(folder,name))==digest,'Report checksum changed: '+name)
    return summary,stats
