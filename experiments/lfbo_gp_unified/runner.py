"""Checkpointed execution. CLI production runs are separate from test fixtures."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import multiprocessing
from pathlib import Path
import traceback
import uuid
from .protocol import *
from .verification import family,verify,audit_unit


def fixture_config(c):
    import copy
    c=copy.deepcopy(c);f=c['fixture'];c['RF']['parameters']['n_estimators']=f['RF_trees'];c['RF']['owner_options']['n_estimators']=f['RF_trees']
    c['GP']['optimizer']=f['GP_optimizer'];c['BO'].update(budget=f['BO_budget'],candidates=f['BO_candidates'],reference_grid=f['BO_reference_grid'])
    return c


def fixture_data(family_seed,tasks,c):
    import numpy as np
    from .data import prepare_task
    values=[]
    for i in range(1,tasks+1):
        X=np.linspace(.01,.99,32);y=(X-.3)**2+.1*np.sin(5*X)+i*.01
        spec=dict(uid=i,task_id=f'meta{i:04d}',fixture_function='quadratic_plus_sine',offset=i*.01)
        values.append(prepare_task(spec,dict(X=X[:,None].tolist(),y=y.tolist()),family_seed,c))
    return dict(version='synthetic-fixture-v1',family_seed=family_seed,tasks=values,excluded_target_uid=0,objective_observations=0)


def initialize(folder,mode,track,evaluation,c,fixture=False,checkpoint_policy='none'):
    env=environment(c);folder=Path(folder)
    require(mode in ('smoke','family','full'),'Execution mode')
    job=dict(mode=mode,track=track,evaluation=evaluation,seeds=c['dataset']['seeds'] if mode=='full' else [c['dataset']['base_seed']],
        tasks=1 if evaluation=='bo' else (c['dataset']['smoke_tasks'] if mode=='smoke' else c['dataset']['tasks']),
        fixture=fixture,status=c['fixture']['status'] if fixture else c['status'],run_id=folder.name,
        checkpoint_policy=checkpoint_policy)
    if fixture:job['tasks']=1 if evaluation=='bo' else c['fixture']['tasks'];job['seeds']=c['fixture']['seeds']
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'config.json',c);save(folder/'job.json',job);save(folder/'environment.json',env)
    save(folder/'registration.json',dict(config_sha256=canonical(c),fixed_seeds=job['seeds'],dataset_policy='fresh deterministic family manifests sealed before first fit',model_fits=0,objective_calls=0))
    lock(folder,job,c)
    return job


def emit(directory,row):
    line=json.dumps(row,allow_nan=False);print(line,flush=True)
    with (directory/'terminal.jsonl').open('a') as f:f.write(line+'\n');f.flush();os.fsync(f.fileno())


def execute_family(folder_text,index):
    from threadpoolctl import threadpool_limits
    folder=Path(folder_text);pid=check_lock(folder,current=True);c=read(folder/'config.json');job=read(folder/'job.json');environment(c)
    directory=folder/'families'/f'{index:03d}';directory.mkdir(parents=True,exist_ok=True);family_seed=job['seeds'][index]
    if (directory/'completed.json').exists():family(folder,index);return dict(index=index,cached=True,new_fits=0)
    path=directory/'dataset.json'
    if not path.exists():
        if job['evaluation']=='bo':
            from .bo import prepare
            data=prepare(family_seed,c,job['fixture'])
        elif job['fixture']:data=fixture_data(family_seed,job['tasks'],c)
        else:
            from .data import family_manifest
            data=family_manifest(family_seed,job['tasks'],c)
        seal(path,dict(protocol_id=pid,data=data))
    dataset=unseal(path);require(dataset['protocol_id']==pid,'Dataset source mismatch');data=dataset['data'];new=0
    with threadpool_limits(limits=1):
        if job['evaluation']=='bo':
            from .bo import execute
            for method in ('LFBO_EI','GP_EI'):
                p=directory/'units'/(method+'.json')
                if p.exists():continue
                result=execute(data['tasks'][0],family_seed,method,c)
                seal(p,dict(identity=dict(protocol_id=pid,dataset_sha256=sha(path),family_index=index,method=method),result=result));new+=result['model_fits']
        else:
            from .lfbo_runner import fit as fit_rf
            from .gp_runner import fit as fit_gp
            from .checkpoints import selected,model_path
            for task_index,t in enumerate(data['tasks']):
                for method,fit in zip(methods(c,job['track']),(fit_rf,fit_gp)):
                    p=directory/'units'/t['task_id']/(method+'.json');identity=dict(protocol_id=pid,dataset_sha256=sha(path),family_index=index,task_id=t['task_id'],track=job['track'],method=method)
                    if p.exists():audit_unit(unseal(p),t,job['track'],c,identity);continue
                    options={}
                    if selected(job.get('checkpoint_policy','none'),index,task_index):
                        options['checkpoint_path']=model_path(directory,t['task_id'],method)
                    result=fit(t,job['track'],c,**options);seal(p,dict(identity=identity,result=result));new+=1
                emit(directory,dict(event='TASK_COMPLETE',seed=family_seed,task=t['task_id'],new_fits=new))
    stats,_,_=family(folder,index);seal(directory/'completed.json',stats)
    emit(directory,dict(event='FAMILY_COMPLETE',metrics=stats,new_fits=new));return dict(index=index,cached=False,new_fits=new)


def guarded_family(folder,index):
    try:return execute_family(str(folder),index)
    except BaseException as exc:
        save(Path(folder)/'failures'/(uuid.uuid4().hex+'.json'),dict(family_index=index,error=str(exc),traceback=traceback.format_exc(),status='FAILED_NOT_REPLACED'))
        raise


def run(folder,mode,track,evaluation,c,workers=1,resume=False,fixture=False,checkpoint_policy=None):
    folder=Path(folder)
    require(checkpoint_policy in (None,'none','representative','all'),'Unknown checkpoint policy')
    if folder.exists():
        require(resume,'Run exists; choose new ID or resume');check_lock(folder,current=True)
        require(canonical(read(folder/'config.json'))==canonical(c),'Resume config differs')
        job=read(folder/'job.json');require((job['track'],job['evaluation'],job['fixture'])==(track,evaluation,fixture),'Resume track/evaluation/profile differs')
        require(checkpoint_policy is None or checkpoint_policy==job.get('checkpoint_policy','none'),'Resume checkpoint policy differs')
        require(read(folder/'environment.json')==environment(c),'Resume runtime differs')
    else:
        require(not resume,'Missing run to resume');job=initialize(folder,mode,track,evaluation,c,fixture,checkpoint_policy or 'none')
    with (folder/'.execution.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'report_manifest.json').exists():verify(folder);print('COMPLETE CACHE: new_fits=0, new_objective_calls=0');return
        if workers==1:
            for index in range(len(job['seeds'])):guarded_family(folder,index)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(guarded_family,str(folder),i) for i in range(len(job['seeds']))]
                for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
        from .reporting import export
        export(folder)
