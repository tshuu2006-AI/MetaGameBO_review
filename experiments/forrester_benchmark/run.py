"""Run the paired 100-target Forrester benchmark with atomic caches and resume."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,traceback,uuid
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.evaluate import (OBSERVATION_COLUMNS,RUN_COLUMNS,_write_csv,content_hash,environment_metadata,run_single_task,write_json)
from experiments.metrics import aggregate_runs,empirical_cvar
from experiments.forrester_benchmark.records import verify_record,csv_rows
from experiments.forrester_benchmark.objective import ROOT,AuthorTarget
from experiments.forrester_benchmark.protocol import load,load_config,validate,checkpoint_identity,environment_identity,source_identity
from models.base_model import validate_points
from models.random_search import RandomSearch
from models.lfbo import LFBO
from models.pfn import PFNOptimizer

OPTIMIZERS={'random':RandomSearch,'lfbo':LFBO,'pfn':PFNOptimizer}
def csv_exact(path,rows,fields):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if Path(path).read_bytes()!=stream.getvalue().encode():raise ValueError('Root CSV differs from committed caches: '+str(path))
def rid(protocol,algorithm,task_id,seed=0):return content_hash([protocol,algorithm,task_id,seed])[:24]
def result_path(lock,method,audit=False,root=None):
    base=Path(root) if root is not None else ROOT/('results/forrester_benchmark_replays' if audit else 'results/forrester_benchmark')
    return base/lock['protocol_id']/method

class CountedTarget:
    def __init__(self,spec,trace_path=None,budget=16):
        self._target=AuthorTarget(spec);self.task_id=spec['task_id'];self.calls=0;self.budget=budget;self.trace_path=trace_path
    def evaluate(self,X):
        points=validate_points(X,np.array([[0.,1.]]))
        if self.calls+len(points)>self.budget:raise RuntimeError('Online objective budget exceeded')
        values=self._target.evaluate(points)
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite objective')
        for x,y in zip(points[:,0],values):
            self.calls+=1
            if self.trace_path is not None:
                with Path(self.trace_path).open('a') as f:
                    f.write(json.dumps(dict(online_call_index=self.calls,x=float(x),y=float(y),utc=datetime.now(timezone.utc).isoformat()),allow_nan=False)+'\n')
                    f.flush();os.fsync(f.fileno())
        return values

def read_completed(folder,run_id,identity,p):
    one=Path(folder)/'runs'/run_id;payload=json.loads((one/'cache.json').read_text())
    digest=payload.pop('content_sha256',None)
    if digest!=content_hash(payload) or payload.get('status')!='completed' or payload.get('identity')!=identity:raise ValueError('Corrupt/mismatched completed cache')
    if payload['online_objective_calls']!=16:raise ValueError('Wrong cached actual online count')
    expected_files=set(payload['artifact_sha256'])|{'cache.json'}
    if {f.name for f in one.iterdir() if f.is_file()}!=expected_files:raise ValueError('Missing/unexpected completed artifact')
    for name,h in payload['artifact_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((one/name).read_bytes()).hexdigest()!=h:raise ValueError('Modified completed artifact: '+name)
    r=payload['result']
    if r['run_id']!=run_id or json.loads((one/'measured_result.json').read_text())!=r:raise ValueError('Result/cache identity mismatch')
    if json.loads((one/'optimizer_diagnostics.json').read_text())!=r['optimizer_diagnostics']:raise ValueError('Diagnostics/cache mismatch')
    spec=next(s for s in p['manifest']['target_tasks'] if s['task_id']==r['task_id'])
    design=next(d for d in p['initial_design'] if d['task_id']==r['task_id'])
    verify_record(r,spec,design,p['references'][r['task_id']],identity['algorithm'])
    trace=[json.loads(line) for line in (one/'online_calls.jsonl').read_text().splitlines()]
    if len(trace)!=16 or [v['online_call_index'] for v in trace]!=list(range(1,17)):raise ValueError('Actual query trace mismatch')
    if [(v['x'],v['y']) for v in trace]!=[(v['x'],v['y']) for v in r['observations']]:raise ValueError('Actual calls differ from observations')
    csv=csv_rows(one/'observations.csv')
    if len(csv)!=16:raise ValueError('Per-run CSV count mismatch')
    for c,j in zip(csv,r['observations']):
        for k,v in j.items():
            if isinstance(v,(int,float)):
                if float(c[k])!=v:raise ValueError('CSV/cache numeric mismatch')
            elif c[k]!=v:raise ValueError('CSV/cache field mismatch')
    return r

def summarize(results,p,method,protocol_id,audit=False):
    return dict(status='completed' if audit or len(results)==p['expected']['unique_targets'] else 'partial',
        cohort_id=p['cohort_id'],mode=p['mode'],protocol_id=protocol_id,algorithm=method,
        protocol_label=p['protocol_label'],claim_scope=p['claim_scope'],table1_reproduction=False,paper_equivalence_status='UNVERIFIED',
        audit_replay=audit,expected_runs=1 if audit else p['expected']['runs_per_algorithm'],completed_runs=len(results),
        unique_targets=len({r['task_id'] for r in results}),online_objective_calls=sum(len(r['observations']) for r in results),
        sampling_unit='unique_target_task_one_trial',manifest_sha256=p['manifest_sha256'],initial_design_sha256=p['initial_design_sha256'],
        references_sha256=p['references_sha256'],budget=16,n_init=2,
        metrics={m:aggregate_runs([r['final_metrics'][m] for r in results]) for m in ('final_normalized_regret','final_simple_regret','final_best_raw_objective')},
        exploratory_CVaR=dict(alpha=.8,value=empirical_cvar([r['final_metrics']['final_normalized_regret'] for r in results],.8),
            input='final_normalized_regret',sampling_unit='unique_target',estimator='fractional_empirical_upper_tail',paper_equivalence=False),
        runtime=aggregate_runs([r['elapsed_seconds'] for r in results]))

def run(lock,method,mode,*,resume=False,audit=False,output_root=None,optimizer_factory=None):
    p=validate(lock,mode)
    if method not in OPTIMIZERS:raise ValueError('Unsupported algorithm')
    folder=result_path(lock,method,audit,output_root).resolve()
    published=(ROOT/'artifacts').resolve()
    if folder==published or published in folder.parents:
        raise ValueError('New executions must be outside published artifacts')
    env=environment_metadata()
    current=source_identity();env.update(current)
    if environment_identity(env)!=p['environments']['pfn' if method=='pfn' else 'standard']:raise ValueError('Runtime environment differs from lock')
    if method=='pfn':
        from models.pfn_backend import checkpoint_metadata
        actual=checkpoint_metadata()
        if checkpoint_identity(actual)!=p['checkpoint']:raise ValueError('PFN checkpoint substitution refused')
        env['resolved_checkpoint']=actual
    identity=dict(protocol_id=lock['protocol_id'],mode=mode,algorithm=method,cohort_id=p['cohort_id'],
        manifest_sha256=p['manifest_sha256'],initial_design_sha256=p['initial_design_sha256'],references_sha256=p['references_sha256'],
        source_fingerprint=current['source_fingerprint'],historical_source_fingerprint=p['source_fingerprint'],
        runner_source_sha256=current['source_sha256'],environment=environment_identity(env),audit_replay=audit,
        checkpoint_sha256=p['checkpoint']['sha256'] if method=='pfn' else None)
    snapshots={'experiment_identity.json':identity,'protocol_lock.json':lock,'task_manifest.json':p['manifest'],
        'initial_design.json':p['initial_design'],'references.json':p['references'],
        'config_snapshot.json':dict(mode=mode,algorithm=method,model_options=p['model_options'][method],evaluation=p['evaluation'],
            protocol_label=p['protocol_label'],table1_reproduction=False,paper_equivalence_status='UNVERIFIED')}
    if folder.exists():
        if not resume:raise FileExistsError('Output exists; explicitly use --resume')
        for name,value in snapshots.items():
            if json.loads((folder/name).read_text())!=value:raise ValueError('Root snapshot changed: '+name)
    else:
        folder.mkdir(parents=True)
        for name,value in snapshots.items():write_json(folder/name,value)
        write_json(folder/'environment.json',env)
    (folder/'runs').mkdir(exist_ok=True);(folder/'.staging').mkdir(exist_ok=True)
    design=p['initial_design'][:1] if audit else p['initial_design'];results=[];calls=0;skipped=0
    factory=OPTIMIZERS[method] if optimizer_factory is None else optimizer_factory
    specs={s['task_id']:s for s in p['manifest']['target_tasks']}
    expected_ids={rid(lock['protocol_id'],method,d['task_id']) for d in design}
    if {f.name for f in (folder/'runs').iterdir() if f.is_dir()}-expected_ids:raise ValueError('Unexpected completed run identity')
    for d in design:
        run_id=rid(lock['protocol_id'],method,d['task_id'])
        if (folder/'runs'/run_id).exists():
            results.append(read_completed(folder,run_id,identity,p));skipped+=1;continue
        attempt=run_id+'-'+uuid.uuid4().hex;staging=folder/'.staging'/attempt;staging.mkdir()
        target=None
        try:
            write_json(staging/'attempt.json',dict(run_id=run_id,identity=identity,task_id=d['task_id'],trial_seed=0,
                started_at_utc=datetime.now(timezone.utc).isoformat(),initializations_are_online=True))
            target=CountedTarget(specs[d['task_id']],staging/'online_calls.jsonl')
            start=perf_counter()
            # The only constructor kwargs: public bounds, derived seed, model options.
            optimizer=factory(bounds=np.array([[0.,1.]]),seed=d['optimizer_seed'],**p['model_options'][method])
            constructor_seconds=perf_counter()-start
            measured=run_single_task(optimizer,target,16,2,0,dict(run_id=run_id,algorithm=method,optimizer_seed=d['optimizer_seed']),
                initial_points=np.asarray(d['initial_X']),reference=p['references'][d['task_id']],regret_tolerance=1e-8)
            result=asdict(measured)
            if target.calls!=16:raise RuntimeError('Incorrect actual online evaluation count')
            verify_record(result,specs[d['task_id']],d,p['references'][d['task_id']],method)
            _write_csv(staging/'observations.csv',measured.observations,OBSERVATION_COLUMNS)
            write_json(staging/'measured_result.json',result)
            write_json(staging/'optimizer_diagnostics.json',measured.optimizer_diagnostics)
            payload=dict(status='completed',identity=identity,result=result,online_objective_calls=target.calls,
                constructor_seconds=constructor_seconds,source_fingerprint=current['source_fingerprint'],
                artifact_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in staging.iterdir() if f.is_file()})
            write_json(staging/'cache.json',dict(payload,content_sha256=content_hash(payload)))
            # Single atomic directory commit; incomplete attempts never look complete.
            for file in staging.iterdir():
                with file.open('rb') as f:os.fsync(f.fileno())
            descriptor=os.open(staging,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
            staging.replace(folder/'runs'/run_id)
            descriptor=os.open(folder/'runs',os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
            results.append(read_completed(folder,run_id,identity,p));calls+=target.calls
            progress=dict(status='completed',run_id=run_id,algorithm=method,completed_runs=len(results),expected_runs=len(design),
                online_calls_in_this_run=16,task_id=d['task_id'],completed_at_utc=measured.completed_at_utc)
            with (folder/'progress.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True)
        except BaseException as exc:
            failure=dict(status='failed',run_id=run_id,identity=identity,task_id=d['task_id'],trial_seed=0,
                online_calls_before_failure=target.calls if target else 0,error_type=type(exc).__name__,error=str(exc),
                traceback=traceback.format_exc(),attempt_path=str(staging.relative_to(folder)),failed_at_utc=datetime.now(timezone.utc).isoformat())
            failed=folder/'failures';failed.mkdir(exist_ok=True);write_json(failed/(attempt+'.failed.json'),failure)
            raise
    summary=summarize(results,p,method,lock['protocol_id'],audit)
    completion=dict(identity=identity,status=summary['status'],completed_runs=len(results),online_objective_calls=16*len(results))
    done=folder/'completion.json'
    observations=[row for r in results for row in r['observations']]
    runs=[dict(run_id=r['run_id'],task_id=r['task_id'],seed=r['seed'],optimizer_seed=r['optimizer_seed'],
            initialization_seed=r['initialization_seed'],algorithm=method,**r['final_metrics'],elapsed_seconds=r['elapsed_seconds'],wandb_url=None) for r in results]
    if calls or not done.exists() or json.loads(done.read_text())!=completion:
        _write_csv(folder/'observations.csv',observations,OBSERVATION_COLUMNS)
        _write_csv(folder/'runs.csv',runs,RUN_COLUMNS)
        write_json(folder/'summary.json',summary);write_json(done,completion)
    else:
        if json.loads((folder/'summary.json').read_text())!=summary:raise ValueError('Completed summary has changed')
        csv_exact(folder/'observations.csv',observations,OBSERVATION_COLUMNS);csv_exact(folder/'runs.csv',runs,RUN_COLUMNS)
    return dict(output_directory=str(folder),summary=summary,new_online_objective_calls=calls,skipped_completed_runs=skipped)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='configs/forrester_benchmark/{random,lfbo,pfn}.yaml')
    parser.add_argument('--output-root',help='New output parent; defaults to results/forrester_benchmark')
    parser.add_argument('--dry-run',action='store_true',help='Validate the measured design without online model calls')
    parser.add_argument('--resume',action='store_true',help='Validate and skip already committed runs')
    parser.add_argument('--replay',action='store_true',help='Execute only the first target, separate from 100-target results')
    a=parser.parse_args(argv)
    try:
        config,lock=load_config(a.config)
        if a.dry_run:
            print(json.dumps(dict(status='VALIDATED_ONLY',protocol_id=lock['protocol_id'],algorithm=config['algorithm'],
                expected=lock['payload']['expected'],new_online_objective_calls=0,table1_reproduction=False)))
            return 0
        output=Path(a.output_root) if a.output_root else ROOT/('results/forrester_benchmark_replays' if a.replay else config['output_root'])
        result=run(lock,config['algorithm'],'full',resume=a.resume,audit=a.replay,output_root=output)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
        return 0
    except (ValueError,OSError,RuntimeError,AssertionError) as exc:
        parser.exit(1,'Forrester benchmark failed: '+str(exc)+'\n')
if __name__=='__main__':raise SystemExit(main())
