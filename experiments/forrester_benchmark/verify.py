"""Read-only audit and independent statistics; no PFN weights/model calls."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from experiments.evaluate import content_hash
from experiments.forrester_benchmark.records import csv_rows,numeric_rows,verify_record,close,independent_cvar
from experiments.forrester_benchmark.protocol import ROOT,load,validate,source_identity
from experiments.forrester_benchmark.run import read_completed,result_path,rid

METHODS=('random','lfbo','pfn')
METRICS=('final_normalized_regret','final_simple_regret','final_best_raw_objective')
def stats(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Expected finite measured values')
    std=float(a.std(ddof=1)) if len(a)>1 else None
    return dict(n=len(a),mean=float(a.mean()),sample_std=std,standard_error=std/np.sqrt(len(a)) if std is not None else None,
        median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),
        p95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()))

def derive(records,p):
    summary={};paired=[];lookup={m:{r['task_id']:r for r in records[m]} for m in METHODS}
    targets=sorted(lookup['random'])
    if any(set(lookup[m])!=set(targets) for m in METHODS):raise ValueError('Cannot pair unequal task sets')
    for method in METHODS:
        summary[method]={metric:stats([r['final_metrics'][metric] for r in records[method]]) for metric in METRICS}
        summary[method]['exploratory_CVaR_0_8']=independent_cvar([r['final_metrics'][METRICS[0]] for r in records[method]])
        summary[method]['runtime_seconds']=stats([r['elapsed_seconds'] for r in records[method]])
    for target in targets:
        row=dict(task_id=target)
        for m in METHODS:row[m]=lookup[m][target]['final_metrics'][METRICS[0]]
        for a,b in [('pfn','random'),('lfbo','random'),('pfn','lfbo')]:row[a+'_minus_'+b]=row[a]-row[b]
        paired.append(row)
    comparisons={};tol=p['tie_absolute_tolerance']
    for key in ('pfn_minus_random','lfbo_minus_random','pfn_minus_lfbo'):
        d=np.array([r[key] for r in paired]);comparisons[key]=dict(**stats(d),wins=int((d < -tol).sum()),ties=int((np.abs(d)<=tol).sum()),
            losses=int((d>tol).sum()),win_fraction=float((d < -tol).mean()),tie_atol=tol,negative_favors='first_named_method',
            worst_for_first=max(paired,key=lambda r:r[key]),best_for_first=min(paired,key=lambda r:r[key]))
    return dict(algorithms=summary,paired_comparisons=comparisons,paired_rows=paired,sampling_unit='unique_target_task_one_trial',
        table1_reproduction=False,paper_equivalence_status='UNVERIFIED')

def audit(lock,root=None,allow_partial=False):
    p=validate(lock);n=p['expected']['unique_targets'];records={};details={};hashes={};run_ids=[]
    designs={d['task_id']:d for d in p['initial_design']};specs={s['task_id']:s for s in p['manifest']['target_tasks']}
    checks=[];replay_calls=0
    for method in METHODS:
        folder=result_path(lock,method,False,root);summary=json.loads((folder/'summary.json').read_text());identity=json.loads((folder/'experiment_identity.json').read_text())
        for name,value in [('protocol_lock.json',lock),('task_manifest.json',p['manifest']),('initial_design.json',p['initial_design']),('references.json',p['references'])]:
            if json.loads((folder/name).read_text())!=value:raise ValueError('Root evaluator snapshot mismatch')
        if identity['protocol_id']!=lock['protocol_id'] or identity['algorithm']!=method or identity['audit_replay'] is not False:raise ValueError('Wrong experiment identity')
        if 'historical_source_fingerprint' in identity:
            if identity['historical_source_fingerprint']!=p['source_fingerprint']:
                raise ValueError('Wrong historical source fingerprint')
            if content_hash(identity['runner_source_sha256'])!=identity['source_fingerprint']:
                raise ValueError('Wrong distributable runner fingerprint')
            if identity['runner_source_sha256']!=source_identity()['source_sha256']:
                raise ValueError('Execution was produced by different runner source')
        elif identity['source_fingerprint']!=p['source_fingerprint']:
            raise ValueError('Wrong measured source fingerprint')
        if summary['table1_reproduction'] is not False or summary['paper_equivalence_status']!='UNVERIFIED' or summary['mode']!=p['mode']:raise ValueError('Wrong claim/mode')
        dirs=list((folder/'runs').iterdir());table=csv_rows(folder/'runs.csv');obs=csv_rows(folder/'observations.csv')
        if len(dirs)!=len(table) or len(obs)!=16*len(table):raise ValueError('Wrong run/observation counts')
        if len(table)!=n and not allow_partial:raise ValueError('Incomplete cohort')
        if len({r['task_id'] for r in table})!=len(table):raise ValueError('More than one trial per target')
        if not {r['task_id'] for r in table}<=set(specs):raise ValueError('Unknown target identity')
        records[method]=[];details[method]=[];offset=0
        for row in table:
            run_id=rid(lock['protocol_id'],method,row['task_id']);run_ids.append(run_id)
            if row['run_id']!=run_id:raise ValueError('Unstable run ID')
            r=read_completed(folder,run_id,identity,p);records[method].append(r)
            details[method].append(dict(task_id=r['task_id'],**verify_record(r,specs[r['task_id']],designs[r['task_id']],p['references'][r['task_id']],method)))
            selected=obs[offset:offset+16];offset+=16
            if csv_rows(folder/'runs'/run_id/'observations.csv')!=selected:raise ValueError('Root observations differ from run')
            for key,v in r['final_metrics'].items():
                if float(row[key])!=v:raise ValueError('Root run metric mismatch')
            for key in ('seed','optimizer_seed','initialization_seed','elapsed_seconds'):
                if float(row[key])!=r[key]:raise ValueError('Root run provenance mismatch')
        if summary['completed_runs']!=len(table) or summary['online_objective_calls']!=len(obs):raise ValueError('Summary count mismatch')
        expected_status='completed' if len(table)==n else 'partial'
        if summary['status']!=expected_status:raise ValueError('Summary completion claim mismatch')
        for key in ('manifest_sha256','initial_design_sha256','references_sha256'):
            if summary[key]!=p[key]:raise ValueError('Summary component identity mismatch')
        completion=json.loads((folder/'completion.json').read_text())
        if completion!=dict(identity=identity,status=expected_status,completed_runs=len(table),online_objective_calls=len(obs)):raise ValueError('Completion marker mismatch')
        cfg=json.loads((folder/'config_snapshot.json').read_text())
        if cfg['table1_reproduction'] is not False or cfg['model_options']!=p['model_options'][method] or cfg['evaluation']!=p['evaluation']:raise ValueError('Config snapshot mismatch')
        if len(table)==n and {r['task_id'] for r in table}!=set(specs):raise ValueError('Missing full target set')
        for metric in METRICS:
            independently=stats([r['final_metrics'][metric] for r in records[method]])
            for key in ('n','mean','sample_std','standard_error'):close(independently[key],summary['metrics'][metric][key])
        close(independent_cvar([r['final_metrics'][METRICS[0]] for r in records[method]]),summary['exploratory_CVaR']['value'])
        checks.append(dict(name=method+'_raw_counts_formula_metrics_cache',status='PASS',completed_runs=len(table),online_calls=len(obs)))
        hashes[method]={str(f.relative_to(folder)):__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in folder.rglob('*') if f.is_file()}
    if len(run_ids)!=len(set(run_ids)):raise ValueError('Repeated main run ID')
    lookup={m:{r['task_id']:r for r in records[m]} for m in METHODS}
    shared=set(lookup['random'])
    if any(set(lookup[m])!=shared for m in METHODS):raise ValueError('Algorithms have different target sets')
    for target in shared:
        initialization=[[(v['x'],v['y']) for v in lookup[m][target]['observations'][:2]] for m in METHODS]
        if initialization[0]!=initialization[1] or initialization[1]!=initialization[2]:raise ValueError('Unpaired exact initial values')
        if any(lookup[m][target]['reference']!=p['references'][target] for m in METHODS):raise ValueError('Reference mismatch')
    checks.append(dict(name='exact_shared_tasks_initial_xy_references',status='PASS',unique_targets=len(shared)))
    return dict(status='PASS' if len(shared)==n else 'PARTIAL',mode=p['mode'],protocol_id=lock['protocol_id'],table1_reproduction=False,
        actual_completed_runs=sum(map(len,records.values())),actual_online_calls=16*sum(map(len,records.values())),
        actual_unique_targets=len(shared),separate_replay_online_calls=replay_calls,checks=checks,per_run=details,
        independent_statistics=derive(records,p),artifact_sha256=hashes),records

def verify_published(root=None,verify_checksums=True):
    root=Path(root) if root is not None else ROOT/'artifacts/forrester_benchmark'
    if verify_checksums:
        import hashlib
        entries=(root/'MANIFEST.sha256').read_text().splitlines()
        seen=set()
        for line in entries:
            digest,relative=line.split('  ',1);path=Path(relative)
            if path.is_absolute() or '..' in path.parts or relative in seen:
                raise ValueError('Unsafe or repeated checksum path')
            seen.add(relative)
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Published file checksum mismatch: '+relative)
        covered={str(f.relative_to(ROOT)) for f in root.rglob('*') if f.is_file() and f.name!='MANIFEST.sha256'}
        covered.update(str(f.relative_to(ROOT)) for f in (ROOT/'configs/forrester_benchmark').rglob('*') if f.is_file())
        if covered!=seen:raise ValueError('Published manifest coverage mismatch')
    lock=load()
    report,records=audit(lock,root=root/'raw')
    summary=json.loads((root/'analysis/metrics_summary.json').read_text())
    derived=report['independent_statistics']
    if summary['algorithms']!=derived['algorithms'] or summary['paired_comparisons']!=derived['paired_comparisons']:
        raise ValueError('Published analysis differs from independently recomputed raw data')
    for filename,expected in [('per_target_metrics.csv',300),('paired_differences.csv',100),('comparison.csv',3),('chart_ready_long.csv',4800)]:
        if len(csv_rows(root/'analysis'/filename))!=expected:
            raise ValueError('Published CSV count mismatch: '+filename)
    if len(list((root/'analysis/figures').glob('*.png')))!=6:
        raise ValueError('Missing published figures')
    report['published_checksums']='PASS' if verify_checksums else 'NOT_CHECKED'
    report['published_analysis']='PASS'
    return report,records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',help='Audit a new execution root instead of the published artifact')
    parser.add_argument('--report',help='Write a new JSON report outside the immutable published data')
    a=parser.parse_args(argv)
    try:
        if a.report and Path(a.report).exists():raise FileExistsError('Report exists; choose a new path')
        report,_=audit(load(),root=a.results_root) if a.results_root else verify_published()
        if a.report:
            out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('independent_statistics','artifact_sha256','per_run')},indent=2))
        return 0 if report['status']=='PASS' else 1
    except Exception as exc:
        parser.exit(1,'Verification failed: '+str(exc)+'\n')
if __name__=='__main__':raise SystemExit(main())
