"""Controlled production entry point; preparation/tests never dispatch full runs."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .protocol import *


def preflight(c):
    env=environment(c);sources=source_identity(c)
    manifest=ROOT/c['dataset']['split']['registered_manifest']
    require(sha(manifest)==c['dataset']['split']['manifest_sha256'],'Registered split schedule changed')
    require(c['RF']['parameters']['n_estimators']==c['RF']['owner_options']['n_estimators']==1000,'RF1000 profile inconsistent')
    return dict(status='PASS',protocol=c['status'],config_sha256=sha(CONFIG),source_hashes=sources,environment=env,
        registered_splits=20000,split_manifest_sha256=sha(manifest),production_runs_executed=0,model_fits=0,objective_calls=0,
        GP_evaluator='GROUP_DERIVED_EVALUATOR / NOT_AUTHOR_VERIFIED',paper_equivalence=c['paper_equivalence'])


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv);p=argparse.ArgumentParser(description='Paper-inspired independent LFBO+GP protocols; no Table1 equivalence claim')
    p.add_argument('--mode',choices=['preflight','smoke','family','full','resume','verify','show','compare',
        'export-evidence','verify-evidence','replay-checkpoints'],required=True)
    p.add_argument('--track',choices=['B'],default='B')
    p.set_defaults(evaluation='classifier')
    p.add_argument('--run-id');p.add_argument('--workers',type=int,default=1);p.add_argument('--confirm-full',action='store_true')
    p.add_argument('--checkpoints',choices=['none','representative','all'],default=None,
        help='New runs default to representative: save native LFBO/GP models for first task of first family. all can use substantial disk space.')
    p.add_argument('--evidence-dir',help='New destination for export-evidence; existing directory for verify-evidence/replay-checkpoints')
    p.add_argument('--acknowledge-independent-protocol',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(args)
    try:
        c=config()
        evidence_mode=a.mode in ('export-evidence','verify-evidence','replay-checkpoints')
        if evidence_mode:
            require(a.evidence_dir,'--evidence-dir required')
            require(a.checkpoints is None,'--checkpoints is only for experiment execution')
            if a.mode=='export-evidence':
                require(a.run_id,'--run-id required')
                require(not Path(a.evidence_dir).exists(),'Evidence destination exists; choose a new directory')
        if a.mode in ('verify','show','compare'):
            require(a.run_id,'--run-id required');folder=OUTPUT/safe_id(a.run_id)
            from .historical import is_historical,show
            if is_historical(folder):return show(folder,a.mode)
            from .verification import verify
            summary,_=verify(folder)
            print('PASS: raw data/labels/weights/log-probabilities/BCE/CVaR/lineage. No models imported.' if a.mode=='verify' else (folder/'comparison.md').read_text())
            return 0
        require(1<=a.workers<=16,'workers must be 1..16')
        if a.mode=='full':require(a.confirm_full and a.acknowledge_independent_protocol,'Full requires --confirm-full --acknowledge-independent-protocol; no fitting started')
        if a.mode!='preflight' and not evidence_mode:require(a.run_id,'--run-id required')
        if a.mode=='resume':
            from .historical import is_historical
            require(not is_historical(OUTPUT/safe_id(a.run_id)),
                'Archived v1 run is read-only under portable v2. Use its original source/runtime to resume; do not change its lock.')
            folder=OUTPUT/safe_id(a.run_id);check_lock(folder,current=True);job=read(folder/'job.json')
            require(not job['fixture'],'CLI cannot promote fixture outputs to production')
            if job['mode']=='full':require(a.confirm_full and a.acknowledge_independent_protocol,'Resuming full requires original full acknowledgements')
        if not a.worker:
            runtime=ROOT/c['runtime']['executable'];require(runtime.is_file(),'Locked runtime missing')
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            return subprocess.call([str(runtime),'-B','-u','-m','experiments.lfbo_gp_unified.cli',*args,'--worker'],cwd=ROOT,env=env)
        if evidence_mode:
            from .evidence import export_evidence,verify_evidence,replay_checkpoints
            if a.mode=='export-evidence':result=export_evidence(OUTPUT/safe_id(a.run_id),a.evidence_dir)
            elif a.mode=='verify-evidence':result=verify_evidence(a.evidence_dir)
            else:result=replay_checkpoints(a.evidence_dir)
            print(json.dumps(result,indent=2));return 0
        result=preflight(c)
        if a.mode=='preflight':print(json.dumps(result,indent=2));return 0
        from .runner import run
        folder=OUTPUT/safe_id(a.run_id)
        if a.mode=='resume':
            job=read(folder/'job.json');return run(folder,job['mode'],job['track'],job['evaluation'],c,a.workers,resume=True,checkpoint_policy=a.checkpoints) or 0
        run(folder,a.mode,a.track,a.evaluation,c,a.workers,checkpoint_policy=a.checkpoints or 'representative');return 0
    except (Exception,KeyboardInterrupt) as exc:
        print('Unified protocol error: '+str(exc),file=sys.stderr);return 130 if isinstance(exc,KeyboardInterrupt) else 1


if __name__=='__main__':raise SystemExit(main())
