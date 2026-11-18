"""Separate track tables, paired differences, failure tails and future measured plots."""
from __future__ import annotations
from .protocol import *
from .verification import verify,family
from .metrics import calibration


def render(summary,c):
    label='Matched raw-event' if summary['track']=='A' else 'EI-augmented held-out'
    lines=[f"{summary['status']} | {label if summary['evaluation']=='classifier' else 'Paired BO'}",
        f"Run={summary['run_id']} | config={summary['config_sha256']} | protocol={summary['protocol_id']}",summary['track_status'],
        c['paper_equivalence'],'GP derived evaluator is GROUP_DERIVED_EVALUATOR / NOT_AUTHOR_VERIFIED.',
        f"Independent families={summary['families']}; tasks/family={summary['tasks_per_family']}; singleton SE=N/A.",
        '| Method | Mean BCE | CVaR0.8 | SE mean | SE CVaR |' if summary['evaluation']=='classifier' else '| Method | Mean Final Normalized Regret | SE |',
        '|---|---:|---:|---:|---:|' if summary['evaluation']=='classifier' else '|---|---:|---:|']
    for method,stats in summary['methods'].items():
        if summary['evaluation']=='classifier':
            a,b=stats['mean_BCE'],stats['CVaR'];lines.append(f"| {method} | {a['mean']:.10g} | {b['mean']:.10g} | {a['SE']} | {b['SE']} |")
        else:
            a=stats['final_normalized_regret'];lines.append(f"| {method} | {a['mean']:.10g} | {a['SE']} |")
    lines += ['', 'Separate Paper Reference — not measured data; no protocol equivalence claimed.',
        '| Method | Paper Average Loss | Paper CVaR |','|---|---:|---:|']
    for method,r in c['paper_reference'].items():lines.append(f"| {method} | {r['average_loss']} | {r['CVaR']} |")
    return '\n'.join(lines)+'\n'


def export(folder):
    folder=Path(folder);summary,stats=verify(folder,reports=False);c=read(folder/'config.json');job=read(folder/'job.json')
    predictions=[];taskrows=[];paired=[];cal={};observations=[];curves={}
    for i in range(len(job['seeds'])):
        _,units,_=family(folder,i);bytask={}
        for task,r in units:
            if job['evaluation']=='classifier':
                taskrows.append(dict(family_index=i,seed=job['seeds'][i],task_id=task,method=r['method'],task_loss=r['task_loss'],rows=len(r['predictions']),positive_support=sum(v['label'] for v in r['predictions']),negative_support=sum(v['label']==0 for v in r['predictions'])))
                for row in r['predictions']:predictions.append(dict(family_index=i,seed=job['seeds'][i],task_id=task,method=r['method'],**row))
                bytask.setdefault(task,{})[r['method']]=r['task_loss']
            else:
                observations += [dict(family_index=i,seed=job['seeds'][i],method=r['method'],**row) for row in r['observations']]
                curves.setdefault(r['method'],[]).append(r['regret_curve'])
        for task,values in bytask.items():
            names=methods(c,job['track']);paired.append(dict(family_index=i,seed=job['seeds'][i],task_id=task,left_method=names[0],right_method=names[1],difference=values[names[0]]-values[names[1]]))
    save(folder/'summary.json',summary,replace=True);save(folder/'run_metrics.json',stats,replace=True)
    save(folder/'verification.json',dict(status='PASS',raw_recomputed=True,model_fits=0,objective_API_calls=0),replace=True)
    write_text(folder/'comparison.md',render(summary,c),replace=True)
    files=['summary.json','run_metrics.json','verification.json','comparison.md']
    def csv(name,rows):
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r));write_csv(folder/name,rows,fields,replace=True);files.append(name)
    csv('predictions.csv',predictions);csv('task_losses.csv',taskrows);csv('paired_task_differences.csv',paired);csv('bo_observations.csv',observations)
    if job['evaluation']=='classifier':
        for method in methods(c,job['track']):
            selected=[r for r in predictions if r['method']==method]
            cal[method]=calibration(selected,c) if job['track']=='A' else dict(status='NOT_APPLICABLE_RAW_EVENT_CALIBRATION',reason='EI weighted augmented distribution has different semantics')
        save(folder/'calibration.json',cal,replace=True);files.append('calibration.json')
        save(folder/'failure_tails.json',sorted(taskrows,key=lambda r:r['task_loss'],reverse=True)[:50],replace=True);files.append('failure_tails.json')
    files+=plots(folder,job,predictions,taskrows,paired,cal,curves,summary)
    save(folder/'report_manifest.json',{name:sha(folder/name) for name in files});verify(folder)
    print(render(summary,c),flush=True)


def plots(folder,job,predictions,tasks,paired,cal,curves,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    files=[]
    def finish(fig,name):
        fig.suptitle(job['status']+' | '+job['evaluation']+(' Track '+job['track'] if job['evaluation']=='classifier' else ''))
        fig.tight_layout();fig.savefig(folder/name,dpi=150);plt.close(fig);files.append(name)
    if job['evaluation']=='bo':
        fig,ax=plt.subplots()
        for method,values in curves.items():ax.plot(np.arange(1,len(values[0])+1),np.mean(values,axis=0),label=method)
        ax.set(xlabel='Total online evaluations (includes initialization)',ylabel='Mean normalized regret');ax.legend();finish(fig,'bo_convergence.png')
        return files
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in summary['methods']:
        values=[r['task_loss'] for r in tasks if r['method']==method];axes[0].hist(values,bins=min(30,max(3,len(values)//2)),alpha=.5,label=method)
        sorted_values=sorted(values,reverse=True);axes[1].plot(sorted_values,label=method)
    axes[0].set(xlabel='Task Weighted BCE',ylabel='Task count (descriptive)');axes[1].set(xlabel='Task rank, worst first',ylabel='Task loss')
    for ax in axes:ax.legend(fontsize=7)
    finish(fig,'task_loss_distribution_and_tails.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4));names=list(summary['methods'])
    for ax,key in zip(axes,['mean_BCE','CVaR']):
        values=[summary['methods'][m][key]['mean'] for m in names];se=[summary['methods'][m][key]['SE'] for m in names]
        ax.bar(range(len(names)),values,yerr=None if any(v is None for v in se) else se);ax.set_xticks(range(len(names)),names,rotation=15,fontsize=7);ax.set_ylabel(key)
    finish(fig,'BCE_CVaR_same_protocol.png')
    fig,ax=plt.subplots();ax.hist([r['difference'] for r in paired],bins=min(30,max(3,len(paired)//2)));ax.axvline(0,color='black');ax.set(xlabel='RF/LFBO minus GP per-task loss',ylabel='Paired task count');finish(fig,'paired_task_differences.png')
    valid={m:r for m,r in cal.items() if r['status']=='DIAGNOSTIC_NO_CALIBRATION_FIT'}
    if valid:
        fig,ax=plt.subplots();ax.plot([0,1],[0,1],'--',color='gray')
        for method,r in valid.items():
            bins=[b for b in r['bins'] if b['n']];ax.plot([b['mean_probability'] for b in bins],[b['empirical_positive'] for b in bins],marker='o',label=method)
        ax.set(xlabel='Mean predicted raw-event probability',ylabel='Empirical positive fraction');ax.legend(fontsize=7);finish(fig,'calibration_raw_event.png')
    return files


def pending(c):
    lines=['PAPER_INSPIRED_INDEPENDENT_REPRODUCTION / NOT_PAPER_EQUIVALENT',
        'No production experiment has been executed for this protocol. Fixture values are excluded.']
    for a,title in [('A','Matched Raw-Event'),('B','EI-Augmented Classifier')]:
        lines += ['',title,'| Method | Mean BCE / Weighted BCE | CVaR0.8 | SE |','|---|---:|---:|---:|']
        for method in methods(c,a):lines.append(f'| {method} | PENDING | PENDING | PENDING |')
    lines += ['','Paper reference (separate; not measured)','| Method | Paper Average Loss | Paper CVaR |','|---|---:|---:|']
    for m,r in c['paper_reference'].items():lines.append(f"| {m} | {r['average_loss']} | {r['CVaR']} |")
    return '\n'.join(lines)+'\n'
