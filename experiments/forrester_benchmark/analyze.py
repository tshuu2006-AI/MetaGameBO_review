"""Analyze verified measured data only; six scientific figures, no BO calls."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.forrester_benchmark.protocol import ROOT, load
from experiments.forrester_benchmark.verify import audit, stats, METHODS, METRICS
from experiments.forrester_benchmark.run import result_path

def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    with Path(path).open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

LABEL={'random':'Random','lfbo':'LFBO','pfn':'PFN'};COLOR={'random':'#687986','lfbo':'#ce572d','pfn':'#2369ad'}
def csvout(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def analyze(output):
    source=ROOT/'artifacts/forrester_benchmark'
    lock=load();p=lock['payload']
    verification,records=audit(lock,root=source/'raw')
    if verification['status']!='PASS':raise ValueError('Analyze only a fully verified cohort')
    folder=Path(output).resolve()
    protected=(ROOT/'artifacts').resolve()
    if folder==protected or protected in folder.parents:
        raise ValueError('Write regenerated analysis outside the immutable published artifacts')
    folder.mkdir(parents=True,exist_ok=False)
    figures=folder/'figures';figures.mkdir()
    independent=verification['independent_statistics'];specs={s['task_id']:s for s in p['manifest']['target_tasks']}
    n=len(specs);raw=[];long=[];comparisons=[];runtime={};convergence=[]
    for method in METHODS:
        location=result_path(lock,method,root=source/'raw');cache=[read(f) for f in location.glob('runs/*/cache.json')]
        constructors={r['result']['task_id']:r['constructor_seconds'] for r in cache}
        diagnostics={r['task_id']:r['optimizer_diagnostics'] for r in records[method]}
        command=read(source/'logs'/f'{method}_measured.command.json')
        rinfo=dict(CLI_wall_seconds=command['wall_seconds'],peak_RSS_KiB=command['peak_rss_kib'],
            BO_loop_seconds=stats([r['elapsed_seconds'] for r in records[method]]),constructor_seconds=stats(list(constructors.values())),
            BO_loop_total_seconds=sum(r['elapsed_seconds'] for r in records[method]),constructor_total_seconds=sum(constructors.values()),
            failed_attempts=len(list(location.glob('failures/*.failed.json'))),
            stderr_bytes=(source/'logs'/f'{method}_measured.stderr').stat().st_size)
        if method=='lfbo':
            d=list(diagnostics.values());rinfo.update(fit_count=sum(x['fit_count'] for x in d),
                RF_fit_seconds=sum(x['fit_seconds'] for x in d),acquisition_events=sum(len(x['acquisition_events']) for x in d),
                acquisition_seconds=sum(x['acquisition_seconds'] for x in d),fallback_events=sum(len(x['fallback_events']) for x in d),
                sampled_candidates=sum(e['n_sampled'] for x in d for e in x['acquisition_events']),
                usable_candidates=sum(e['n_usable'] for x in d for e in x['acquisition_events']),
                initial_one_observation_fit_skips=sum(len(x['fit_skips']) for x in d))
        if method=='pfn':
            d=list(diagnostics.values());rinfo.update(real_forward_calls=sum(x['model_forward_calls'] for x in d),
                inference_events=sum(len(x['events']) for x in d),inference_seconds=sum(x['inference_seconds'] for x in d),
                model_load_seconds=sum(x['backend']['load_seconds'] for x in d),gradient_updates=sum(x['gradient_updates'] for x in d),
                scored_candidates=sum(e['n_usable'] for x in d for e in x['events']),checkpoint_sha256=p['checkpoint']['sha256'])
        runtime[method]=rinfo
        for r in records[method]:
            task=specs[r['task_id']]
            raw.append(dict(algorithm=method,task_id=r['task_id'],source_author_seed=task['source_author_seed'],trial_seed=0,
                a=task['a'],b=task['b'],c=task['c'],**r['final_metrics'],BO_loop_seconds=r['elapsed_seconds'],constructor_seconds=constructors[r['task_id']]))
            for o in r['observations']:long.append(dict(algorithm=method,task_id=r['task_id'],source_author_seed=task['source_author_seed'],trial_seed=0,
                query_index=o['query_index'],step=o['step'],phase=o['phase'],x=o['x'],y=o['y'],normalized_regret=o['normalized_regret'],
                simple_regret=o['simple_regret'],best_so_far=o['best_so_far'],elapsed_seconds=o['elapsed_seconds']))
        data=independent['algorithms'][method];nr=data[METRICS[0]];sr=data[METRICS[1]];br=data[METRICS[2]]
        comparisons.append(dict(algorithm=method,n_targets=n,mean_final_normalized_regret=nr['mean'],sample_std=nr['sample_std'],standard_error=nr['standard_error'],
            median=nr['median'],q25=nr['q25'],q75=nr['q75'],p90=nr['p90'],p95=nr['p95'],worst=nr['max'],exploratory_CVaR_alpha_0_8=data['exploratory_CVaR_0_8'],
            mean_final_simple_regret=sr['mean'],SE_final_simple_regret=sr['standard_error'],mean_final_best_raw_objective=br['mean'],SE_final_best_raw_objective=br['standard_error'],
            mean_BO_loop_seconds=rinfo['BO_loop_seconds']['mean'],mean_constructor_seconds=rinfo['constructor_seconds']['mean'],CLI_wall_seconds=rinfo['CLI_wall_seconds'],peak_RSS_KiB=rinfo['peak_RSS_KiB']))
        for step in range(1,17):
            for field in ('normalized_regret','simple_regret','best_so_far'):
                values=[r['observations'][step-1][field] for r in records[method]]
                convergence.append(dict(algorithm=method,step=step,metric=field,**stats(values)))
    csvout(folder/'per_target_metrics.csv',raw);csvout(folder/'comparison.csv',comparisons)
    csvout(folder/'paired_differences.csv',independent['paired_rows']);csvout(folder/'chart_ready_long.csv',long);csvout(folder/'convergence_summary.csv',convergence)
    dump(folder/'runtime_diagnostics.json',runtime)
    summary=dict(status='PASS',mode='full',protocol_id=lock['protocol_id'],cohort_id=p['cohort_id'],unique_targets=n,
        actual_runs=verification['actual_completed_runs'],actual_online_calls=verification['actual_online_calls'],
        table1_reproduction=False,paper_equivalence_status='UNVERIFIED',sampling_unit='unique_target_task_one_trial',
        algorithms=independent['algorithms'],paired_comparisons=independent['paired_comparisons'],runtime=runtime,
        exploratory_CVaR=dict(alpha=.8,input='final normalized regret',tail_mass=n*.2,estimator='fractional empirical upper tail',paper_equivalence=False),
        uncertainty='mean ±1 sample SE, ddof1, sqrt(n targets); descriptive band, not a95%CI',
        auxiliary_evaluations=p['auxiliary_evaluations'],optional_AUC=False,optional_bootstrap=False)
    dump(folder/'metrics_summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.savefig(figures/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.5))
    for m in METHODS:
        series=[r for r in convergence if r['algorithm']==m and r['metric']=='normalized_regret']
        x=np.array([r['step'] for r in series]);y=np.array([r['mean'] for r in series]);se=np.array([r['standard_error'] for r in series])
        ax.plot(x,y,marker='o',ms=3,color=COLOR[m],label=LABEL[m]);ax.fill_between(x,y-se,y+se,color=COLOR[m],alpha=.16)
    ax.axvline(2.5,color='grey',ls=':',label='2 shared initial calls');ax.set_xticks(range(1,17));ax.grid(alpha=.2)
    ax.set(title=f'Forrester group benchmark: {n} unique targets, mean ±1 SE',xlabel='Online objective evaluations (initialization included)',ylabel='Normalized regret ↓')
    ax.legend(fontsize=9);fig.text(.5,.015,'Fixed per-target grid4097 + refinement reference; group settings, not exact Table1.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'convergence_normalized_regret.png')
    fig,ax=plt.subplots(figsize=(8,5))
    for m in METHODS:
        values=sorted(r['final_metrics']['final_normalized_regret'] for r in records[m])
        ax.step(values,np.arange(1,n+1)/n,where='post',label=LABEL[m],color=COLOR[m])
    ax.set(title=f'Final regret ECDF across {n} unique targets',xlabel='Final normalized regret ↓',ylabel='Fraction of targets at or below x');ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,'final_regret_distribution.png')
    fig,ax=plt.subplots(figsize=(7.5,5))
    values=[independent['algorithms'][m][METRICS[0]] for m in METHODS]
    ax.bar(range(3),[s['mean'] for s in values],yerr=[s['standard_error'] for s in values],capsize=5,color=[COLOR[m] for m in METHODS],width=.6)
    for i,s in enumerate(values):ax.text(i,s['mean']+s['standard_error'],f" {s['mean']:.6f} ± {s['standard_error']:.6f}",ha='center',va='bottom',fontsize=9)
    ax.set_xticks(range(3),[LABEL[m] for m in METHODS]);ax.set(ylabel='Final normalized regret ↓',title=f'Mean ±1 SE over {n} target tasks')
    ax.set_ylim(0,max(s['mean']+s['standard_error'] for s in values)*1.3);ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'final_regret_bar_mean_se.png')
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for ax,key in zip(axes,('pfn_minus_random','lfbo_minus_random','pfn_minus_lfbo')):
        values=[r[key] for r in independent['paired_rows']];s=independent['paired_comparisons'][key]
        ax.scatter(range(n),values,s=18,c=['#2369ad' if v < -p['tie_absolute_tolerance'] else '#ce572d' if v>p['tie_absolute_tolerance'] else 'grey' for v in values])
        ax.axhline(0,color='black',lw=1);ax.grid(axis='y',alpha=.2)
        ax.set(title=key.replace('_minus_',' − ').upper()+f"; mean {s['mean']:.5g} ±SE{s['standard_error']:.3g}; W/T/L {s['wins']}/{s['ties']}/{s['losses']}",ylabel='A − B')
    axes[-1].set_xlabel('Target index in author seed order; negative favors A');fig.suptitle(f'Paired final-regret differences; {n} paired target units');fig.tight_layout(rect=(0,0,1,.96));save(fig,'paired_differences.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    axes[0].boxplot([[r['elapsed_seconds'] for r in records[m]] for m in METHODS],tick_labels=[LABEL[m] for m in METHODS],showfliers=True)
    axes[0].set_yscale('log');axes[0].set(title='Per-run measured BO loop (log scale)',ylabel='Seconds; excludes PFN constructor/reference')
    axes[1].bar(range(3),[runtime[m]['CLI_wall_seconds'] for m in METHODS],color=[COLOR[m] for m in METHODS]);axes[1].set_xticks(range(3),[LABEL[m] for m in METHODS])
    axes[1].set(title=f'Whole CLI for {n} runs / method',ylabel='Seconds incl. imports/setup/commits/audits');axes[0].grid(axis='y',alpha=.2);axes[1].grid(axis='y',alpha=.2)
    fig.text(.5,.01,'Different Python/library environments; CPU sequential; equal objective budget does not imply equal compute.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,'runtime_comparison.png')
    ranked=sorted(independent['paired_rows'],key=lambda r:r['pfn_minus_random'])
    indices=sorted(set([0,1,max(0,n//2-1),n//2,max(0,n-2),n-1]))
    chosen=[ranked[i] for i in indices];case_rows=[]
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    lookup={m:{r['task_id']:r for r in records[m]} for m in METHODS}
    for ax,row,index in zip(axes.flat,chosen,indices):
        task=row['task_id'];seed=specs[task]['source_author_seed']
        for m in METHODS:ax.plot(range(1,17),[o['normalized_regret'] for o in lookup[m][task]['observations']],color=COLOR[m],label=LABEL[m])
        ax.set(title=f"Author seed{seed}; PFN−Random={row['pfn_minus_random']:.4g}",xlabel='Total calls',ylabel='Normalized regret');ax.grid(alpha=.2)
        case_rows.append(dict(task_id=task,source_author_seed=seed,selection='posthoc illustrative paired-delta order statistic',rank_zero_based=index,**{k:v for k,v in row.items() if k!='task_id'}))
    for ax in list(axes.flat)[len(chosen):]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Illustrative cases: two extremes on each side + middle paired PFN−Random deltas\nAll targets retained in primary statistics; case selection is posthoc descriptive',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'per_target_cases.png');csvout(folder/'case_selection.csv',case_rows)
    return summary,folder

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='New directory outside artifacts/, e.g. results/forrester_analysis')
    a=parser.parse_args()
    summary,folder=analyze(a.output_dir)
    print(json.dumps(dict(status='PASS',analysis_directory=str(folder),actual_runs=summary['actual_runs'],
        actual_online_calls=summary['actual_online_calls'],protocol_id=summary['protocol_id']),indent=2))
if __name__=='__main__':main()
