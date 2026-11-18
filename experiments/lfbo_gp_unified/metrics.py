"""Task row-count loss, literal task CVaR, family SE and calibration diagnostics."""
import math
import statistics
from .protocol import require


def sample(log_p,log_not_p,label,weight):
    require(label in (0,1) and weight>=0 and all(math.isfinite(v) for v in (log_p,log_not_p,weight)),'Invalid binary loss terms')
    require(log_p<=1e-12 and log_not_p<=1e-12,'Invalid log probabilities')
    bce=-(log_p if label else log_not_p)
    return dict(logit=log_p-log_not_p,BCE=bce,weighted_BCE=weight*bce)


def task_loss(rows):
    require(bool(rows),'Empty task must fail, not disappear from CVaR')
    return statistics.mean(r['weighted_BCE'] for r in rows)


def task_summary(losses,alpha):
    require(bool(losses) and all(math.isfinite(x) for x in losses),'Missing/nonfinite task loss')
    k=max(1,int(len(losses)*(1-alpha)))
    return dict(N=len(losses),k=k,mean_BCE=statistics.mean(losses),CVaR=statistics.mean(sorted(losses,reverse=True)[:k]))


def aggregate(values):
    require(bool(values),'No completed independent families')
    return dict(runs=len(values),mean=statistics.mean(values),SE=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)


def calibration(rows,c):
    spec=c['classification']['calibration']
    if len(rows)<spec['minimum_rows']:return dict(status='INSUFFICIENT_ROWS',rows=len(rows),minimum=spec['minimum_rows'])
    bins=[];count=spec['bins']
    for i in range(count):
        r=[v for v in rows if min(int(v['probability']*count),count-1)==i]
        bins.append(dict(bin=i,lower=i/count,upper=(i+1)/count,n=len(r),
            mean_probability=statistics.mean(v['probability'] for v in r) if r else None,
            empirical_positive=statistics.mean(v['label'] for v in r) if r else None))
    return dict(status='DIAGNOSTIC_NO_CALIBRATION_FIT',rows=len(rows),Brier=statistics.mean((r['probability']-r['label'])**2 for r in rows),bins=bins)
