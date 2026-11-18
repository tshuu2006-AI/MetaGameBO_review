"""Stable Gaussian population probabilities; stdlib, independently verifiable."""
import math
from .protocol import require


def logadd(a,b):
    m=max(a,b)
    return m+math.log1p(math.exp(-abs(a-b)))


def logcdf(z,c):
    require(math.isfinite(z),'Nonfinite Gaussian z')
    if z>=c['numerics']['tail_boundary']:
        return math.log(.5*math.erfc(-z/math.sqrt(2)))
    a=-z;quad=c['numerics']
    integral=math.fsum(w*math.exp(-.5*(x/a)**2) for x,w in zip(quad['nodes'],quad['weights']))
    return -.5*z*z-.5*math.log(2*math.pi)-math.log(a)+math.log(integral)


def gaussian(mean,variance,tau,U,c):
    require(all(math.isfinite(x) for x in (mean,variance,tau,U)) and variance>0 and U>0,'Degenerate/nonfinite Gaussian or U_train; no variance floor')
    sigma=math.sqrt(variance);z=(tau-mean)/sigma;lp,lm=logcdf(z,c),logcdf(-z,c)
    phi=-.5*z*z-.5*math.log(2*math.pi)
    if z<c['numerics']['tail_boundary']:
        a=-z;quad=c['numerics']
        integral=math.fsum(w*x*math.exp(-.5*(x/a)**2) for x,w in zip(quad['nodes'],quad['weights']))
        lei=math.log(sigma)+phi-2*math.log(a)+math.log(integral)
    elif z<0:
        lei=math.log(sigma)+phi+math.log(-math.expm1(math.log(-z)+lp-phi))
    elif z==0:lei=math.log(sigma)+phi
    else:lei=math.log(sigma)+logadd(math.log(z)+lp,phi)
    denominator=logadd(lei,math.log(U));lq=lei-denominator;lnotq=math.log(U)-denominator
    result=dict(z=z,event_probability=math.exp(lp),event_log_p=lp,event_log_not_p=lm,
        log_EI=lei,EI=math.exp(lei),augmented_probability=math.exp(lq),augmented_log_p=lq,augmented_log_not_p=lnotq)
    require(all(math.isfinite(v) for v in result.values()),'Nonfinite probability; run must fail explicitly')
    return result


def rf_scores(p,epsilon):
    require(math.isfinite(p) and 0<=p<=1,'Invalid RF probability')
    used=min(max(p,epsilon),1-epsilon)
    return dict(probability=p,p_used=used,log_p=math.log(used),log_not_p=math.log1p(-used),clipped=p!=used)
