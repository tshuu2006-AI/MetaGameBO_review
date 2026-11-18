"""Real GPR posterior, numerical stability, acquisition and reproducibility."""
import numpy as np
import pytest
from scipy.special import ndtr
from models.gp import GP
from models.base_model import BaseOptimizer


def fitted(**options):
    x=np.array([[.05],[.25],[.55],[.85],[.95]])
    y=np.sin(6*x[:,0])*7+31
    return GP([[0,1]],seed=42,**options).fit(x,y)


def test_real_posterior_kernel_units_and_noise():
    gp=fitted();x=np.array([[.2],[.7]])
    assert isinstance(gp,BaseOptimizer)
    mean,var=gp.posterior(x);noisy_mean,noisy=gp.posterior(x,include_observation_noise=True)
    assert np.isfinite(mean).all() and np.all(var>=0)
    np.testing.assert_allclose(gp.predict(x,return_std=True)[1]**2,var)
    np.testing.assert_array_equal(noisy_mean,mean)
    np.testing.assert_allclose(noisy-var,gp.options['observation_noise']*gp.fit_metadata['output_scale']**2)
    assert np.all(mean>15)  # original output units, not normalized mean
    assert gp.model.kernel_.k2.nu==2.5
    assert np.isfinite(gp.fit_metadata['log_marginal_likelihood'])
    assert gp.model.kernel_(np.array([[0],[1]]))[0,0]>gp.model.kernel_(np.array([[0],[1]]))[0,1]


def test_event_probability_is_latent_normal_cdf():
    gp=fitted();queries=np.array([[.18],[.72]])
    p,mu,var=gp.event_probability(queries,30)
    np.testing.assert_allclose(p,ndtr((30-mu)/np.sqrt(var)))
    assert np.all((p>=0)&(p<=1))


def test_zero_variance_cdf_and_minimization_ei(monkeypatch):
    gp=GP([[0,1]])
    monkeypatch.setattr(gp,'posterior',lambda x,**kw:(np.array([-2.,1.,1.]),np.zeros(3)))
    p,_,_=gp.event_probability([[0],[.5],[1]],1.)
    np.testing.assert_array_equal(p,[1,1,1])
    np.testing.assert_array_equal(gp.expected_improvement([[0],[.5],[1]],best_y=0),[2,0,0])
    monkeypatch.setattr(gp,'posterior',lambda x,**kw:(np.array([-1.,1.]),np.ones(2)))
    ei=gp.expected_improvement([[0],[1]],best_y=0)
    reference=np.array([1,-1])*ndtr([1,-1])+np.exp(-.5)/np.sqrt(2*np.pi)
    np.testing.assert_allclose(ei,reference);assert ei[0]>ei[1]


def test_seed_reproducibility_and_candidate_argmax():
    a=fitted();b=fitted()
    np.testing.assert_array_equal(a.predict([[.14],[.68]]),b.predict([[.14],[.68]]))
    a.rng=np.random.default_rng(55);b.rng=np.random.default_rng(55)
    candidates=b._uniform(b.options['n_candidates'])
    expected=candidates[np.argmax(b.expected_improvement(candidates))]
    np.testing.assert_array_equal(a.suggest()[0],expected)
    a.rng=np.random.default_rng(88);b.rng=np.random.default_rng(88)
    np.testing.assert_array_equal(a.suggest(),b.suggest())


def test_duplicate_observations_and_constant_output_are_stable():
    gp=GP([[0,1]],seed=1,observation_noise=0,optimizer=None).fit([[.3],[.3],[.9]],[4,4,4])
    mu,var=gp.posterior([[.3],[.6]])
    assert np.isfinite(mu).all() and np.all(var>=0)
    np.testing.assert_allclose(mu,4)


def test_prespecified_cholesky_retry(monkeypatch):
    from sklearn.gaussian_process import GaussianProcessRegressor
    native=GaussianProcessRegressor.fit;calls=[]
    def unstable(self,X,y):
        calls.append(self.alpha)
        if len(calls)==1:raise np.linalg.LinAlgError('injected factorization failure')
        return native(self,X,y)
    monkeypatch.setattr(GaussianProcessRegressor,'fit',unstable)
    gp=fitted(optimizer=None)
    assert len(calls)==2 and gp.fit_metadata['attempts'][0]['status']=='CHOLESKY_FAILED'


@pytest.mark.parametrize('options',[{'observation_noise':-1},{'length_scale':0},{'n_candidates':False},
    {'nu':1.5},{'acquisition':'ucb'},{'n_restarts_optimizer':-1},{'jitter_schedule':[]},{'unknown':1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):GP([[0,1]],**options)


def test_observe_fit_reset_and_budget():
    gp=GP([[0,1]],seed=5,n_candidates=32,optimizer=None)
    gp.observe([[.2]],[1]);gp.observe([[.8]],[2]);assert len(gp.history.y)==2
    gp.fit([[.4]],[9]);assert gp.history.y==[9]
    gp.reset();assert gp.model is None and not gp.history.y
    history=gp.optimize(lambda X:(X[:,0]-.3)**2,budget=5,n_init=2)
    assert len(history.y)==5 and gp.best[1]==min(history.y)
    for n in (0,2,True,1.0):
        with pytest.raises(ValueError):gp.suggest(n)
    with pytest.raises(ValueError):gp.fit([[2]],[0])
    with pytest.raises(ValueError):gp.event_probability([[.4]],np.nan)
