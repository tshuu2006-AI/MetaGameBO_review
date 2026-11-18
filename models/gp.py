"""Gaussian-process regression and minimization EI; independent of PFN code.

Model choices not specified by MetaGameBO are explicit configuration assumptions.
The posterior event probability is a Normal CDF, never an acquisition score.
"""
from __future__ import annotations

import copy
from numbers import Integral, Real
from time import perf_counter
import warnings

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from models.base_model import BaseOptimizer, History, validate_points, validate_values


DEFAULT_OPTIONS = dict(length_scale=.2, length_scale_bounds=(.01, 10.),
    amplitude=1., amplitude_bounds=(.001, 1000.), nu=2.5,
    observation_noise=1e-6, jitter_schedule=(1e-10, 1e-9, 1e-8),
    normalize_y=True, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
    n_candidates=5000, duplicate_atol=1e-8, acquisition='ei')


def validate_gp_options(options):
    if not isinstance(options, dict) or set(options)-set(DEFAULT_OPTIONS):
        raise ValueError('Unknown GP configuration option')
    result = {**DEFAULT_OPTIONS, **options}
    for key in ('length_scale', 'amplitude', 'observation_noise', 'duplicate_atol'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError('Invalid GP option: '+key)
    if not result['length_scale'] > 0 or not result['amplitude'] > 0 or result['nu'] != 2.5:
        raise ValueError('This audited GP profile uses positive amplitude/scale and Matern nu=2.5')
    for key in ('length_scale_bounds', 'amplitude_bounds'):
        lo, hi = result[key]
        if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
            raise ValueError('Invalid GP kernel bounds')
    if not result['jitter_schedule'] or any(not np.isfinite(j) or j <= 0 for j in result['jitter_schedule']):
        raise ValueError('Need a positive finite prespecified jitter schedule')
    for key, minimum in [('n_candidates', 1), ('n_restarts_optimizer', 0)]:
        if isinstance(result[key], bool) or not isinstance(result[key], Integral) or result[key] < minimum:
            raise ValueError('Invalid GP integer option: '+key)
    if result['acquisition'] != 'ei' or result['optimizer'] not in (None, 'fmin_l_bfgs_b'):
        raise ValueError('Supported GP acquisition=ei; optimizer=L-BFGS-B or None')
    if not isinstance(result['normalize_y'], bool): raise ValueError('normalize_y must be boolean')
    return result


class GP(BaseOptimizer):
    """Exact sklearn GPR with normalized inputs and explicit latent/noisy posterior."""

    is_meta = False

    def __init__(self, bounds, seed=None, **options):
        self.options = validate_gp_options(options)
        super().__init__(bounds, seed)
        self._reset_task_state()

    def _reset_task_state(self):
        self.model = None
        self.fit_metadata = {}

    def _normalize(self, X):
        points = validate_points(X, self.bounds)
        return (points-self.bounds[:, 0])/(self.bounds[:, 1]-self.bounds[:, 0])

    def fit(self, X, y):
        points = validate_points(X, self.bounds); values = validate_values(y, len(points))
        self.history = History(); self.history.add(points, values)
        self._update(points, values)
        return self

    def _update(self, X, y):
        points = self._normalize(X); values = validate_values(y, len(points))
        self.model = None; attempts = []; started = perf_counter()
        for jitter in self.options['jitter_schedule']:
            kernel = ConstantKernel(self.options['amplitude'], self.options['amplitude_bounds'])*Matern(
                length_scale=np.repeat(self.options['length_scale'], self.dim),
                length_scale_bounds=self.options['length_scale_bounds'], nu=self.options['nu'])
            model = GaussianProcessRegressor(kernel=kernel,
                alpha=self.options['observation_noise']+jitter,
                normalize_y=self.options['normalize_y'], optimizer=self.options['optimizer'],
                n_restarts_optimizer=self.options['n_restarts_optimizer'], random_state=self.seed)
            try:
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter('always'); model.fit(points, values)
                if (not np.isfinite(model.L_).all() or not np.isfinite(model.alpha_).all()
                        or not np.isfinite(model.log_marginal_likelihood_value_)):
                    raise FloatingPointError('Nonfinite GP factorization')
                self.model = model
                attempts.append(dict(jitter=jitter, status='PASS'))
                self.fit_metadata = dict(training_rows=len(values), kernel=str(model.kernel_),
                    kernel_theta=model.kernel_.theta.tolist(), log_marginal_likelihood=float(model.log_marginal_likelihood_value_),
                    jitter=jitter, attempts=attempts, warnings=[str(w.message) for w in observed],
                    output_center=float(model._y_train_mean), output_scale=float(model._y_train_std),
                    observation_noise_normalized_variance=self.options['observation_noise'],
                    mean_function='training empirical mean; zero in standardized coordinates' if self.options['normalize_y'] else 'zero',
                    fit_seconds=perf_counter()-started)
                return
            except np.linalg.LinAlgError as exc:
                attempts.append(dict(jitter=jitter, status='CHOLESKY_FAILED', reason=str(exc)))
        raise np.linalg.LinAlgError('GP failed all prespecified jitter attempts: '+str(attempts))

    def posterior(self, X, *, include_observation_noise=False):
        if self.model is None: raise ValueError('GP must be fitted before posterior prediction')
        points = self._normalize(X)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always'); mean, std = self.model.predict(points, return_std=True)
        mean = np.asarray(mean, dtype=float); variance = np.asarray(std, dtype=float)**2
        if include_observation_noise:
            variance += self.options['observation_noise']*self.fit_metadata['output_scale']**2
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
            raise FloatingPointError('Invalid GP posterior mean/variance')
        return mean, variance

    def predict(self, X, *, return_std=False):
        mean, variance = self.posterior(X)
        return (mean, np.sqrt(variance)) if return_std else mean

    def event_probability(self, X, tau, *, include_observation_noise=False):
        """P(f(x)<=tau) for latent f by default; original objective units."""
        if not np.isfinite(tau): raise ValueError('tau must be finite')
        mean, variance = self.posterior(X, include_observation_noise=include_observation_noise)
        std = np.sqrt(variance); p = np.empty(len(mean)); positive = std > 0
        p[positive] = ndtr((tau-mean[positive])/std[positive])
        p[~positive] = (mean[~positive] <= tau).astype(float)
        if not np.isfinite(p).all() or np.any((p < 0)|(p > 1)): raise FloatingPointError('Invalid GP event probability')
        return p, mean, variance

    def expected_improvement(self, X, best_y=None):
        if best_y is None:
            if not self.history.y: raise ValueError('EI needs observations or a finite best_y')
            best_y = min(self.history.y)
        if not np.isfinite(best_y): raise ValueError('best_y must be finite')
        mean, variance = self.posterior(X); std = np.sqrt(variance)
        improvement = best_y-mean; result = np.maximum(improvement, 0.)
        mask = std > 0; z = improvement[mask]/std[mask]
        result[mask] = improvement[mask]*ndtr(z)+std[mask]*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        if not np.isfinite(result).all() or np.any(result < -1e-12): raise FloatingPointError('Invalid GP EI')
        return np.maximum(result, 0.)

    def suggest(self, n=1):
        if isinstance(n, bool) or not isinstance(n, Integral) or n != 1: raise ValueError('GP currently supports suggest(1) only')
        if not self.history.y: return self._uniform(1)
        candidates = self._uniform(self.options['n_candidates']); observed, _ = self.history.arrays
        duplicates = np.any(np.all(np.isclose(candidates[:, None], observed[None], rtol=0,
            atol=self.options['duplicate_atol']), axis=2), axis=1)
        candidates = candidates[~duplicates]
        if not len(candidates): raise RuntimeError('GP candidate pool contains no novel point')
        values = self.expected_improvement(candidates)
        return candidates[[int(np.argmax(values))]]

    def get_config(self):
        return copy.deepcopy(dict(method='gp', seed=self.seed, bounds=self.bounds.tolist(), **self.options))

    @property
    def diagnostics(self):
        return copy.deepcopy(dict(config=self.get_config(), fit=self.fit_metadata,
            posterior='latent f in original objective units; optional observation noise explicit',
            acquisition='minimization expected improvement; not a classification probability'))


GaussianProcess = GP
