"""Target-task adaptation of a meta-trained classifier.

Bayesian logistic regression on the task embedding z with a Laplace posterior
(Sigma_N = inverse Hessian, paper Eq. 4), Thompson sampling, and an init
estimator that lets a gradient-boosting residual start from a Thompson sample
(as in MALIBO).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from models.malibo.meta_net import DT, MetaNet


class BLRAdapter:
    """Bayesian logistic regression on the target-task embedding z (Laplace posterior).

    The meta-learned features stay fixed; only z (prior N(0, I)) is fitted.

    Args:
        net: Trained meta-classifier.
        hessian: ``"exact"`` uses the Hessian of the weighted negative
            log-posterior, ``I + Phi^T diag(w p (1 - p)) Phi`` (paper: Sigma_N is
            the inverse Hessian). ``"upstream"`` uses the MALIBO code's term
            ``p (1 - p) (lab * w + 1)`` instead.

    Attributes:
        z: Posterior mode, shape (d,).
        H: Posterior precision, shape (d, d).
        cov: Posterior covariance, shape (d, d).
    """

    def __init__(self, net: MetaNet, hessian: str = "exact"):
        if hessian not in ("exact", "upstream"):
            raise ValueError(f"unknown hessian {hessian!r}")
        self.net = net
        self.hessian = hessian
        self.d = net.num_features
        self.reset()

    def reset(self):
        """Reset the posterior to the prior N(0, I)."""
        self.z = np.zeros(self.d)
        self.H = np.eye(self.d)
        self.cov = np.eye(self.d)

    def phi_m(self, X):
        """Meta-learned features and mean as numpy arrays.

        Args:
            X: Inputs, shape (n, d) or (d,).

        Returns:
            Tuple ``(phi, m)`` of shapes (n, num_features) and (n,).
        """
        with torch.no_grad():
            phi, m = self.net.features(torch.tensor(np.atleast_2d(X), dtype=DT))
        return phi.numpy(), m.numpy()

    def fit(self, X, lab, w):
        """Fit the Laplace posterior over z (MAP via L-BFGS, then Hessian).

        Args:
            X: Inputs, shape (n, d).
            lab: Binary labels, shape (n,).
            w: Sample weights, shape (n,).
        """
        phi, m = self.phi_m(X)

        def f(z):
            a = m + phi @ z
            p = expit(a)
            nll = np.sum(w * (np.logaddexp(0, a) - lab * a)) + 0.5 * z @ z
            g = phi.T @ (w * (p - lab)) + z
            return nll, g

        self.z = minimize(f, np.zeros(self.d), jac=True, method="L-BFGS-B").x
        p = expit(m + phi @ self.z)
        if self.hessian == "exact":
            r = w * p * (1 - p)
        else:
            r = p * (1 - p) * (lab * w + 1.0)  # upstream MALIBO curvature term
        self.H = np.eye(self.d) + (phi * r[:, None]).T @ phi
        self.cov = np.linalg.inv(self.H)

    def prob_max(self, X):
        """Posterior predictive P(good | x) (probit approximation).

        Args:
            X: Inputs, shape (n, d).

        Returns:
            Probabilities, shape (n,).
        """
        phi, m = self.phi_m(X)
        var = np.einsum("ij,jk,ik->i", phi, self.cov, phi)
        return expit((m + phi @ self.z) / np.sqrt(1 + var * math.pi / 8))

    def sample_z(self, rng):
        """Draw one Thompson sample of z from the posterior.

        Args:
            rng: Numpy random generator.

        Returns:
            Embedding sample, shape (num_features,).
        """
        return rng.multivariate_normal(self.z, self.cov)

    def prob_ts(self, X, zs):
        """P(good | x) under a fixed embedding sample.

        Args:
            X: Inputs, shape (n, d).
            zs: Embedding sample from ``sample_z``, shape (num_features,).

        Returns:
            Probabilities, shape (n,).
        """
        phi, m = self.phi_m(X)
        return expit(m + phi @ zs)


class ThompsonInit(BaseEstimator, ClassifierMixin):
    """Gradient-boosting init estimator: one Thompson sample of the meta-classifier.

    Args:
        owner: Object with an ``adapter`` (``BLRAdapter``) and the current
            Thompson sample ``_z_ts`` (the ``MALIBO`` optimizer).
    """

    def __init__(self, owner=None):
        self.owner = owner

    def fit(self, X, y, sample_weight=None):
        """No-op fit (the meta-classifier is already trained); sets ``classes_``.

        Args:
            X: Ignored.
            y: Ignored.
            sample_weight: Ignored.

        Returns:
            self
        """
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X):
        """Class probabilities from the Thompson-sampled meta-classifier.

        Args:
            X: Points, shape (n, d).

        Returns:
            Array of shape (n, 2) with ``[P(bad), P(good)]``.
        """
        p = np.clip(self.owner.adapter.prob_ts(X, self.owner._z_ts), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])
