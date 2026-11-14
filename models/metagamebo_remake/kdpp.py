"""Exact k-DPP sampling and marginals (Kulesza & Taskar 2011/2012).

The paper uses an L-ensemble with diagonal kernel K_t = diag(w_t).  We support
that exactly, plus a quality-diversity kernel L = diag(sqrt w) S diag(sqrt w)
(S = RBF similarity over inputs) for comparison: with a *diagonal* kernel all
eigenvectors are basis vectors, so there is no repulsion between items and the
"diversity" claimed in the paper cannot arise.

The author's approximate ``DiagonalKDPP`` lives in ``models/metagamebo_original``.
"""
from __future__ import annotations

import numpy as np


def _esp_table(lams: np.ndarray, k: int) -> np.ndarray:
    """Table of elementary symmetric polynomials of eigenvalue prefixes.

    Args:
        lams: Eigenvalues, shape (n,).
        k: Highest polynomial order.

    Returns:
        Array ``E`` of shape (k + 1, n + 1) with ``E[l, m] = e_l(lams[:m])``.
    """
    n = len(lams)
    E = np.zeros((k + 1, n + 1))
    E[0, :] = 1.0
    for l in range(1, k + 1):
        for m in range(1, n + 1):
            E[l, m] = E[l, m - 1] + lams[m - 1] * E[l - 1, m - 1]
    return E


def _esp_excluding(lams: np.ndarray, k: int) -> np.ndarray:
    """Elementary symmetric polynomials with one eigenvalue left out, O(n^2 k).

    Args:
        lams: Eigenvalues, shape (n,).
        k: Highest polynomial order.

    Returns:
        Array ``R`` of shape (n, k + 1) with ``R[m, l] = e_l(lams without item m)``.
    """
    n = len(lams)
    R = np.zeros((n, k + 1))
    R[:, 0] = 1.0
    for m in range(n):
        # every row except m absorbs item m; iterate l downward (0/1 knapsack style)
        mask = np.ones(n, dtype=bool)
        mask[m] = False
        R[mask, 1:] = R[mask, 1:] + lams[m] * R[mask, :-1]
    return R


class KDPP:
    """k-DPP over n items defined by an L-ensemble kernel.

    Args:
        L: Symmetric PSD kernel, shape (n, n). It is symmetrised and its
            eigenvalues are rescaled (a k-DPP is invariant to scaling L).
        k: Subset size, clipped to ``[1, n]``.

    Attributes:
        n: Number of items.
        k: Subset size.
        lams: Rescaled eigenvalues of ``L``, shape (n,).
        V: Eigenvectors of ``L``, shape (n, n).
    """

    def __init__(self, L: np.ndarray, k: int):
        L = 0.5 * (L + L.T)
        self.n = L.shape[0]
        self.k = int(min(max(k, 1), self.n))
        lams, V = np.linalg.eigh(L)
        lams = np.clip(lams, 0.0, None)
        # rescale for numerical range; k-DPP is invariant to scaling L
        s = lams.max() if lams.max() > 0 else 1.0
        self.lams, self.V = lams / s, V

    @classmethod
    def diagonal(cls, w: np.ndarray, k: int) -> "KDPP":
        """Build the paper's k-DPP with diagonal kernel ``L = diag(w)``.

        Args:
            w: Non-negative item weights, shape (n,).
            k: Subset size.

        Returns:
            The k-DPP instance.
        """
        return cls(np.diag(np.asarray(w, dtype=float)), k)

    @classmethod
    def quality_diversity(cls, w: np.ndarray, X: np.ndarray, k: int,
                          lengthscale: float = 0.1) -> "KDPP":
        """Build a quality-diversity k-DPP, ``L = diag(sqrt w) S diag(sqrt w)``.

        Args:
            w: Non-negative item qualities, shape (n,).
            X: Item inputs used for the RBF similarity ``S``, shape (n, d).
            k: Subset size.
            lengthscale: RBF lengthscale of ``S``.

        Returns:
            The k-DPP instance.
        """
        X = np.atleast_2d(X)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        S = np.exp(-0.5 * d2 / lengthscale ** 2)
        q = np.sqrt(np.asarray(w, dtype=float))
        return cls(q[:, None] * S * q[None, :] + 1e-9 * np.eye(len(q)), k)

    def marginals(self) -> np.ndarray:
        """Exact inclusion probabilities ``P(i in S)``.

        Computed as ``sum_n lam_n e_{k-1}(lam_-n) / e_k(lam) * V[i, n]^2``.

        Returns:
            Array of shape (n,) with values in [0, 1], summing to ``k``.
        """
        R = _esp_excluding(self.lams, self.k)
        ek = _esp_table(self.lams, self.k)[self.k, -1]
        coef = self.lams * R[:, self.k - 1] / ek  # P(eigvec n selected)
        return np.clip((self.V ** 2) @ coef, 0.0, 1.0)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one exact sample (eigenvector selection, then projection DPP).

        Args:
            rng: Random generator.

        Returns:
            Sorted indices of the ``k`` selected items, shape (k,).
        """
        lams, n, k = self.lams, self.n, self.k
        E = _esp_table(lams, k)
        # phase 1: choose k eigenvectors
        chosen, l = [], k
        for m in range(n, 0, -1):
            if l == 0:
                break
            if m == l or rng.random() < lams[m - 1] * E[l - 1, m - 1] / E[l, m]:
                chosen.append(m - 1)
                l -= 1
        V = self.V[:, chosen].copy()
        # phase 2: projection DPP sampling
        out = []
        while V.shape[1] > 0:
            p = (V ** 2).sum(1)
            p = p / p.sum()
            i = rng.choice(n, p=p)
            out.append(i)
            j = np.argmax(np.abs(V[i, :]))
            Vj = V[:, j].copy()
            V = np.delete(V, j, axis=1)
            V = V - np.outer(Vj, V[i, :] / Vj[i])
            if V.shape[1] > 0:
                V, _ = np.linalg.qr(V)
        return np.array(sorted(out), dtype=int)
