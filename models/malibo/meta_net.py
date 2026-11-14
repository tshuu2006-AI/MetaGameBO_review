"""MALIBO meta-classifier network and its embedding regulariser.

    logit(x; tau) = m(x) + phi(x)^T z_tau

Attributes:
    DT: Torch dtype used throughout meta-training and adaptation.
"""
from __future__ import annotations

import math

import torch

DT = torch.float64


class MetaNet(torch.nn.Module):
    """Residual MLP meta-classifier, structured like upstream TaskAgnosticModel.

    Computes ``logit(x; tau) = m(x) + phi(x)^T z_tau``.

    Args:
        input_dim: Input dimension d.
        hidden: Width of the hidden layers.
        num_layers: Total number of layers (input + residual + feature).
        num_features: Feature / embedding dimension.
        num_tasks: Number of meta-training tasks M.
        emb_noise: Std of Gaussian noise added to embeddings in training mode.

    Attributes:
        emb: Task embeddings, shape (M + 1, num_features); row 0 is the
            target task and stays zero.
    """

    def __init__(self, input_dim=1, hidden=64, num_layers=5, num_features=50,
                 num_tasks=0, emb_noise=0.1):
        super().__init__()
        self.inp = torch.nn.Linear(input_dim, hidden)
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden) for _ in range(num_layers - 2)])
        self.feat = torch.nn.Linear(hidden, num_features)
        self.mean = torch.nn.Linear(num_features, 1)
        self.num_features = num_features
        self.emb_noise = emb_noise
        # index 0 = target task (kept at zero), 1..M = meta tasks
        self.emb = torch.nn.Parameter(torch.zeros(num_tasks + 1, num_features))

    def features(self, x):
        """Shared features and task-agnostic mean.

        Args:
            x: Inputs, tensor of shape (n, d).

        Returns:
            Tuple ``(phi, m)`` of shapes (n, num_features) and (n,).
        """
        h =torch.nn.functional.elu(self.inp(x))
        for lin in self.hidden:
            h = torch.nn.functional.elu(h + lin(h))
        phi = torch.nn.functional.elu(self.feat(h))
        return phi, self.mean(phi).squeeze(-1)

    def forward(self, x, task_idx):
        """Task-conditioned logits.

        Args:
            x: Inputs, tensor of shape (n, d).
            task_idx: Task index per input (0 = target), long tensor of shape (n,).

        Returns:
            Logits, tensor of shape (n,).
        """
        phi, m = self.features(x)
        z = self.emb[task_idx]
        if self.training:
            z = z + self.emb_noise * torch.randn_like(z)
        return m + (phi * z).sum(-1)


def ks_cov_loss(E: torch.Tensor) -> torch.Tensor:
    """MALIBO embedding regulariser towards N(0, I).

    Args:
        E: Task embeddings, shape (M, d).

    Returns:
        Tuple ``(ks, cv)``: squared KS-style distance of the marginals to N(0, 1),
        and MSE between the empirical covariance and the identity.
    """
    n = E.shape[0]
    s = torch.sort(E, dim=0).values
    ecdf = torch.linspace(1 / n, 1.0, n, dtype=E.dtype).unsqueeze(-1)
    ks = ((0.5 * (1 + torch.erf(s / math.sqrt(2))) - ecdf) ** 2).mean()
    Ec = E - E.mean(0)
    cov = Ec.T @ Ec / n
    cv = torch.nn.functional.mse_loss(cov, torch.eye(E.shape[1], dtype=E.dtype))
    return ks, cv


def reg_scales(n, d, reps=64, seed=0):
    """Scales that normalise the KS and covariance losses of true N(0, I) samples to 0.5.

    Args:
        n: Number of embeddings.
        d: Embedding dimension.
        reps: Number of Monte Carlo draws.
        seed: Seed of the torch generator.

    Returns:
        Tuple ``(ks_scale, cv_scale)``.
    """
    g = torch.Generator().manual_seed(seed)
    ks, cv = zip(*[ks_cov_loss(torch.randn(n, d, generator=g, dtype=DT)) for _ in range(reps)])
    return 0.5 / float(torch.stack(ks).mean()), 0.5 / float(torch.stack(cv).mean())
