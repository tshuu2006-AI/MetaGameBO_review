"""Executable MALIBO-style likelihood-free meta-BO agent."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.base import BenchmarkTask
from .components import (
    MetaClassifier, ResidualStumpBoost, binary_good_labels, laplace_embedding,
)


class MALIBOAgent:
    """Variational task embeddings plus Laplace/TS target adaptation."""

    def __init__(self, bounds: np.ndarray, seed: int = 0, embedding_dim: int = 6,
                 hidden_dim: int = 64, gamma: float = 0.25, beta: float = 1e-3,
                 meta_epochs: int = 80, adaptation_steps: int = 60,
                 device: str = "cpu"):
        torch.manual_seed(seed)
        self.bounds, self.seed = np.asarray(bounds), seed
        self.rng = np.random.default_rng(seed)
        self.embedding_dim, self.gamma, self.beta = embedding_dim, gamma, beta
        self.meta_epochs = meta_epochs
        self.adaptation_steps, self.device = adaptation_steps, device
        self.model = MetaClassifier(len(bounds), embedding_dim, hidden_dim).to(device)
        self._trained = False

    def _task_loss(self, task: BenchmarkTask, mu: torch.Tensor,
                   logvar: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(task.X, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(binary_good_labels(task.y, self.gamma), device=self.device)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        bce = F.binary_cross_entropy_with_logits(self.model(X, z), labels)
        # KL[q(z_tau)||N(0,I)] = .5 sum(mu^2+sigma^2-1-log sigma^2).
        kl = 0.5 * torch.sum(mu.square() + logvar.exp() - 1.0 - logvar)
        return bce + self.beta * kl / len(task.y)

    def meta_train(self, tasks: list[BenchmarkTask]) -> None:
        means = torch.nn.Parameter(torch.zeros(len(tasks), self.embedding_dim, device=self.device))
        logvars = torch.nn.Parameter(torch.full_like(means, -2.0))
        optimizer = torch.optim.Adam([*self.model.parameters(), means, logvars], lr=2e-3)
        for _ in range(self.meta_epochs):
            optimizer.zero_grad()
            loss = torch.stack([
                self._task_loss(task, means[i], logvars[i])
                for i, task in enumerate(tasks)
            ]).mean()
            loss.backward()
            optimizer.step()
        self._trained = True

    def suggest(self, X: np.ndarray, y: np.ndarray,
                candidates: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Call meta_train before suggest")
        z_map, covariance = laplace_embedding(
            self.model, X, y, self.embedding_dim, self.gamma,
            self.adaptation_steps, 0.04, 1e-3,
        )
        # Thompson sampling: z_tilde ~ N(z_MAP, Sigma_N).
        distribution = torch.distributions.MultivariateNormal(
            z_map, covariance_matrix=covariance
        )
        z_sample = distribution.sample()
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        ct = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            train_logits = self.model(xt, z_sample).cpu().numpy()
            candidate_logits = self.model(ct, z_sample).cpu().numpy()
        labels = binary_good_labels(y, self.gamma)
        probabilities = 1.0 / (1.0 + np.exp(-train_logits))
        booster = ResidualStumpBoost().fit(X, labels - probabilities)
        acquisition = candidate_logits + booster.predict(candidates)
        return candidates[int(np.argmax(acquisition))]
