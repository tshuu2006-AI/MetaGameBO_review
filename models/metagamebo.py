"""CVaR task game and quality-diversity sample game for MetaGameBO."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.base import BenchmarkTask
from .components import binary_good_labels, sample_k_dpp
from .malibo import MALIBOAgent


class MetaGameBOAgent(MALIBOAgent):
    def __init__(self, *args, cvar_alpha: float = 0.8,
                 dpp_batch_size: int = 96,
                 diversity_lengthscale: float = 0.15,
                 sample_learning_rate: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.cvar_alpha = cvar_alpha
        self.dpp_batch_size = dpp_batch_size
        self.diversity_lengthscale = diversity_lengthscale
        self.sample_learning_rate = sample_learning_rate

    @staticmethod
    def _kde_quantile(losses: np.ndarray, alpha: float) -> float:
        """Gaussian-KDE estimate of xi_alpha used by the task selector."""
        values = torch.as_tensor(losses, dtype=torch.float64)
        if len(values) < 2 or float(values.std()) < 1e-12:
            return float(torch.quantile(values, alpha))
        bandwidth = 1.06 * values.std(unbiased=True) * len(values) ** (-0.2)
        grid = torch.linspace(
            float(values.min() - 3 * bandwidth),
            float(values.max() + 3 * bandwidth), 512,
            dtype=torch.float64,
        )
        cdf = torch.special.ndtr(
            (grid[:, None] - values[None, :]) / bandwidth
        ).mean(dim=1)
        target = torch.tensor(alpha, dtype=torch.float64)
        index = int(torch.searchsorted(cdf, target).clamp_max(511))
        return float(grid[index])

    def _initial_sample_weights(self, task: BenchmarkTask) -> np.ndarray:
        """Eq. (3): improvement utility plus class balancing/normalisation."""
        y = np.asarray(task.y, dtype=np.float64)
        threshold = float(np.quantile(y, self.gamma))
        promising = y < threshold
        improvement = np.maximum(threshold - y, 0.0)
        baseline = max(float(improvement[promising].mean())
                       if promising.any() else 1.0, 1e-6)
        weights = np.where(promising, improvement + 1e-6, baseline)
        # Cân bằng tổng trọng số giữa promising/reference classes.
        if promising.any():
            weights[promising] /= promising.sum()
        if (~promising).any():
            weights[~promising] /= (~promising).sum()
        return weights / weights.mean()

    def meta_train(self, tasks: list[BenchmarkTask]) -> None:
        means = torch.nn.Parameter(
            torch.zeros(len(tasks), self.embedding_dim, device=self.device)
        )
        logvars = torch.nn.Parameter(torch.full_like(means, -2.0))
        optimizer = torch.optim.Adam(
            [*self.model.parameters(), means, logvars], lr=2e-3
        )
        rng = np.random.default_rng(self.seed)
        sample_weights = [self._initial_sample_weights(task) for task in tasks]
        for _ in range(self.meta_epochs):
            # Leader: T_hard={tau_i | L_i >= xi_alpha}.
            with torch.no_grad():
                task_losses = np.array([
                    float(self._task_loss(task, means[i], logvars[i]).cpu())
                    for i, task in enumerate(tasks)
                ])
            threshold = self._kde_quantile(task_losses, self.cvar_alpha)
            hard_indices = np.flatnonzero(task_losses >= threshold)
            if hard_indices.size == 0:  # KDE smoothing can exceed max on tiny M.
                hard_indices = np.array([int(np.argmax(task_losses))])
            optimizer.zero_grad()
            losses = []
            for i in hard_indices:
                task = tasks[int(i)]
                X = np.asarray(task.X)
                labels = binary_good_labels(task.y, self.gamma)
                xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
                z = means[i] + torch.exp(0.5 * logvars[i]) * torch.randn_like(means[i])
                logits = self.model(xt, z)
                point_loss = F.binary_cross_entropy_with_logits(
                    logits, torch.as_tensor(labels, device=self.device),
                    reduction="none",
                )
                # Follower: Eq. (3)/(6) quality; RBF adds repulsive diversity.
                quality = sample_weights[int(i)] + 1e-8
                # Giới hạn eigendecomposition vào pool chất lượng cao để k-DPP
                # vẫn O(p^3) với p nhỏ khi mỗi task có 512+ samples.
                pool_size = min(len(X), max(2 * self.dpp_batch_size, 192))
                pool = np.argpartition(quality, -pool_size)[-pool_size:]
                X_pool, quality_pool = X[pool], quality[pool]
                distances = np.sum(
                    (X_pool[:, None, :] - X_pool[None, :, :]) ** 2, axis=-1
                )
                similarity = np.exp(
                    -distances / (2.0 * self.diversity_lengthscale ** 2)
                )
                kernel = np.sqrt(
                    quality_pool[:, None] * quality_pool[None, :]
                ) * similarity
                selected_local = sample_k_dpp(
                    kernel, min(self.dpp_batch_size, len(X_pool)), rng
                )
                selected = pool[selected_local]
                selected_t = torch.as_tensor(
                    selected, dtype=torch.long, device=self.device
                )
                weighted_bce = point_loss[selected_t].mean()
                kl = 0.5 * torch.sum(
                    means[i].square() + logvars[i].exp() - 1.0 - logvars[i]
                )
                losses.append(weighted_bce + self.beta * kl / len(selected))
                # Eq. (6): multiplicative update with approximate inclusion
                # probability q_i from the DPP marginal kernel.
                marginal = kernel @ np.linalg.inv(
                    np.eye(len(kernel)) + kernel
                )
                q = np.clip(np.diag(marginal)[selected_local], 1e-3, 1.0)
                observed_loss = point_loss.detach().cpu().numpy()[selected]
                centered = observed_loss - observed_loss.mean()
                exponent = np.clip(
                    self.sample_learning_rate * centered / q, -5.0, 5.0
                )
                sample_weights[int(i)][selected] *= np.exp(exponent)
                sample_weights[int(i)] /= sample_weights[int(i)].mean()
            torch.stack(losses).mean().backward()
            optimizer.step()
        self._trained = True
