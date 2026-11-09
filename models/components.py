"""Reusable neural, Laplace, boosting, and k-DPP components."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MetaClassifier(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0).expand(X.shape[0], -1)
        return self.network(torch.cat((X, z), dim=-1)).squeeze(-1)


class ResidualStumpBoost:
    """Dependency-free gradient booster using one-dimensional stumps."""

    def __init__(self, estimators: int = 12, learning_rate: float = 0.15):
        self.estimators, self.learning_rate = estimators, learning_rate
        self.base = 0.0
        self.trees: list[tuple[int, float, float, float]] = []

    def fit(self, X: np.ndarray, residual: np.ndarray) -> "ResidualStumpBoost":
        X, residual = np.asarray(X), np.asarray(residual).reshape(-1)
        self.base, self.trees = float(residual.mean()), []
        prediction = np.full_like(residual, self.base)
        for _ in range(self.estimators):
            target = residual - prediction
            best = None
            for feature in range(X.shape[1]):
                thresholds = np.unique(np.quantile(X[:, feature], np.linspace(.1, .9, 9)))
                for threshold in thresholds:
                    left = X[:, feature] <= threshold
                    if left.all() or (~left).all():
                        continue
                    lv, rv = float(target[left].mean()), float(target[~left].mean())
                    error = float(np.sum((target - np.where(left, lv, rv)) ** 2))
                    if best is None or error < best[0]:
                        best = (error, feature, float(threshold), lv, rv)
            if best is None:
                break
            _, feature, threshold, lv, rv = best
            self.trees.append((feature, threshold, lv, rv))
            prediction += self.learning_rate * np.where(X[:, feature] <= threshold, lv, rv)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = np.full(X.shape[0], self.base)
        for feature, threshold, lv, rv in self.trees:
            out += self.learning_rate * np.where(X[:, feature] <= threshold, lv, rv)
        return out


def binary_good_labels(y: np.ndarray, gamma: float) -> np.ndarray:
    cutoff = np.quantile(np.asarray(y).reshape(-1), gamma)
    return (np.asarray(y).reshape(-1) <= cutoff).astype(np.float32)


def laplace_embedding(
    model: MetaClassifier,
    X: np.ndarray,
    y: np.ndarray,
    embedding_dim: int,
    gamma: float,
    steps: int,
    lr: float,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MAP embedding and inverse Hessian covariance of its posterior."""
    device = next(model.parameters()).device
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    labels = torch.as_tensor(binary_good_labels(y, gamma), device=device)
    z = torch.zeros(embedding_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        nll = F.binary_cross_entropy_with_logits(model(xt, z), labels)
        loss = nll + 0.5 * z.square().sum() / max(1, len(y))
        loss.backward()
        optimizer.step()
    z_map = z.detach()

    def negative_log_posterior(value: torch.Tensor) -> torch.Tensor:
        nll = F.binary_cross_entropy_with_logits(model(xt, value), labels, reduction="sum")
        return nll + 0.5 * value.square().sum()

    # Do mạng meta phi tuyến theo z, Hessian số có thể hơi bất định dương
    # (đặc biệt khi Adam chưa hội tụ đúng MAP). Laplace cần precision PSD;
    # vì vậy đối xứng hóa và chiếu eigenvalues về một sàn dương ổn định.
    hessian = torch.autograd.functional.hessian(negative_log_posterior, z_map)
    hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
    identity = torch.eye(embedding_dim, device=device, dtype=hessian.dtype)
    precision = hessian + float(jitter) * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(precision)
    if not torch.isfinite(eigenvalues).all():
        # Fallback an toàn khi Hessian overflow/underflow trong một run.
        covariance = identity
    else:
        # Giới hạn cả precision quá nhỏ (variance nổ) và quá lớn (variance 0).
        precision_floor = max(float(jitter), 1e-3)
        clipped = eigenvalues.clamp(min=precision_floor, max=1e4)
        covariance = (eigenvectors * clipped.reciprocal()) @ eigenvectors.transpose(-1, -2)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        covariance = covariance + (precision_floor * 1e-3) * identity
    return z_map, covariance


def sample_k_dpp(L: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Exact fixed-size k-DPP sampler for a positive semidefinite L kernel."""
    n = L.shape[0]
    k = min(k, n)
    eigenvalues, eigenvectors = np.linalg.eigh((L + L.T) * 0.5)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # E[l,n] stores elementary symmetric polynomials.
    E = np.zeros((k + 1, n + 1), dtype=np.float64)
    E[0, :] = 1.0
    for length in range(1, k + 1):
        for count in range(1, n + 1):
            E[length, count] = (
                E[length, count - 1]
                + eigenvalues[count - 1] * E[length - 1, count - 1]
            )
    chosen: list[int] = []
    remaining = k
    for i in range(n, 0, -1):
        if remaining == 0:
            break
        denominator = E[remaining, i]
        probability = 0.0 if denominator <= 0 else (
            eigenvalues[i - 1] * E[remaining - 1, i - 1] / denominator
        )
        if rng.random() < np.clip(probability, 0.0, 1.0):
            chosen.append(i - 1)
            remaining -= 1
    if remaining:  # Degenerate kernel fallback.
        pool = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=int))
        chosen.extend(rng.choice(pool, remaining, replace=False).tolist())
    V = eigenvectors[:, chosen]
    selected: list[int] = []
    while V.shape[1] and len(selected) < k:
        probabilities = np.sum(V * V, axis=1)
        probabilities /= probabilities.sum()
        item = int(rng.choice(n, p=probabilities))
        selected.append(item)
        nonzero = np.flatnonzero(np.abs(V[item]) > 1e-12)
        if nonzero.size == 0:
            break
        column = int(nonzero[0])
        pivot = V[:, column].copy()
        V = np.delete(V, column, axis=1)
        if V.shape[1]:
            V -= np.outer(pivot, V[item] / pivot[item])
            V, _ = np.linalg.qr(V)
    return np.asarray(selected, dtype=int)
