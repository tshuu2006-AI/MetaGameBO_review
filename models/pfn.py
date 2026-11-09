"""Prior-fitted in-context surrogate with gradient-free inference."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from benchmarks.base import BenchmarkTask


class InContextSurrogate(nn.Module):
    def __init__(self, input_dim: int, width: int = 64,
                 layers: int = 2, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(input_dim + 2, width)
        block = nn.TransformerEncoderLayer(
            width, heads, width * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.output = nn.Linear(width, 1)

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor,
                candidates: torch.Tensor) -> torch.Tensor:
        batch, context_n, _ = context_x.shape
        candidate_n = candidates.shape[1]
        observed = torch.ones(batch, context_n, 1, device=context_x.device)
        pending = torch.zeros(batch, candidate_n, 1, device=context_x.device)
        context_tokens = torch.cat(
            (context_x, context_y.unsqueeze(-1), observed), dim=-1
        )
        candidate_tokens = torch.cat(
            (candidates, torch.zeros(batch, candidate_n, 1,
                                     device=context_x.device), pending), dim=-1
        )
        tokens = torch.cat((context_tokens, candidate_tokens), dim=1)
        encoded = self.transformer(self.input(tokens))
        return self.output(encoded[:, -candidate_n:]).squeeze(-1)


class PFNAgent:
    """Train once on a task prior; condition only through context at inference."""

    def __init__(self, bounds: np.ndarray, seed: int = 0,
                 meta_epochs: int = 120, context_size: int = 32,
                 device: str = "cpu"):
        torch.manual_seed(seed)
        self.rng, self.device = np.random.default_rng(seed), device
        self.network = InContextSurrogate(len(bounds)).to(device)
        self.meta_epochs, self.context_size = meta_epochs, context_size
        self._trained = False

    def meta_train(self, tasks: list[BenchmarkTask]) -> None:
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-3)
        self.network.train()
        for _ in range(self.meta_epochs):
            task = tasks[int(self.rng.integers(len(tasks)))]
            count = min(self.context_size + 16, len(task.y))
            indices = self.rng.choice(len(task.y), count, replace=False)
            split = min(self.context_size, len(indices) - 1)
            context, query = indices[:split], indices[split:]
            yc_np = np.asarray(task.y)[context]
            center, scale = yc_np.mean(), max(yc_np.std(), 1e-6)
            Xc = torch.as_tensor(task.X[context][None], dtype=torch.float32,
                                 device=self.device)
            yc = torch.as_tensor(((yc_np - center) / scale)[None],
                                 dtype=torch.float32, device=self.device)
            Xq = torch.as_tensor(task.X[query][None], dtype=torch.float32,
                                 device=self.device)
            target = torch.as_tensor(
                ((np.asarray(task.y)[query] - center) / scale)[None],
                dtype=torch.float32, device=self.device,
            )
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.network(Xc, yc, Xq), target)
            loss.backward()
            optimizer.step()
        self._trained = True

    @torch.no_grad()
    def suggest(self, X: np.ndarray, y: np.ndarray,
                candidates: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Call meta_train before suggest")
        self.network.eval()  # target inference không cập nhật gradient.
        center, scale = float(np.mean(y)), max(float(np.std(y)), 1e-6)
        Xc = torch.as_tensor(X[None], dtype=torch.float32, device=self.device)
        yc = torch.as_tensor(((y - center) / scale)[None], dtype=torch.float32,
                             device=self.device)
        Xq = torch.as_tensor(candidates[None], dtype=torch.float32,
                             device=self.device)
        predicted_loss = self.network(Xc, yc, Xq).squeeze(0).cpu().numpy()
        return candidates[int(np.argmin(predicted_loss))]
