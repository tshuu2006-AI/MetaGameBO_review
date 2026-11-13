"""Local Forrester-prior PFN with masked attention, Normal predictive and EI.

This reproduction is not the pretrained PFNs4BO checkpoint used by the paper.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .numerics import seed_everything, require_finite


def expected_improvement(mean, std, best):
    """Analytic E[max(best-Y, 0)] for minimisation under a Normal predictive."""
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    safe = std.clamp_min(torch.finfo(mean.dtype).eps)
    improvement = best - mean
    z = improvement / safe
    ei = improvement * torch.special.ndtr(z) + safe * torch.exp(-.5*z.square()) / math.sqrt(2*math.pi)
    return torch.where(std > 0, ei.clamp_min(0), improvement.clamp_min(0))


class InContextSurrogate(nn.Module):
    def __init__(self, input_dim, width=64, layers=2, heads=4):
        super().__init__()
        self.input = nn.Linear(input_dim + 2, width)
        block = nn.TransformerEncoderLayer(width, heads, width*2, dropout=0., batch_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, 2)

    @staticmethod
    def attention_mask(context_n, query_n, device=None):
        # True blocks attention: context sees context, queries see context only.
        mask = torch.ones(context_n+query_n, context_n+query_n, dtype=torch.bool, device=device)
        mask[:, :context_n] = False
        return mask

    def forward(self, context_x, context_y, candidates):
        b, c, _ = context_x.shape
        q = candidates.shape[1]
        if c < 1 or q < 1:
            raise ValueError('nonempty context and queries required')
        observed = torch.ones(b, c, 1, device=context_x.device, dtype=context_x.dtype)
        pending = torch.zeros(b, q, 1, device=context_x.device, dtype=context_x.dtype)
        tokens = torch.cat((torch.cat((context_x, context_y.unsqueeze(-1), observed), -1),
                            torch.cat((candidates, pending, pending), -1)), 1)
        encoded = self.transformer(self.input(tokens), mask=self.attention_mask(c, q, tokens.device))
        raw = self.output(encoded[:, c:])
        return torch.distributions.Normal(raw[..., 0], F.softplus(raw[..., 1]) + 1e-3)


class PFNAgent:
    def __init__(self, bounds, seed=44, meta_epochs=100, context_size=31,
                 device='cpu', task_batch_size=10):
        seed_everything(seed)
        self.bounds, self.seed, self.device = np.asarray(bounds), seed, device
        self.meta_epochs, self.context_size, self.task_batch_size = meta_epochs, context_size, task_batch_size
        self.rng = np.random.default_rng(seed)
        self.network = InContextSurrogate(len(bounds)).to(device)
        self.training_diagnostics = []
        self._trained = False

    def _tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def meta_train(self, tasks):
        if not tasks or any(len(t.y) < 2 for t in tasks):
            raise ValueError('tasks with at least two samples required')
        seed_everything(self.seed)
        self.rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=1e-3)
        self.training_diagnostics = []
        self.network.train()
        for epoch in range(self.meta_epochs):
            order = self.rng.permutation(len(tasks))
            losses, contexts = [], []
            for start in range(0, len(tasks), self.task_batch_size):
                batch = [tasks[i] for i in order[start:start+self.task_batch_size]]
                n = min(len(t.y) for t in batch)
                c = int(self.rng.integers(1, min(self.context_size, n-1)+1))
                xc, yc, xq, targets = [], [], [], []
                for task in batch:
                    idx = self.rng.permutation(len(task.y))[:n]
                    context, query = idx[:c], idx[c:]
                    cy = np.asarray(task.y)[context]
                    center, scale = cy.mean(), max(float(cy.std()), 1.)
                    xc.append(task.X[context]); yc.append((cy-center)/scale)
                    xq.append(task.X[query]); targets.append((np.asarray(task.y)[query]-center)/scale)
                optimizer.zero_grad(set_to_none=True)
                distribution = self.network(self._tensor(xc), self._tensor(yc), self._tensor(xq))
                loss = -distribution.log_prob(self._tensor(targets)).mean()
                require_finite(loss, 'PFN NLL')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach())); contexts.append(c)
            self.training_diagnostics.append({'epoch': epoch+1, 'train_gaussian_nll': float(np.mean(losses)),
                                              'optimizer_steps': len(losses), 'tasks_seen': len(order),
                                              'context_lengths': contexts})
        self.network.eval()
        self._trained = True

    @torch.no_grad()
    def predict_distribution(self, X, y, candidates):
        if not self._trained:
            raise RuntimeError('Call meta_train before predict')
        self.network.eval()
        y = np.asarray(y)
        center, scale = float(y.mean()), max(float(y.std()), 1.)
        d = self.network(self._tensor(np.asarray(X)[None]), self._tensor(((y-center)/scale)[None]),
                         self._tensor(np.asarray(candidates)[None]))
        return torch.distributions.Normal(d.mean.squeeze(0)*scale+center, d.stddev.squeeze(0)*scale)

    @torch.no_grad()
    def suggest(self, X, y, candidates):
        d = self.predict_distribution(X, y, candidates)
        ei = expected_improvement(d.mean, d.stddev, float(np.min(y)))
        require_finite(ei, 'PFN EI')
        return candidates[int(torch.argmax(ei))]
