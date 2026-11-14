"""Meta-training of the meta-classifier (MALIBO; MetaGameBO Algorithm 1 via selectors).

Each iteration: per-task losses -> a *task selector* picks the tasks of the
gradient step -> a *sample selector* picks the samples and their gradient
weights -> one Adam step -> the sample selector may update its weights.

MALIBO = ``AllTasks`` + ``AllSamples`` (ERM with the Eq. 3 utility weights).
MetaGameBO plugs in its own selectors (``models.metagamebo_remake.game``).

Evaluation (paper Table 1, as stated by the authors and computed in the
author's ``meta_learning/trainer.py``): the reweighted samples of all tasks are
pooled and split at random 80/20 *by sample*; training uses the 80%. A task's
loss is ``sum_i w_i BCE(p_i, z_i) / n_task`` and CVaR averages the
``max(1, int(M (1 - alpha)))`` largest task losses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from models.base import cvar
from models.lfbo.weighting import reweight_samples
from models.malibo.meta_net import DT, MetaNet, ks_cov_loss, reg_scales


@dataclass
class MetaTrainConfig:
    """Settings of the meta-training loop.

    Attributes:
        iters: Number of gradient steps.
        lr: Adam learning rate.
        gamma: Quantile threshold of ``reweight_samples``.
        weighting: Eq. (3) scheme (``"lfbo"`` as in MALIBO/author code, or ``"paper"``).
        emb_weight: Weight of the embedding regulariser.
        alpha: CVaR level used for logging (0.8 = worst 20% of tasks).
        val_frac: Fraction of the pooled reweighted samples held out (author: 0.2).
        max_per_task: Subsample tasks to at most this many points (None = all).
        seed: Seed of torch, numpy and the train/validation split.
        net: Keyword arguments of ``MetaNet``.
    """
    iters: int = 2000
    lr: float = 1e-3
    gamma: float = 1 / 3
    weighting: str = "lfbo"
    emb_weight: float = 0.1
    alpha: float = 0.8
    val_frac: float = 0.2
    max_per_task: Optional[int] = None
    seed: int = 0
    net: dict = field(default_factory=dict)


@dataclass
class MetaData:
    """Reweighted, pooled meta-training samples and their train/validation split.

    Attributes:
        X: Inputs (unit cube), shape (n, d).
        lab: Binary labels, shape (n,).
        w: Eq. (3) utility weights, shape (n,).
        tid: Task index 1..M of every sample, shape (n,).
        src: Index of the source point inside its task, shape (n,).
        is_val: True for held-out samples, shape (n,).
        n_tasks: Number of tasks M.
    """
    X: np.ndarray
    lab: np.ndarray
    w: np.ndarray
    tid: np.ndarray
    src: np.ndarray
    is_val: np.ndarray
    n_tasks: int


def prepare_meta_data(tasks: Sequence, gamma: float = 1 / 3, scheme: str = "lfbo",
                      val_frac: float = 0.2, seed: int = 0) -> MetaData:
    """Reweight every task, pool the samples and split them by sample (author protocol).

    Args:
        tasks: Objects with ``X`` (n_i, d) in the unit cube and ``y`` (n_i,).
        gamma: Quantile threshold.
        scheme: Eq. (3) scheme.
        val_frac: Held-out fraction; every task keeps at least one training sample.
        seed: Seed of the split.
    """
    xs, labs, ws, tids, srcs = [], [], [], [], []
    for i, t in enumerate(tasks):
        x, lab, w, idx = reweight_samples(t.X, t.y, gamma, scheme, return_index=True)
        xs.append(x); labs.append(lab); ws.append(w); srcs.append(idx)
        tids.append(np.full(len(x), i + 1))
    tid = np.concatenate(tids)
    is_val = np.zeros(len(tid), bool)
    if val_frac > 0:
        perm = np.random.default_rng([seed, 1]).permutation(len(tid))
        is_val[perm[:int(len(tid) * val_frac)]] = True
        for i in range(len(tasks)):
            own = np.flatnonzero(tid == i + 1)
            if is_val[own].all():
                is_val[own[0]] = False
    return MetaData(np.concatenate(xs), np.concatenate(labs), np.concatenate(ws), tid,
                    np.concatenate(srcs), is_val, len(tasks))


def weighted_bce(p: np.ndarray, lab: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-sample ``w * BCE(p, lab)`` (the Table-1 loss terms)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return -w * (lab * np.log(p) + (1 - lab) * np.log(1 - p))


@dataclass
class TrainLog:
    """Training curves, recorded every ``log_every`` iterations and at the last one.

    Attributes:
        iteration: Iteration of each record.
        hard_frac: Fraction of tasks used by the gradient step.
        avg_task_loss / cvar_task_loss: Mean / CVaR of training task losses.
        val_avg_task_loss / val_cvar_task_loss: Same on the held-out samples (Table 1).
        final: Losses of the trained network (``train_avg``, ``train_cvar``,
            ``val_avg``, ``val_cvar``) and ``val_task_losses`` (list, length M).
    """
    iteration: list = field(default_factory=list)
    hard_frac: list = field(default_factory=list)
    avg_task_loss: list = field(default_factory=list)
    cvar_task_loss: list = field(default_factory=list)
    val_avg_task_loss: list = field(default_factory=list)
    val_cvar_task_loss: list = field(default_factory=list)
    final: dict = field(default_factory=dict)


@dataclass
class TrainSnapshot:
    """Training state at one logged iteration, passed to the optional callback."""
    it: int
    net: MetaNet
    task_losses: np.ndarray
    hard: np.ndarray
    X: np.ndarray
    labels: np.ndarray
    task_index: np.ndarray
    util_weights: np.ndarray
    sample_weights: np.ndarray
    val_task_losses: Optional[np.ndarray] = None
    xi: Optional[float] = None  # CVaR threshold of the task selector, if any


def per_task_loss(net: MetaNet, X, lab, w, tid, n_tasks) -> np.ndarray:
    """Mean weighted BCE of each task (eval mode); NaN for a task without samples. Shape (M,)."""
    was_training = net.training
    net.eval()
    with torch.no_grad():
        logit = net(X, tid)
        l = torch.nn.functional.binary_cross_entropy_with_logits(logit, lab, reduction="none") * w
        s = torch.zeros(n_tasks + 1, dtype=DT).index_add_(0, tid, l)
        c = torch.zeros(n_tasks + 1, dtype=DT).index_add_(0, tid, torch.ones_like(l))
    net.train(was_training)
    s, c = s[1:].numpy(), c[1:].numpy()
    return np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)


class AllTasks:
    """Task selector of MALIBO (ERM): every task in every step."""

    def select(self, task_losses: np.ndarray) -> np.ndarray:
        return np.arange(len(task_losses))


class AllSamples:
    """Sample selector of MALIBO: all samples of the selected tasks, utility weights."""

    def init(self, W_util: torch.Tensor, slices: list, X: torch.Tensor, rng) -> None:
        self.W, self.slices = W_util, slices

    @property
    def weights(self) -> torch.Tensor:
        return self.W

    def select(self, hard: np.ndarray):
        """Return ``(idx, gradient weights)`` for the selected tasks."""
        idx = np.concatenate([self.slices[i] for i in hard])
        return idx, self.W[idx]

    def update(self, hard, idx, per_sample_loss) -> None:
        """No sample-weight update."""


def meta_train(tasks: Sequence, cfg: MetaTrainConfig, task_selector=None, sample_selector=None,
               callback: Optional[Callable[[TrainSnapshot], None]] = None, log_every: int = 50,
               callback_every: Optional[int] = None):
    """Meta-train the classifier on ``tasks`` (inputs in the unit cube).

    Args:
        tasks: Objects with ``X`` (n_i, d) and ``y`` (n_i,).
        cfg: Training configuration.
        task_selector: Object with ``select(task_losses) -> indices``; default ``AllTasks``.
        sample_selector: Object with ``init``, ``select``, ``update`` and
            ``weights``; default ``AllSamples``.
        callback: Called with a ``TrainSnapshot`` every ``callback_every`` iterations.
        log_every: ``TrainLog`` period.
        callback_every: Callback period (default ``log_every``).

    Returns:
        ``(net, log)``: the trained network (eval mode) and its ``TrainLog``.
    """
    task_selector = task_selector or AllTasks()
    sample_selector = sample_selector or AllSamples()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    M = len(tasks)
    net = MetaNet(input_dim=tasks[0].X.shape[1], num_tasks=M, **cfg.net).to(DT)
    ks_w, cv_w = reg_scales(M, net.num_features, seed=cfg.seed)

    md = prepare_meta_data(tasks, cfg.gamma, cfg.weighting, cfg.val_frac, cfg.seed)
    tr = ~md.is_val
    slices, off = [], 0
    for i in range(M):
        n_i = int(np.sum(md.tid[tr] == i + 1))
        slices.append(np.arange(off, off + n_i)); off += n_i
    X = torch.tensor(md.X[tr], dtype=DT)
    LAB = torch.tensor(md.lab[tr], dtype=DT)
    W_util = torch.tensor(md.w[tr], dtype=DT)
    TID = torch.tensor(md.tid[tr], dtype=torch.long)
    val = None
    if md.is_val.any():
        v = md.is_val
        val = (torch.tensor(md.X[v], dtype=DT), torch.tensor(md.lab[v], dtype=DT),
               torch.tensor(md.w[v], dtype=DT), torch.tensor(md.tid[v], dtype=torch.long))
    sample_selector.init(W_util.clone(), slices, X, rng)

    cb_every = callback_every or log_every
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    log = TrainLog()
    for it in range(cfg.iters):
        L = per_task_loss(net, X, LAB, W_util, TID, M)
        hard = task_selector.select(L)
        last = it == cfg.iters - 1
        do_log = it % log_every == 0 or last
        do_cb = callback is not None and (it % cb_every == 0 or last)
        Lv = per_task_loss(net, *val, M) if val is not None and (do_log or do_cb) else None
        if do_log:
            log.iteration.append(it)
            log.hard_frac.append(len(hard) / M)
            log.avg_task_loss.append(float(L.mean()))
            log.cvar_task_loss.append(cvar(L, cfg.alpha))
            if Lv is not None:
                log.val_avg_task_loss.append(float(np.nanmean(Lv)))
                log.val_cvar_task_loss.append(cvar(Lv, cfg.alpha))
        if do_cb:
            callback(TrainSnapshot(it, net, L.copy(), np.array(hard), X.numpy(), LAB.numpy(), TID.numpy(),
                                   W_util.numpy().copy(), sample_selector.weights.numpy().copy(), Lv,
                                   getattr(task_selector, "xi", None)))

        idx, gw = sample_selector.select(hard)
        idx_t = torch.as_tensor(idx)
        logit = net(X[idx_t], TID[idx_t])
        per = torch.nn.functional.binary_cross_entropy_with_logits(logit, LAB[idx_t], reduction="none")
        task_of = TID[idx_t]
        cnt = torch.zeros(M + 1, dtype=DT).index_add_(0, task_of, torch.ones_like(per))
        pred_loss = ((per * gw) / cnt[task_of]).sum() / len(hard)  # mean over tasks of task means
        ks, cv = ks_cov_loss(net.emb[1:])
        loss = pred_loss + cfg.emb_weight * (ks_w * ks + cv_w * cv)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sample_selector.update(hard, idx, per.detach().numpy())

    L = per_task_loss(net, X, LAB, W_util, TID, M)
    log.final = dict(train_avg=float(L.mean()), train_cvar=cvar(L, cfg.alpha))
    if val is not None:
        Lv = per_task_loss(net, *val, M)
        log.final.update(val_avg=float(np.nanmean(Lv)), val_cvar=cvar(Lv, cfg.alpha),
                         val_task_losses=Lv.tolist())
    net.eval()
    return net, log
