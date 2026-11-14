"""MetaGameBO, re-implemented from the paper: MALIBO + the Stackelberg/CVaR task
leader and the k-DPP sample follower during meta-training (Algorithm 1).

The target-task phase (Laplace adapter, Thompson sampling, GB residual) is MALIBO's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.malibo.model import MALIBO
from models.malibo.training import AllSamples, AllTasks, MetaTrainConfig
from models.metagamebo_remake.game import CVaRTasks, KDPPSamples


@dataclass
class GameConfig:
    """Settings of the two levels of the game (each can be switched off for ablations).

    Attributes:
        task_level: ``"cvar"`` (leader) or ``"erm"`` (all tasks).
        sample_level: ``"kdpp"`` (follower, diagonal kernel), ``"kdpp_qd"`` or None.
        alpha_task: CVaR level of the leader.
        kdpp_frac: k-DPP subset size as a fraction of a task's samples.
        eta_s: Learning rate of Eq. (6).
        is_grad: Importance-weighted (``w / q``) k-DPP gradient.
    """
    task_level: str = "cvar"
    sample_level: Optional[str] = "kdpp"
    alpha_task: float = 0.8
    kdpp_frac: float = 0.4
    eta_s: float = 0.1
    is_grad: bool = True


class MetaGameBORemake(MALIBO):
    """MetaGameBO re-implementation.

    Args:
        bounds: Search box of shape (d, 2).
        seed: Random seed.
        cfg: Meta-training configuration (shared loop with MALIBO).
        game: ``GameConfig`` of the leader/follower.
        use_gb: Add the gradient-boosting residual on the target task.
        hessian: Laplace Hessian (paper: inverse Hessian -> ``"exact"``).
        iters: Override ``cfg.iters``.
    """

    name = "MetaGameBO (remake)"

    def __init__(self, bounds, seed: Optional[int] = None, cfg: Optional[MetaTrainConfig] = None,
                 game: Optional[GameConfig] = None, use_gb: bool = True, hessian: str = "exact",
                 iters: Optional[int] = None):
        super().__init__(bounds, seed, cfg, use_gb, hessian, iters)
        self.game = game or GameConfig()

    def _selectors(self):
        g = self.game
        ts = CVaRTasks(g.alpha_task) if g.task_level == "cvar" else AllTasks()
        if g.sample_level is None:
            ss = AllSamples()
        else:
            ss = KDPPSamples(g.kdpp_frac, g.eta_s, g.is_grad, "qd" if g.sample_level == "kdpp_qd" else "diag")
        return ts, ss
