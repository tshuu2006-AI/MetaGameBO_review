"""Optimizers compared in the experiments; each lives in its own sub-package
together with its components and training code.

Packages:
    base: ``BaseOptimizer`` interface, ``History``, ``Task``, ``cvar``.
    random_search: Random search.
    gp: GP with Expected Improvement.
    lfbo: LFBO and the Eq. (3) labelling (``reweight_samples``).
    pfn: PFNs4BO (pre-trained transformer).
    ablr: ABLR (meta-learned features + Bayesian linear regression).
    rgpe: RGPE (ranking-weighted GP ensemble).
    malibo: MALIBO (meta-network, meta-training loop, Laplace adapter).
    metagamebo_remake: MetaGameBO re-implemented from the paper.
    metagamebo_original: The authors' MetaGameBO code, wrapped.

``build(name, bounds, seed, **kw)`` creates an optimizer by its display name.
"""
from __future__ import annotations

METHODS = ["Random", "GP", "LFBO", "PFN", "ABLR", "RGPE", "MALIBO",
           "MetaGameBO (original)", "MetaGameBO (remake)"]


def build(name: str, bounds, seed: int = 0, **kw):
    """Create the optimizer called ``name`` (one of ``METHODS``); imports lazily."""
    if name == "Random":
        from models.random_search import RandomSearch as C
    elif name == "GP":
        from models.gp import GPEI as C
    elif name == "LFBO":
        from models.lfbo import LFBO as C
    elif name == "PFN":
        from models.pfn import PFN as C
    elif name == "ABLR":
        from models.ablr import ABLR as C
    elif name == "RGPE":
        from models.rgpe import RGPE as C
    elif name == "MALIBO":
        from models.malibo import MALIBO as C
    elif name == "MetaGameBO (original)":
        from models.metagamebo_original import MetaGameBOOriginal as C
    elif name == "MetaGameBO (remake)":
        from models.metagamebo_remake import MetaGameBORemake as C
    else:
        raise ValueError(f"unknown method {name!r}; choose from {METHODS}")
    return C(bounds, seed=seed, **kw)
