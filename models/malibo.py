"""Adapter to the author's public MALIBO path (both game switches disabled)."""
import numpy as np
import parameterspace as ps
import torch
from blackboxopt import Evaluation

from vendor.metagamebo_author.MetaGameBO import MetaGameBO
from .numerics import seed_everything, require_finite


class MALIBOAgent:
    def __init__(self, bounds, seed=44, meta_epochs=100, task_batch_size=10,
                 device='cpu'):
        seed_everything(seed)
        self.bounds, self.seed = np.asarray(bounds), seed
        self.meta_epochs, self.task_batch_size = meta_epochs, task_batch_size
        space = ps.ParameterSpace()
        for i, bound in enumerate(bounds):
            space.add(ps.ContinuousParameter(name=f'x{i}', bounds=list(bound)))
        self.optimizer = MetaGameBO(space, seed=seed, use_kdpp=False,
                                   num_layers=5, num_features=50, num_hidden_units=64,
                                   device=device)
        self.training_diagnostics = []
        self._trained = False

    @property
    def model(self):
        return self.optimizer.classifier.meta_model

    def meta_train(self, tasks):
        if not tasks:
            raise ValueError('at least one source task required')
        seed_everything(self.seed)
        # Author task ids are 1-based; embedding 0 is reserved for target inference.
        metadata = {i+1: [Evaluation(configuration={f'x{j}': float(x[j]) for j in range(len(x))},
                                     objectives={'loss': float(y)})
                          for x, y in zip(task.X, task.y)] for i, task in enumerate(tasks)}
        self.optimizer.meta_fit(metadata, num_epochs=self.meta_epochs,
                                batch_size=self.task_batch_size, use_dro=False,
                                use_kdpp=False, seed=self.seed)
        self.training_diagnostics = self.model.training_diagnostics
        self.classifier_metrics = self.training_diagnostics[-1]
        self._trained = True

    def suggest(self, X, y, candidates):
        if not self._trained:
            raise RuntimeError('Call meta_train before suggest')
        index = self.optimizer.observe_and_suggest(
            np.asarray(X), np.asarray(y), X_pen=np.asarray(candidates),
            seed=int(torch.randint(32767, ()).item()))
        require_finite(candidates[index], 'MALIBO proposal')
        return candidates[index]
