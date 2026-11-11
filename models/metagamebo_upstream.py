"""Copy of the upstream MetaGameBO optimizer (MALIBO + CVaR training + approximate k-DPP).

Requires the upstream ``meta_learning`` and ``utils`` modules on the path.
"""
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import parameterspace as ps
from blackboxopt.evaluation import Evaluation, EvaluationSpecification
from sklearn.ensemble import GradientBoostingClassifier

from components.kdpp import DiagonalKDPP
from components.upstream import RandomSearchSampler
from meta_learning.meta_classifier import MetaBLOR  # upstream, needs MetaGameBO/ on sys.path
from utils import manual_seed  # upstream


class MetaGameBO:
    """Meta-learned likelihood-free BO (MALIBO) with optional CVaR and k-DPP.

    Args:
        search_space: Search space of the problem.
        seed: Random seed.
        gamma: Quantile of configurations labelled "good"; controls
            exploration vs. exploitation.
        num_samples_acquisition_function: Random candidates used to maximise
            the acquisition function.
        use_kdpp: Whether to adjust sample weights with the k-DPP.
        alpha: k-DPP subset fraction, ``k = alpha * n`` (named alpha upstream).
        kdpp_lr: k-DPP weight learning rate.
        **classifier_kwargs: Passed to ``MetaBLOR`` (e.g. ``num_layers``,
            ``num_features``, ``num_hidden_units``, ``device``).
    """

    def __init__(
            self,
            search_space: ps.ParameterSpace,
            seed: Optional[int] = None,
            gamma: float = 1 / 3,
            num_samples_acquisition_function: int = 5120,
            use_kdpp: bool = True,  # Whether to use k-DPP
            alpha: float = 0.8,  # Alpha parameter for CVaR
            kdpp_lr: float = 0.1,  # k-DPP learning rate
            **classifier_kwargs,
    ):

        self.search_space = search_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.num_samples_acquisition_function = num_samples_acquisition_function
        self.random_sampler = RandomSearchSampler(search_space, seed=seed)

        # Added k-DPP related parameters
        self.use_kdpp = use_kdpp
        self.alpha = alpha
        self.kdpp_lr = kdpp_lr
        self.kdpp = None  # Initialize later

        # meta-learning classifier
        self.classifier = MetaBLOR(
            input_dim=len(search_space),
            output_dim=1,
            **classifier_kwargs
        )
        # gradient boosting classifier
        self.classifier_gb = None

        # saved previous observations
        self.X = []
        self.losses = []

    @staticmethod
    def _reweight_samples(X, y, gamma, weight_type='ei'):
        """Label observations with threshold ``tau`` (gamma-quantile) and weight them.

        Args:
            X: Observation inputs, shape (n, d).
            y: Observation values, shape (n, 1) or (n,).
            gamma: Quantile of observations labelled "good".
            weight_type: ``"ei"`` weights by ``max(tau - y, 0)``; ``"pi"`` uses unit weights.

        Returns:
            Tuple ``(x, z, w)``: inputs, float class labels and weights.
        """
        tau = np.quantile(np.unique(y), q=gamma)
        z = np.less(y, tau)

        if weight_type == 'ei' and len(X) > 1:
            z_idx = z.squeeze()

            x1, z1 = X[z_idx], z[z_idx]
            x0, z0 = X, np.zeros_like(z)

            w1 = (tau - y)[z_idx]
            w1 = w1 / np.mean(w1) if len(w1) else w1
            w0 = 1 - z0

            x = np.concatenate([x1, x0], axis=0)
            z = np.concatenate([z1, z0], axis=0)
            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / np.mean(w)

        elif weight_type == 'pi' or len(X) == 1:
            x = X
            w = np.ones_like(z)

        return x, z.astype(float), w.squeeze()

    def _apply_kdpp_weights(self, x, z, w, y):
        """Multiply sample weights by k-DPP marginals updated from normalised losses.

        Args:
            x: Inputs, shape (m, d).
            z: Class labels, shape (m,).
            w: Original weights.
            y: Original loss values (``1 - z`` is used if lengths differ).

        Returns:
            Adjusted weights with the same shape as ``w`` (``w`` unchanged if
            the k-DPP is disabled or fails).
        """
        # If k-DPP is not enabled or too few samples, return original weights directly
        if not self.use_kdpp or len(x) <= 1:
            return w

        try:
            # Ensure w is a 1D array for processing
            original_shape = w.shape
            w_1d = w.squeeze()

            # If k-DPP has not been initialized yet, initialize now
            k = max(1, int(self.alpha * len(x)))
            if self.kdpp is None or self.kdpp.n != len(x) or self.kdpp.k != k:
                self.kdpp = DiagonalKDPP(n=len(x), k=k, seed=self.seed)

            # Prepare normalized losses for k-DPP
            norm_losses = np.copy(y)
            if len(norm_losses) != len(x):
                # If lengths don't match, use z as loss indicator
                norm_losses = 1.0 - z.astype(float)
            else:
                # Normalize loss values to [0,1] range
                norm_min = np.min(norm_losses)
                norm_max = np.max(norm_losses)
                if norm_max > norm_min:
                    norm_losses = (norm_losses - norm_min) / (norm_max - norm_min)

            # Update k-DPP weights using losses
            self.kdpp.update_weights(np.arange(len(x)), norm_losses, self.kdpp_lr)

            # Get k-DPP marginal probabilities
            kdpp_marginals = self.kdpp.marginals

            # Adjust weights using marginal probabilities
            w_adjusted = w_1d * kdpp_marginals

            # Avoid zero weights
            w_adjusted[w_adjusted < 1e-10] = 1e-10

            # Re-normalize
            mean_w = np.mean(w_adjusted)
            if mean_w > 0:
                w_adjusted = w_adjusted / mean_w

            # Restore original shape
            if len(original_shape) > 1:
                w_adjusted = w_adjusted.reshape(original_shape)

            return w_adjusted

        except Exception as e:
            print(f"k-DPP processing warning: {e}, using original weights")
            return w

    def meta_fit(
            self,
            meta_data: Optional[Dict[str, List[Evaluation]]] = None,
            num_epochs: int = 2048,
            batch_size: int = 256,
            meta_dir: Optional[str] = None,
            override: bool = True,
            **train_config
    ):
        """Meta-train the classifier on meta-data (meta-learning part of Algorithm 1).

        Args:
            meta_data: Dict mapping task uid to its list of evaluations.
            num_epochs: Training epochs.
            batch_size: Batch size (number of tasks).
            meta_dir: If given, save the trained model there.
            override: Overwrite an existing ``meta_dir``.
            **train_config: Passed to the classifier (e.g. ``use_dro``, ``alpha``,
                ``use_kdpp``, ``kdpp_alpha``, ``kdpp_lr``, ``seed``).
        """
        # Get k-DPP related parameters
        self.use_kdpp = train_config.get('use_kdpp', self.use_kdpp)
        self.alpha = train_config.get('kdpp_alpha', self.alpha)
        self.kdpp_lr = train_config.get('kdpp_lr', self.kdpp_lr)

        converted_meta_data = dict()
        for task_uid, evaluations in meta_data.items():
            X = np.array([self.search_space.to_numerical(e.configuration) for e in evaluations])
            Y = np.array([e.objectives["loss"] for e in evaluations]).reshape(-1, self.classifier.output_dim)

            # First apply original reweighting method
            x, z, w = self._reweight_samples(X, Y, self.gamma, weight_type='ei')

            # Then apply k-DPP adjustment (if enabled)
            if self.use_kdpp:
                w = self._apply_kdpp_weights(x, z, w, Y)

            # Ensure w is a 2D tensor
            w_reshaped = w.reshape(-1, 1) if w.ndim == 1 else w
            converted_meta_data[task_uid] = {"X": x, "Y": z, "w": w_reshaped}

        self.classifier.meta_fit(
            converted_meta_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            **train_config
        )
        if meta_dir is not None:
            self.save(meta_dir, override=override)

    def _update_model(self, weight_type='ei'):
        """Fit the model on the target-task observations.

        Fits gradient boosting on top of the meta-classifier when each class has
        at least 2 samples, otherwise the meta-classifier alone.

        Args:
            weight_type: ``"ei"`` or ``"pi"``, see ``_reweight_samples``.
        """
        # Apply original reweighting method
        x, z, w = self._reweight_samples(
            self.X,
            self.losses,
            self.gamma,
            weight_type
        )

        # Apply k-DPP adjustment (if enabled)
        if self.use_kdpp:
            w = self._apply_kdpp_weights(x, z, w, self.losses)

        # Train classifier
        if sum(z == 0.) >= 2 and sum(z == 1.) >= 2:
            # train a GB with early stopping to estimate n_estimators
            classifier_gb = GradientBoostingClassifier(
                init=self.classifier,
                validation_fraction=0.3,
                n_iter_no_change=5,
            )
            classifier_gb.fit(x, z, sample_weight=w)
            # train on full dataset with estimated n_estimators
            self.classifier_gb = GradientBoostingClassifier(
                init=self.classifier,
                n_estimators=classifier_gb.n_estimators_
            )
            # train classifier weighted by utility
            self.classifier_gb.fit(x, z, sample_weight=w)
        else:
            self.classifier.fit(x, z, sample_weight=w)

    def generate_evaluation_specification(self, sampling='thompson_sampling', **kwargs):
        """Propose the next configuration.

        The first proposal maximises the meta-learned acquisition function; later
        ones maximise a Thompson sample of it. Duplicates are replaced by a
        random configuration.

        Args:
            sampling: Acquisition sampling mode after the first step.
            **kwargs: Passed to ``_sample_new_config``.

        Returns:
            Evaluation specification of the proposed configuration.
        """

        if len(self.X) == 0:
            configuration = self._sample_new_config(sampling="max", **kwargs)
        else:
            self._update_model()
            configuration = self._sample_new_config(sampling=sampling, **kwargs)
        eval_spec = EvaluationSpecification(configuration=configuration)

        # If new proposed configuration is duplicate, we use a random sample as substitution
        if self._is_duplicate(self.search_space.to_numerical(eval_spec.configuration)):
            eval_spec = self.random_sampler()

        return eval_spec

    def observe_and_suggest(
            self,
            X_obs,
            y_obs,
            X_pen=None,
            sampling='thompson_sampling',
            seed=None,
            **kwargs
    ):
        """Set observations and propose the next point (stateless API used by HPO-B).

        Args:
            X_obs: Observed inputs, shape (n, d).
            y_obs: Observed losses, shape (n,) or (n, 1).
            X_pen: Pending candidates for the discrete case; if given, an index is returned.
            sampling: Acquisition sampling mode.
            seed: Seed for Thompson sampling; random if ``None``.
            **kwargs: Passed to ``predict`` / ``_sample_new_config``.

        Returns:
            Index into ``X_pen`` if given, else the new point in numerical form.
        """
        self.X = X_obs
        self.losses = y_obs.ravel()
        rng = np.random.default_rng(seed)
        if seed is None:
            seed = rng.integers(32767, size=1)
            self.seed = seed

        # X_pen not None for HPOB discrete case
        if X_pen is not None:
            if len(np.unique(self.losses)) <= 1:
                scores = self.predict(X_pen, sampling='max', seed=seed, **kwargs)
                candidate = np.argmax(scores)
                return candidate
            else:
                self._update_model()
                scores = self.predict(X_pen, sampling=sampling, seed=seed, **kwargs)
                candidate = np.argmax(scores)
                return candidate
        else:
            if len(np.unique(self.losses)) <= 1:
                configuration = self._sample_new_config(sampling="max", **kwargs)
                new_x = self.search_space.to_numerical(configuration)
                return new_x
            else:
                self._update_model()
                configuration = self._sample_new_config(sampling=sampling, **kwargs)
                # If new proposed configuration is duplicate, then we randomly sample one
                if self._is_duplicate(self.search_space.to_numerical(configuration)):
                    configuration = self.search_space.sample()
                new_x = self.search_space.to_numerical(configuration)
                return new_x

    def predict(self, X, sampling="thompson_sampling", seed=None):
        """Acquisition values (probability of "good") at ``X``.

        Args:
            X: Inputs, shape (n, d).
            sampling: ``"thompson_sampling"`` or ``"max"``; ignored once GB is fitted.
            seed: Seed for reproducible Thompson samples.

        Returns:
            Values, shape (n,).
        """
        # In order to visualize and reproduce the Thompson sampling result later,
        # we use a random seed to generate samples and save the random seed.
        with manual_seed(seed):
            if self.classifier_gb is None:
                class_probability = self.classifier.predict(X, sampling=sampling)
            else:
                class_probability = self.classifier_gb.predict_proba(X)[:, 1]
        return class_probability

    def _sample_new_config(
            self,
            sampling,
            seed: Optional[int] = None,
            **kwargs
    ):
        """Maximise the acquisition function by random search (as in BORE).

        Args:
            sampling: Acquisition sampling mode.
            seed: Seed for candidates and Thompson samples; random if ``None``.
            **kwargs: Passed to ``predict``.

        Returns:
            Best configuration among ``num_samples_acquisition_function`` candidates.
        """
        rng = np.random.default_rng(seed)
        samples = rng.random([
            self.num_samples_acquisition_function, len(self.search_space)
        ])
        if seed is None:
            # generate random seed to produce Thompson samples
            seed = rng.integers(32767, size=1)
            self.seed = seed

        af_samples = self.predict(samples, sampling=sampling, seed=seed, **kwargs)
        best_vector = samples[np.argmax(af_samples)]

        return self.search_space.from_numerical(best_vector)

    def _is_duplicate(self, new_X):
        """Return True if ``new_X`` equals an already observed input."""
        return any(np.array_equal(x, new_X) for x in self.X)

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        """Report one or more evaluations to the optimizer.

        Args:
            evaluations: An evaluation or an iterable of evaluations.
        """
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        for evaluation in _evals:
            self._report(evaluation)

    def _report(self, evaluation: Evaluation) -> None:
        """Append one evaluation to ``X`` and ``losses``."""
        new_X = np.atleast_2d(self.search_space.to_numerical(evaluation.configuration))
        new_loss = evaluation.objectives["loss"]

        if len(self.X) > 0:
            self.X = np.vstack([self.X, new_X])
            self.losses = np.concatenate([self.losses, [new_loss]])
        else:
            self.X = new_X
            self.losses = np.array([new_loss])

    def save(self, save_dir, override=False):
        """Save the meta-classifier.

        Args:
            save_dir: Target directory.
            override: Replace the directory if it exists.

        Raises:
            NotADirectoryError: If ``save_dir`` is an existing file.
            FileExistsError: If ``save_dir`` exists and ``override`` is False.
        """
        save_dir = Path(save_dir)
        if save_dir.exists():
            if not save_dir.is_dir():
                raise NotADirectoryError(
                    f'The directory to which to save the model is an existing file: '
                    f'"{save_dir}"'
                )
            if not override:
                raise FileExistsError(
                    f'The directory to which to save the model, already exists: '
                    f'"{save_dir}"'
                )
            else:
                shutil.rmtree(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        self.classifier.save(save_dir)

    def load(self, load_dir, **kwargs):
        """Load the meta-classifier from ``load_dir``; returns self.

        Args:
            load_dir: Directory written by ``save``.
            **kwargs: Passed to the classifier's ``load``.
        """
        load_dir = Path(load_dir)
        self.classifier = self.classifier.load(load_dir, **kwargs)
        return self