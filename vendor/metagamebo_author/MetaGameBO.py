import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import parameterspace as ps
from blackboxopt.evaluation import Evaluation, EvaluationSpecification
from sklearn.ensemble import GradientBoostingClassifier
from scipy.optimize import fsolve

from .meta_learning.meta_classifier import MetaBLOR
from .utils import manual_seed

class DiagonalKDPP:
    """
    Approximate k-DPP implementation based on Barthelmé et al. (2018) with enhanced numerical stability
    """
    def __init__(self, n, k, seed=None):
        """Initialize diagonal k-DPP"""
        self.n = n
        self.k = min(k, n)  # Ensure k does not exceed n
        self.rng = np.random.RandomState(seed)
        
        # Initialize with uniform weights
        self.weights = np.ones(n)
        
        # Initialize theta parameter
        self.theta = 0.0
        self._normalize()
    
    def _normalize(self):
        """Recalculate theta to ensure sum of marginal probabilities equals k"""
        # Safe computation of sigmoid function
        def safe_sigmoid(x, w):
            if x > 0:
                z = np.exp(-x)
                return w / (w + z)
            else:
                z = np.exp(x)
                return w * z / (w * z + 1)
        
        # Sum of marginal probabilities function
        def margin_sum(x):
            total = 0.0
            for w in self.weights:
                if np.isfinite(w) and w > 0:
                    total += safe_sigmoid(x, w)
            return total - self.k
        
        # Binary search to solve for theta
        try:
            low, high = -20.0, 20.0
            for _ in range(50):
                mid = (low + high) / 2
                sum_val = margin_sum(mid)
                
                if abs(sum_val) < 1e-4:
                    self.theta = mid
                    return
                
                if sum_val < 0:
                    low = mid
                else:
                    high = mid
            
            self.theta = (low + high) / 2
            
        except Exception as e:
            print(f"Normalization warning: {e}, using default value")
            self.theta = 0.0
    
    @property
    def marginals(self):
        """Get approximate marginal probabilities of k-DPP"""
        # Clip weights to prevent outliers
        safe_weights = np.clip(self.weights, 1e-10, 1e10)
        
        # Stable computation of marginal probabilities
        probs = np.zeros(self.n)
        for i, w in enumerate(safe_weights):
            # Use numerically stable sigmoid computation
            if self.theta > 0:
                z = np.exp(-self.theta)
                probs[i] = w / (w + z)
            else:
                z = np.exp(self.theta)
                probs[i] = w * z / (w * z + 1)
        
        # Ensure valid probability distribution
        probs = np.clip(probs, 0, 1)
        
        # If sum of probabilities is 0, return uniform distribution
        if np.sum(probs) < 1e-10:
            return np.ones(self.n) / self.n
            
        return probs
    
    def update_weights(self, indices, losses, learning_rate=0.1):
        """Update weights based on observed losses"""
        # Get current marginal probabilities
        probs = self.marginals
        
        # Clip loss values to prevent extreme values
        clipped_losses = np.clip(losses, -10, 10)
        
        # Normalize learning rate
        scaled_lr = learning_rate / np.sqrt(1 + len(indices))
        
        for idx, loss in zip(indices, clipped_losses):
            if idx >= len(self.weights):
                continue
                
            # Ensure marginal probability is not zero
            q_i = max(probs[idx], 1e-5)
            
            # Calculate update factor with clipping to prevent overflow
            update_factor = np.exp(min(scaled_lr * self.k * loss / q_i, 10))
            
            # Update weights
            if np.isfinite(update_factor):
                self.weights[idx] *= update_factor
        
        # Clip weights to ensure numerical stability
        self.weights = np.clip(self.weights, 1e-10, 1e10)
        
        # Re-normalize
        self._normalize()
    
    def sample(self, batch_size=1):
        """Sample from k-DPP marginal distribution"""
        probs = self.marginals
        
        # Normalize to probability distribution
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        else:
            probs = np.ones_like(probs) / len(probs)
        
        try:
            # Multinomial sampling
            indices = self.rng.choice(
                self.n, 
                size=min(batch_size, self.n),
                replace=True,
                p=probs
            )
            return indices
        except:
            # Use random sampling when error occurs
            return self.rng.choice(self.n, size=min(batch_size, self.n))

class RandomSearchSampler:
    def __init__(self, search_space: ps.ParameterSpace, seed: Optional[int] = None):
        """Random search for configurations"""
        self.search_space = search_space.copy()
        self.seed = seed
        self.search_space.seed(seed)

    def __call__(self) -> EvaluationSpecification:
        return EvaluationSpecification(configuration=self.search_space.sample())


class MetaGameBO:
    """Meta-learning for likelihood-free Bayesian optimization
    Args:
        search_space: Search space of the problem
        seed: Random seed
        gamma: Hyperparameter in LFBO, which is the percentile of configurations
            that are considered as good. It controls the exploration-exploitation,
            default is 0.33.
        num_samples_acquisition_function: Number of random samples for optimizing the
            acquisition function.
    """
    def __init__(
        self,
        search_space: ps.ParameterSpace,
        seed: Optional[int] = None,
        gamma: float = 1/3,
        num_samples_acquisition_function: int = 5120,
        use_kdpp: bool = True,  # Whether to use k-DPP
        alpha: float = 0.8,     # Alpha parameter for CVaR
        kdpp_lr: float = 0.1,   # k-DPP learning rate
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
        """Assigning class labels to observations with the threshold tau computed via gamma.
        Weights are computed by the utility function max(tau - y, 0) from expected improvement (EI).   
        Args:
            X: Observations inputs
            y: Observations values
            gamma: Hyperparameter in LFBO, which is the percentile of configurations
                that are considered as good. It controls the exploration-exploitation,
                default is 0.33.
            weight_type: Determine which utility function to use. EI uses max(tau - y, 0) while PI
                uses 1(tau - y > 0).
        Returns:
            X: Observations inputs
            z: Class labels
            w: Weights calculated by utility function
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
        """
        Apply k-DPP weight adjustment without changing input data shape
        
        Args:
            x: Feature vectors
            z: Class labels
            w: Original weights
            y: Original loss values
            
        Returns:
            Adjusted weights, maintaining the same shape as input weights
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
        """Meta-learning on meta-data, corresponds to the meta-learning part in Algorithm 1."""
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
        """Train optimizer on new observations from the target task.
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
        """
        Propose new configuration x to evaluate.

        1. First proposal is proposed according to the maximum of the meta-learned
        acquisition function.
        2. After the first evaluation, the proposed candidate is selected based on
        the Thompson sample of the acquisition function. 
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
        """Predict the utility of given input X (acquisition function value for given points)"""
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
        """
        Optimizing the acquisition function.

        As pointed out by the BORE paper, random search for optimizing the
        acquisition function generated by tree-based methods is better than
        using evolutionary algorithm. Therefore we use random search for
        optimizing the acquisition function. 
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
        return any(np.array_equal(x, new_X) for x in self.X)

    def report(self, evaluations: Union[Evaluation, Iterable[Evaluation]]) -> None:
        """Report observations to the optimizer"""
        _evals = [evaluations] if isinstance(evaluations, Evaluation) else evaluations

        for evaluation in _evals:
            self._report(evaluation)

    def _report(self, evaluation: Evaluation) -> None:
        new_X = np.atleast_2d(self.search_space.to_numerical(evaluation.configuration))
        new_loss = evaluation.objectives["loss"]

        if len(self.X) > 0:
            self.X = np.vstack([self.X, new_X])
            self.losses = np.concatenate([self.losses, [new_loss]])
        else:
            self.X = new_X
            self.losses = np.array([new_loss])

    def save(self, save_dir, override=False):
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
        load_dir = Path(load_dir)
        self.classifier = self.classifier.load(load_dir, **kwargs)
        return self