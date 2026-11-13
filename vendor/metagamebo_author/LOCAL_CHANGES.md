# Public source and local patches

Source: https://github.com/Allen0497/MetaGameBO
Commit: 489213ddcfbfe275a8b443c0d8b136ef6c36ad2c
License: AGPL-3.0, copyright headers retained. See LICENSE and 3rd-party-licenses.txt.
PROVENANCE.json records original and adapted SHA256 for each upstream file.

The adapter uses the author's MetaGameBO class with both DRO/k-DPP disabled.
The public run_forrester.py refers to an undefined MALIBO symbol; no fallback
to the previous local MALIBO-style model is used.

Local patches:

1. Package-relative imports to avoid collisions with this project's modules.
2. MetaBLOR inherits ClassifierMixin before BaseEstimator and supplies classes_
   for current sklearn's GradientBoostingClassifier. Remove the accidental tuple
   in input_dim (trailing comma) so BaseEstimator's parameter contract is valid.
3. Initialize embedding count using max task id rather than the final row.
4. Keep the random 80/20 row split, expose split datasets and MetaLoss for audit.
5. At each epoch end, audit complete tasks in eval mode, restore the original
   training mode and RNG state, and report macro BCE, micro BCE, exact top-k CVaR, separately
   from regularization and the combined objective. Original minibatch objective
   logging remains in the upstream text log; use classifier_bce.csv for BCE.

The neural architecture, original training updates, noise, loss/regularizer,
learning rate schedule, task adaptation and gradient boosting are retained.
The upstream standalone calculate_cvar helper uses KDE; the training loop's
published top-k rule and project metrics.py are used for reported metrics.

upstream.patch records all changes to upstream Python files. Newly added
adapter/audit code lives in models/malibo.py and models/author_metrics.py.
