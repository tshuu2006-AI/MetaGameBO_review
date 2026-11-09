# Reproducibility notes

## Scope

The implementation follows the mechanisms named in the MetaGameBO paper and
the requested study protocol. It is an independent modular implementation,
not a vendored copy of the authors' AGPL repository. Exact numerical matching
of a paper table additionally requires its task splits, PFN checkpoint/prior,
architecture, optimizer schedule, and all random seeds; no published result is
hard-coded here.

## Operational definitions

- All problems are minimisation problems.
- CVaR at confidence `alpha` is the arithmetic mean of observed losses at or
  above the empirical `alpha` quantile (upper/worst tail).
- `f_min` and `f_max` are computed on a dense grid of the latent/noiseless
  Forrester task. Observation shocks are deliberately not folded into bounds.
- Task-level MetaGameBO estimates its VaR threshold with a Gaussian KDE and
  selects task losses above that adaptive threshold.
- Sample-level quality is initialised from improvement utility and evolves by
  the multiplicative update in Eq. (6). Diversity is an RBF similarity kernel
  and the fixed-size subset is drawn by an exact k-DPP. DPP inclusion
  probabilities use the standard marginal-kernel approximation for correction.
  This RBF factor makes the paper's stated quality-diversity intent explicit;
  a strictly diagonal kernel, as written in the main paper and used by the
  compact official implementation, cannot encode feature-space similarity.
- The compact PFN is fitted offline on source-task episodes. `suggest` is
  decorated with `torch.no_grad`, so target inference performs no update.

## Fairness controls

Each method sees the same offline source tasks, held-out target ID, initial
design seed, candidate pool seed, and freshly reset target-noise stream. No
target observations enter meta-training and no adaptation state crosses runs.
