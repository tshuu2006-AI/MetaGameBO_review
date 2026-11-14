"""Plotting tools for data, meta-training, model distributions and BO results.

Modules:
    style: Palette, fixed per-method colours, figure helpers.
    data: Task family, task parameters, optimum locations, Eq. (3) reweighting.
    training: ``TrainRecorder`` callback and meta-training curves / task-loss
        distributions / sample-weight evolution.
    distributions: k-DPP marginals and samples, task embeddings, target posterior.
    bo: Regret curves and ECDFs from results JSON, single-run BO traces.

Every ``plot_*`` function returns the figure and saves it when ``path`` is given.
"""
