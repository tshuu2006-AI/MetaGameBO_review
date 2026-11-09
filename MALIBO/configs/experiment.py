from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    runs: int = 100
    initial_points: int = 4
    bo_steps: int = 16
    candidate_count: int = 5120
    grid_size_for_bounds: int = 20_001

    # MALIBO meta-training: values from the official MALIBO Forrester script.
    meta_seed: int = 44
    meta_tasks: int = 256
    meta_points_per_task: int = 32
    meta_epochs: int = 2048
    meta_batch_size: int = 256
    malibo_gamma: float = 1.0 / 3.0
    malibo_num_layers: int = 5
    malibo_num_features: int = 50
    malibo_num_hidden_units: int = 64
    malibo_device: str = "cpu"

    # PFNs4BO uses an official pretrained model; the paper does not name which one.
    pfn_model: str = "hebo_plus"  # {"hebo_plus", "bnn"}
    pfn_device: str = "cpu:0"

    # Paper's CVaR definition uses confidence alpha; CVaR95 is the worst 5% tail.
    cvar_confidence: float = 0.95

    def to_dict(self):
        return asdict(self)
