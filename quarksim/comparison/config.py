"""Experiment configuration for the three-method comparison."""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """All tuneable parameters for the comparison experiment."""

    # Channels to run
    channels: list[str] = field(default_factory=lambda: ["1S0", "3S1", "1P1"])

    # VQE settings
    vqe_maxiter: int = 300
    vqe_method: str = "cobyla"
    vqe_shots: int = 8192

    # VQITE settings
    vqite_n_steps: int = 50
    vqite_dtau: float = 0.02
    vqite_regularization: float = 1e-4

    # TTITE settings
    ttite_tau_total: float = 5.0
    ttite_dt: float = 0.02
    ttite_order: int = 10

    # Noise sweep
    noise_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.05]
    )
    noise_repeats: int = 3

    # ZNE settings
    zne_scale_factors: list[int] = field(default_factory=lambda: [1, 3, 5, 7])
    zne_shots: int = 32768

    # Excited-state penalty
    penalty_alpha: float = 10.0

    # General
    seed: int | None = None
    output_dir: str = "output/comparison"
