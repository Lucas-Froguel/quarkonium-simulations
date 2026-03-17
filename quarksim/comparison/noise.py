"""Noise sweep: run all three methods at multiple noise levels."""

from dataclasses import dataclass, field

import numpy as np

from quarksim.comparison.config import ExperimentConfig
from quarksim.comparison.methods import (
    run_vqe_noisy as _run_vqe_noisy_comparison,
    run_vqite_noisy,
    run_ttite_noisy,
    MethodResult,
)


@dataclass
class NoiseSweepResult:
    """Results of a noise sweep for one channel."""

    channel: str
    noise_levels: list[float]
    # method -> noise_level -> list of energies (one per repeat)
    energies: dict[str, dict[float, list[float]]] = field(default_factory=dict)
    # method -> noise_level -> list of fidelities
    fidelities: dict[str, dict[float, list[float]]] = field(default_factory=dict)
    exact_energy: float = 0.0


def run_noise_sweep(
    channel: str,
    cfg: ExperimentConfig | None = None,
) -> NoiseSweepResult:
    """Run all three methods at each noise level with repeats.

    Returns structured results with mean/std accessible via the energies dict.
    """
    cfg = cfg or ExperimentConfig()
    noise_levels = cfg.noise_levels

    result = NoiseSweepResult(channel=channel, noise_levels=noise_levels)

    for method_name in ["VQE", "VQITE", "TTITE"]:
        result.energies[method_name] = {}
        result.fidelities[method_name] = {}

    for rate in noise_levels:
        print(f"  Noise rate {rate:.3f}:")
        for method_name in ["VQE", "VQITE", "TTITE"]:
            e_list: list[float] = []
            f_list: list[float] = []

            for rep in range(cfg.noise_repeats):
                # Use a different seed for each repeat
                rep_cfg = ExperimentConfig(
                    **{k: v for k, v in cfg.__dict__.items() if k != "seed"}
                )
                rep_cfg.seed = (cfg.seed or 0) + rep

                try:
                    if method_name == "VQE":
                        if rate == 0.0:
                            from quarksim.comparison.methods import run_vqe_ideal
                            mr = run_vqe_ideal(channel, rep_cfg)
                        else:
                            mr = _run_vqe_noisy_comparison(channel, rate, rep_cfg)
                    elif method_name == "VQITE":
                        if rate == 0.0:
                            from quarksim.comparison.methods import run_vqite_ideal
                            mr = run_vqite_ideal(channel, rep_cfg)
                        else:
                            mr = run_vqite_noisy(channel, rate, rep_cfg)
                    else:  # TTITE
                        if rate == 0.0:
                            from quarksim.comparison.methods import run_ttite_ideal
                            mr = run_ttite_ideal(channel, rep_cfg)
                        else:
                            mr = run_ttite_noisy(channel, rate, rep_cfg)

                    e_list.append(mr.energy)
                    f_list.append(mr.fidelity)
                    result.exact_energy = mr.exact_energy
                except Exception as e:
                    print(f"    {method_name} repeat {rep} failed: {e}")
                    continue

            result.energies[method_name][rate] = e_list
            result.fidelities[method_name][rate] = f_list

            if e_list:
                mean_e = np.mean(e_list)
                std_e = np.std(e_list) if len(e_list) > 1 else 0.0
                print(f"    {method_name}: E = {mean_e:.4f} +/- {std_e:.4f}")

    return result
