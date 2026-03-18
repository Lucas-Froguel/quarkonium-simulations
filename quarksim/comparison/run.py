"""Main orchestrator for the three-method comparison experiment."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from quarksim.comparison.config import ExperimentConfig
from quarksim.comparison.methods import (
    MethodResult,
    run_vqe_ideal,
    run_vqite_ideal,
    run_ttite_ideal,
    run_vqe_excited as run_vqe_all_states,
    run_vqite_all_excited as run_vqite_all_states,
)
from quarksim.results import SimulationRecord, save_record


def _serialize(obj):
    """Make an object JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, complex):
        return float(obj.real)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    return obj


def run_noiseless(channels: list[str], cfg: ExperimentConfig, output_dir: Path):
    """Run all three methods (noiseless) on each channel."""
    all_results: dict[str, dict[str, MethodResult]] = {}
    excited_results: dict[str, dict[str, list[MethodResult]]] = {}

    for ch in channels:
        print(f"\n{'='*60}")
        print(f"Channel: {ch}")
        print(f"{'='*60}")

        all_results[ch] = {}

        # Ground state
        print("\n  VQE (noiseless)...")
        vqe_res = run_vqe_ideal(ch, cfg)
        all_results[ch]["VQE"] = vqe_res
        print(f"    E = {vqe_res.energy:.6f} (exact: {vqe_res.exact_energy:.6f}, "
              f"error: {vqe_res.error:+.6f}, fidelity: {vqe_res.fidelity:.6f})")

        print("  VQITE (noiseless)...")
        vqite_res = run_vqite_ideal(ch, cfg)
        all_results[ch]["VQITE"] = vqite_res
        print(f"    E = {vqite_res.energy:.6f} (exact: {vqite_res.exact_energy:.6f}, "
              f"error: {vqite_res.error:+.6f}, fidelity: {vqite_res.fidelity:.6f})")

        print("  TTITE (noiseless)...")
        ttite_res = run_ttite_ideal(ch, cfg)
        all_results[ch]["TTITE"] = ttite_res
        print(f"    E = {ttite_res.energy:.6f} (exact: {ttite_res.exact_energy:.6f}, "
              f"error: {ttite_res.error:+.6f}, fidelity: {ttite_res.fidelity:.6f})")

        # Excited states (VQE and VQITE)
        print("\n  VQE excited states...")
        excited_results[ch] = {}
        vqe_exc = run_vqe_all_states(ch, n_states=4, cfg=cfg)
        excited_results[ch]["VQE"] = vqe_exc
        for r in vqe_exc:
            k = r.metadata.get("state_index", "?")
            print(f"    State {k}: E = {r.energy:.4f} (exact: {r.exact_energy:.4f}, "
                  f"error: {r.error:+.4f})")

        # VQITE excited states skipped — too slow with circuit-based
        # metric tensor (swap test overlaps for penalty term)
        print("  VQITE excited states... (skipped — circuit overhead too high)")

    # Save results
    _save_noiseless_results(all_results, excited_results, output_dir)
    return all_results, excited_results


def run_noisy(channels: list[str], cfg: ExperimentConfig, output_dir: Path):
    """Run noise sweep on each channel."""
    from quarksim.comparison.noise import run_noise_sweep

    noise_results = {}
    for ch in channels:
        print(f"\n{'='*60}")
        print(f"Noise sweep: {ch}")
        print(f"{'='*60}")
        noise_results[ch] = run_noise_sweep(ch, cfg)

    # Save
    _save_noise_results(noise_results, output_dir)
    return noise_results


def _save_noiseless_results(
    all_results: dict,
    excited_results: dict,
    output_dir: Path,
):
    """Save noiseless results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for ch, methods in all_results.items():
        data[ch] = {}
        for method_name, mr in methods.items():
            data[ch][method_name] = {
                "energy": mr.energy,
                "exact_energy": mr.exact_energy,
                "error": mr.error,
                "fidelity": mr.fidelity,
                "wall_time": mr.wall_time,
                "convergence": mr.convergence,
                "wavefunction": mr.wavefunction.tolist(),
                "metadata": _serialize(mr.metadata),
            }

    with open(output_dir / "noiseless_results.json", "w") as f:
        json.dump(_serialize(data), f, indent=2)

    # Excited states
    exc_data = {}
    for ch, methods in excited_results.items():
        exc_data[ch] = {}
        for method_name, mr_list in methods.items():
            exc_data[ch][method_name] = [
                {
                    "energy": mr.energy,
                    "exact_energy": mr.exact_energy,
                    "error": mr.error,
                    "state_index": mr.metadata.get("state_index", 0),
                }
                for mr in mr_list
            ]

    with open(output_dir / "excited_results.json", "w") as f:
        json.dump(_serialize(exc_data), f, indent=2)


def _save_noise_results(noise_results: dict, output_dir: Path):
    """Save noise sweep results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for ch, sweep in noise_results.items():
        data[ch] = {
            "noise_levels": sweep.noise_levels,
            "exact_energy": sweep.exact_energy,
            "energies": _serialize(sweep.energies),
            "fidelities": _serialize(sweep.fidelities),
        }

    with open(output_dir / "noise_results.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Three-method comparison: VQE vs VQITE vs TTITE"
    )
    parser.add_argument(
        "--channel", choices=["1S0", "3S1", "1P1", "all"], default="all",
        help="Which channel(s) to simulate (default: all)"
    )
    parser.add_argument(
        "--phase", choices=["noiseless", "noisy", "all"], default="all",
        help="Experiment phase (default: all)"
    )
    parser.add_argument(
        "--noise-levels", nargs="+", type=float, default=None,
        help="Custom noise levels (default: 0.0 0.005 0.01 0.02 0.05)"
    )
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--output", default="output/comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    # Build config
    cfg = ExperimentConfig(seed=args.seed, output_dir=args.output)
    if args.noise_levels:
        cfg.noise_levels = args.noise_levels
    cfg.noise_repeats = args.n_repeats

    channels = cfg.channels if args.channel == "all" else [args.channel]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = None
    excited_results = None
    noise_results = None

    # Phase 1: Noiseless
    if args.phase in ("noiseless", "all"):
        all_results, excited_results = run_noiseless(channels, cfg, output_dir)

    # Phase 2: Noisy
    if args.phase in ("noisy", "all"):
        noise_results = run_noisy(channels, cfg, output_dir)

    # Phase 3: Plots
    if not args.no_plots and all_results:
        print("\nGenerating plots...")
        from quarksim.comparison.plots import generate_all_plots
        generate_all_plots(
            all_results, excited_results, noise_results, output_dir
        )

    # Phase 4: Report
    if not args.no_pdf:
        print("\nGenerating report...")
        from quarksim.comparison.report import generate_report
        generate_report(all_results, excited_results, noise_results, output_dir)

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
