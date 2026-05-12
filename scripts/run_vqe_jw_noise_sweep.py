"""Standalone runner that adds VQE-JW data to an existing noise_results.json.

Run with:
    PYTHONUNBUFFERED=1 uv run python scripts/run_vqe_jw_noise_sweep.py \
        --input output/thesis/noise_results.json \
        --output output/thesis/noise_results.json

It iterates over (channel, noise_rate, repeat) and runs VQE-JW only, then
merges the results in-place under the ``VQE-JW`` key of the existing
``energies`` and ``fidelities`` blocks. Other methods are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from quarksim.comparison.config import ExperimentConfig
from quarksim.comparison.methods import run_vqe_jw_ideal, run_vqe_jw_noisy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/thesis/noise_results.json")
    parser.add_argument("--output", default="output/thesis/noise_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument(
        "--noise-levels", nargs="+", type=float,
        default=[0.0, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--channels", nargs="+", default=["1S0", "3S1", "1P1"],
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input file not found: {in_path}")

    print(f"loading {in_path}")
    with in_path.open() as f:
        data = json.load(f)

    base_cfg = ExperimentConfig(seed=args.seed)

    for channel in args.channels:
        if channel not in data:
            print(f"channel {channel} missing from input -- skipping")
            continue

        block = data[channel]
        block["energies"].setdefault("VQE-JW", {})
        block["fidelities"].setdefault("VQE-JW", {})
        if channel not in data:
            continue

        # Ensure the noise_levels key reflects what we are about to fill
        existing_levels = set(map(float, block.get("noise_levels", [])))
        for r in args.noise_levels:
            existing_levels.add(r)
        block["noise_levels"] = sorted(existing_levels)

        for rate in args.noise_levels:
            key = str(rate)
            print(f"\n[{channel}] rate {rate:.3f}")
            energies: list[float] = []
            fidelities: list[float] = []
            for rep in range(args.n_repeats):
                rep_cfg = ExperimentConfig(
                    **{k: v for k, v in base_cfg.__dict__.items() if k != "seed"},
                )
                rep_cfg.seed = (args.seed or 0) + rep
                t0 = time.perf_counter()
                try:
                    if rate == 0.0:
                        mr = run_vqe_jw_ideal(channel, rep_cfg)
                    else:
                        mr = run_vqe_jw_noisy(channel, rate, rep_cfg)
                except Exception as exc:
                    print(f"  repeat {rep}: FAILED ({exc})")
                    continue
                wall = time.perf_counter() - t0
                energies.append(float(mr.energy))
                fidelities.append(float(mr.fidelity))
                print(f"  repeat {rep}: E={mr.energy:+.4f}  F={mr.fidelity:.4f}"
                      f"  ({wall:.1f}s)")

            block["energies"]["VQE-JW"][key] = energies
            block["fidelities"]["VQE-JW"][key] = fidelities

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
