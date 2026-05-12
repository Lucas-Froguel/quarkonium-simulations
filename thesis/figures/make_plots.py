"""Standalone visualisation script for thesis Chapter 3.

Reads the JSON outputs produced by ``quarksim.comparison.run`` and emits
thesis-quality JPEG figures.

Usage:
    python thesis/figures/make_plots.py \
        --input output/thesis \
        --output thesis/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Channel display order and pretty labels
CHANNELS = ["1S0", "3S1", "1P1"]
CHANNEL_TITLE = {"1S0": r"${}^{1}S_{0}\ (\eta_c)$",
                 "3S1": r"${}^{3}S_{1}\ (J/\psi)$",
                 "1P1": r"${}^{1}P_{1}\ (h_c)$"}

# Method order, colours, and markers
METHOD_ORDER = ["VQE", "VQE-JW", "VQITE", "TTITE"]
METHOD_LABEL = {
    "VQE": "VQE (direct)",
    "VQE-JW": "VQE (Jordan--Wigner)",
    "VQITE": "VQITE",
    "TTITE": "TTITE",
}
COLOUR = {
    "VQE": "#1f77b4",       # blue
    "VQE-JW": "#17becf",    # cyan
    "VQITE": "#d62728",     # red
    "TTITE": "#2ca02c",     # green
}
MARKER = {"VQE": "o", "VQE-JW": "s", "VQITE": "^", "TTITE": "D"}


def _setup_style() -> None:
    """Configure matplotlib for thesis-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.4,
        "lines.markersize": 5.0,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 1: Convergence
# ---------------------------------------------------------------------------

def plot_convergence(noiseless: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.8), sharex=False)
    for ax, ch in zip(axes, CHANNELS):
        data = noiseless.get(ch, {})
        exact = None
        for m in METHOD_ORDER:
            if m not in data:
                continue
            conv = data[m]["convergence"]
            xs = np.arange(1, len(conv) + 1)
            ax.plot(xs, conv, label=METHOD_LABEL[m],
                    color=COLOUR[m], linewidth=1.3)
            if exact is None:
                exact = data[m].get("exact_energy")
        if exact is not None:
            ax.axhline(exact, color="k", linestyle="--", linewidth=0.9,
                       label="exact")
        ax.set_ylabel(r"$E$ (fm$^{-1}$)")
        ax.set_title(CHANNEL_TITLE[ch])
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[-1].set_xlabel("iteration / step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    _save(fig, output_dir / "convergence.jpeg")


# ---------------------------------------------------------------------------
# Figure 2: Energy-error summary
# ---------------------------------------------------------------------------

def plot_energy_summary(noiseless: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(CHANNELS))
    n_methods = sum(1 for m in METHOD_ORDER if m in noiseless[CHANNELS[0]])
    width = 0.8 / n_methods
    methods_present = [m for m in METHOD_ORDER if m in noiseless[CHANNELS[0]]]
    for i, m in enumerate(methods_present):
        errs = [abs(noiseless[ch][m]["error"]) for ch in CHANNELS]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, errs, width=width * 0.95,
               label=METHOD_LABEL[m], color=COLOUR[m])
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_TITLE[ch] for ch in CHANNELS])
    ax.set_yscale("log")
    ax.set_ylabel(r"$|E_{\mathrm{est}} - E_{\mathrm{exact}}|$ (fm$^{-1}$)")
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    _save(fig, output_dir / "energy_summary.jpeg")


# ---------------------------------------------------------------------------
# Figure 3: Excited-state energy diagram
# ---------------------------------------------------------------------------

def plot_excited_states(excited: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
    for ax, ch in zip(axes, CHANNELS):
        data = excited.get(ch, {})
        levels_exact = None
        for m, states in data.items():
            if not states:
                continue
            levels = [s["energy"] for s in states]
            if levels_exact is None:
                levels_exact = [s["exact_energy"] for s in states]
            xs = np.arange(len(levels))
            ax.scatter(xs, levels, color=COLOUR.get(m, "k"),
                       marker=MARKER.get(m, "o"), s=42,
                       edgecolors="white", linewidths=0.7,
                       label=METHOD_LABEL.get(m, m))
        if levels_exact is not None:
            for k, e in enumerate(levels_exact):
                ax.hlines(e, k - 0.35, k + 0.35, colors="k",
                          linestyles="--", linewidth=0.9,
                          label="exact" if k == 0 else None)
        # Empty placeholder for VQITE (future work)
        ax.scatter([], [], color=COLOUR["VQITE"], marker=MARKER["VQITE"],
                   s=42, edgecolors="white", linewidths=0.7,
                   label="VQITE (future)")
        ax.set_ylabel(r"$E_k$ (fm$^{-1}$)")
        ax.set_title(CHANNEL_TITLE[ch])
        ax.set_xticks(range(4))
        ax.grid(alpha=0.3, linewidth=0.5)
    axes[-1].set_xlabel("state index $k$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    _save(fig, output_dir / "excited_states.jpeg")


# ---------------------------------------------------------------------------
# Figures 4 & 5: Noise sweep — energy error and fidelity
# ---------------------------------------------------------------------------

def _aggregate_noise(noise: dict, channel: str, method: str, kind: str):
    """Return (xs, mean, std) for energy or fidelity at each noise rate."""
    block = noise[channel][kind][method]
    xs = sorted(float(k) for k in block.keys())
    means = []
    stds = []
    for rate in xs:
        vals = block[str(rate) if str(rate) in block else float(rate)]
        if not vals:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
    return np.array(xs), np.array(means), np.array(stds)


def _resolve_rate_key(block: dict, rate: float):
    if str(rate) in block:
        return str(rate)
    if f"{rate:.1f}" in block:
        return f"{rate:.1f}"
    for k in block:
        try:
            if abs(float(k) - rate) < 1e-9:
                return k
        except ValueError:
            continue
    return None


def _aggregate_rate_keys(block: dict):
    rates = []
    for k in block:
        try:
            rates.append(float(k))
        except ValueError:
            continue
    return sorted(set(rates))


def plot_noise_sweep(noise: dict, output_dir: Path, kind: str = "energy") -> None:
    """Plot energy error vs depolarising rate (kind='energy') or fidelity."""
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.8), sharex=True)
    key = "energies" if kind == "energy" else "fidelities"
    ylabel = (r"$|E - E_{\mathrm{exact}}|$ (fm$^{-1}$)"
              if kind == "energy" else r"$\mathcal{F}$")

    for ax, ch in zip(axes, CHANNELS):
        block_all = noise[ch][key]
        exact_e = noise[ch].get("exact_energy", 0.0)
        for m in METHOD_ORDER:
            if m not in block_all:
                continue
            block = block_all[m]
            rates = _aggregate_rate_keys(block)
            xs, means, stds = [], [], []
            for r in rates:
                k = _resolve_rate_key(block, r)
                vals = block[k] if k else []
                if not vals:
                    continue
                xs.append(r)
                if kind == "energy":
                    errs = [abs(v - exact_e) for v in vals]
                    means.append(float(np.mean(errs)))
                    stds.append(float(np.std(errs)))
                else:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
            xs = np.array(xs)
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(xs, means, yerr=stds, label=METHOD_LABEL[m],
                        color=COLOUR[m], marker=MARKER[m], markersize=5.0,
                        linewidth=1.2, capsize=2.5)
        ax.set_ylabel(ylabel)
        ax.set_title(CHANNEL_TITLE[ch])
        ax.grid(alpha=0.3, linewidth=0.5)
        if kind == "energy":
            ax.set_yscale("symlog", linthresh=1e-3)
    axes[-1].set_xlabel("depolarising rate $p$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    name = "noise_sweep_energy.jpeg" if kind == "energy" else "noise_sweep_fidelity.jpeg"
    _save(fig, output_dir / name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="output/thesis",
                        help="Directory containing the JSON outputs")
    parser.add_argument("--output", default="thesis/figures",
                        help="Directory for emitted JPEGs")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    _setup_style()

    noiseless_file = in_dir / "noiseless_results.json"
    excited_file = in_dir / "excited_results.json"
    noise_file = in_dir / "noise_results.json"

    if noiseless_file.exists():
        print(f"Reading {noiseless_file}")
        with noiseless_file.open() as f:
            noiseless = json.load(f)
        plot_convergence(noiseless, out_dir)
        plot_energy_summary(noiseless, out_dir)
    else:
        print(f"WARN: {noiseless_file} not found, skipping convergence plot")

    if excited_file.exists():
        print(f"Reading {excited_file}")
        with excited_file.open() as f:
            excited = json.load(f)
        plot_excited_states(excited, out_dir)
    else:
        print(f"WARN: {excited_file} not found, skipping excited-state plot")

    if noise_file.exists():
        print(f"Reading {noise_file}")
        with noise_file.open() as f:
            noise = json.load(f)
        plot_noise_sweep(noise, out_dir, kind="energy")
        plot_noise_sweep(noise, out_dir, kind="fidelity")
    else:
        print(f"WARN: {noise_file} not found, skipping noise-sweep plots")

    print("Done.")


if __name__ == "__main__":
    main()
