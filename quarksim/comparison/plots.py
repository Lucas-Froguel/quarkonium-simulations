"""Comparison-specific plotting functions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"VQE": "#1f77b4", "VQITE": "#2ca02c", "TTITE": "#d62728"}
MARKERS = {"VQE": "o", "VQITE": "s", "TTITE": "^"}


def plot_convergence_comparison(all_results, output_dir: Path):
    """Plot convergence for all methods side by side, one panel per channel."""
    channels = list(all_results.keys())
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), squeeze=False)

    for i, ch in enumerate(channels):
        ax = axes[0, i]
        methods = all_results[ch]

        for method_name, mr in methods.items():
            conv = mr.convergence
            if method_name == "TTITE" and "tau_values" in mr.metadata:
                x = mr.metadata["tau_values"][:len(conv)]
                xlabel = r"Imaginary time $\tau$"
            else:
                x = list(range(len(conv)))
                xlabel = "Iteration / Step"

            ax.plot(x, conv, label=method_name, color=COLORS.get(method_name, "gray"))

        ax.axhline(mr.exact_energy, ls="--", color="black", alpha=0.5, label="Exact")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Energy [fm$^{-1}$]")
        ax.set_title(f"Channel: {ch}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "convergence_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_energy_summary(all_results, output_dir: Path):
    """Grouped bar chart of ground-state energies for each method x channel."""
    channels = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys())
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(channels))
    width = 0.2

    for j, method_name in enumerate(methods):
        energies = [all_results[ch][method_name].energy for ch in channels]
        offset = (j - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, energies, width, label=method_name,
                      color=COLORS.get(method_name, "gray"), alpha=0.85)
        for bar, e in zip(bars, energies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{e:.3f}", ha="center", va="bottom", fontsize=7)

    # Exact energies
    exact = [all_results[ch][methods[0]].exact_energy for ch in channels]
    offset = (n_methods / 2 + 0.5) * width
    ax.bar(x + offset, exact, width, label="Exact", color="black", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel(r"Ground-state energy [fm$^{-1}$]")
    ax.set_title("Noiseless Ground-State Energy Comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "energy_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_excited_states(excited_results, output_dir: Path):
    """Energy level diagram for VQE and VQITE excited states vs exact."""
    if not excited_results:
        return

    channels = list(excited_results.keys())
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 5), squeeze=False)

    for i, ch in enumerate(channels):
        ax = axes[0, i]
        ch_data = excited_results[ch]

        positions = {"Exact": 0, "VQE": 1, "VQITE": 2}
        x_labels = list(positions.keys())

        # Exact eigenvalues from first result
        first_method = next(iter(ch_data.values()))
        exact_eigs = [r.exact_energy for r in first_method]

        for k, e in enumerate(exact_eigs):
            ax.plot([positions["Exact"] - 0.3, positions["Exact"] + 0.3], [e, e],
                    color="black", linewidth=2)
            ax.text(positions["Exact"] + 0.35, e, f"{e:.3f}", fontsize=7, va="center")

        for method_name, mr_list in ch_data.items():
            pos = positions.get(method_name, 0)
            for mr in mr_list:
                color = COLORS.get(method_name, "gray")
                ax.plot([pos - 0.3, pos + 0.3], [mr.energy, mr.energy],
                        color=color, linewidth=2)
                ax.text(pos + 0.35, mr.energy, f"{mr.energy:.3f}", fontsize=7,
                        va="center", color=color)

        ax.set_xticks(list(positions.values()))
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(r"Energy [fm$^{-1}$]")
        ax.set_title(f"Energy Levels: {ch}")

    fig.tight_layout()
    fig.savefig(output_dir / "excited_states.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_noise_sweep(noise_results, output_dir: Path):
    """Energy error vs noise rate for all three methods."""
    if not noise_results:
        return

    channels = list(noise_results.keys())
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), squeeze=False)

    for i, ch in enumerate(channels):
        ax = axes[0, i]
        sweep = noise_results[ch]
        exact = sweep.exact_energy

        for method_name in ["VQE", "VQITE", "TTITE"]:
            rates = []
            means = []
            stds = []
            for rate, e_list in sorted(sweep.energies.get(method_name, {}).items()):
                if e_list:
                    rates.append(rate)
                    errors = [e - exact for e in e_list]
                    means.append(np.mean(errors))
                    stds.append(np.std(errors) if len(errors) > 1 else 0.0)

            if rates:
                ax.errorbar(rates, means, yerr=stds, label=method_name,
                            color=COLORS.get(method_name, "gray"),
                            marker=MARKERS.get(method_name, "o"),
                            capsize=3, markersize=5)

        ax.axhline(0, ls="--", color="black", alpha=0.3)
        ax.set_xlabel("Depolarizing rate")
        ax.set_ylabel(r"Energy error [fm$^{-1}$]")
        ax.set_title(f"Noise Sensitivity: {ch}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "noise_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fidelity_vs_noise(noise_results, output_dir: Path):
    """Fidelity vs noise rate for all three methods."""
    if not noise_results:
        return

    channels = list(noise_results.keys())
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 4), squeeze=False)

    for i, ch in enumerate(channels):
        ax = axes[0, i]
        sweep = noise_results[ch]

        for method_name in ["VQE", "VQITE", "TTITE"]:
            rates = []
            means = []
            stds = []
            for rate, f_list in sorted(sweep.fidelities.get(method_name, {}).items()):
                if f_list:
                    rates.append(rate)
                    means.append(np.mean(f_list))
                    stds.append(np.std(f_list) if len(f_list) > 1 else 0.0)

            if rates:
                ax.errorbar(rates, means, yerr=stds, label=method_name,
                            color=COLORS.get(method_name, "gray"),
                            marker=MARKERS.get(method_name, "o"),
                            capsize=3, markersize=5)

        ax.axhline(1.0, ls="--", color="black", alpha=0.3)
        ax.set_xlabel("Depolarizing rate")
        ax.set_ylabel("Fidelity")
        ax.set_title(f"Fidelity vs Noise: {ch}")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "fidelity_vs_noise.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_circuit_cost(all_results, output_dir: Path):
    """Bar chart of wall-clock time and convergence steps per method."""
    channels = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Wall time
    x = np.arange(len(channels))
    width = 0.25
    for j, method_name in enumerate(methods):
        times = [all_results[ch][method_name].wall_time for ch in channels]
        offset = (j - len(methods) / 2 + 0.5) * width
        ax1.bar(x + offset, times, width, label=method_name,
                color=COLORS.get(method_name, "gray"), alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(channels)
    ax1.set_ylabel("Wall time [s]")
    ax1.set_title("Computation Time")
    ax1.legend()

    # Convergence length
    for j, method_name in enumerate(methods):
        lengths = [len(all_results[ch][method_name].convergence) for ch in channels]
        offset = (j - len(methods) / 2 + 0.5) * width
        ax2.bar(x + offset, lengths, width, label=method_name,
                color=COLORS.get(method_name, "gray"), alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(channels)
    ax2.set_ylabel("Number of iterations / steps")
    ax2.set_title("Convergence Steps")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "circuit_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(all_results, excited_results, noise_results, output_dir: Path):
    """Generate all comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_results:
        plot_convergence_comparison(all_results, output_dir)
        plot_energy_summary(all_results, output_dir)
        plot_circuit_cost(all_results, output_dir)

    if excited_results:
        plot_excited_states(excited_results, output_dir)

    if noise_results:
        plot_noise_sweep(noise_results, output_dir)
        plot_fidelity_vs_noise(noise_results, output_dir)

    print(f"  Plots saved to {output_dir}/")
