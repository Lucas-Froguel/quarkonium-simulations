"""Visualization tools for quarkonium simulations.

Provides plotting functions for VQE convergence, energy level comparison,
wavefunction amplitudes, and cross-method comparison.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(
    energies: list[float],
    exact_energy: float | None = None,
    title: str = "VQE Convergence",
    ylabel: str = "Energy (MeV)",
    save_path: str | Path | None = None,
):
    """Plot energy vs function evaluations during VQE optimization.

    Args:
        energies: Energy at each cost function evaluation.
        exact_energy: Exact ground state energy for reference line.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energies, "b-", linewidth=1, alpha=0.8, label="VQE energy")

    if exact_energy is not None:
        ax.axhline(
            y=exact_energy,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Exact: {exact_energy:.1f} MeV",
        )

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_energy_levels(
    levels_dict: dict[str, list[float]],
    title: str = "Energy Level Comparison",
    ylabel: str = "Energy (MeV)",
    save_path: str | Path | None = None,
):
    """Plot energy levels from different methods side by side.

    Args:
        levels_dict: {"method_name": [E0, E1, ...], ...}
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = list(levels_dict.keys())
    n_methods = len(methods)
    width = 0.6

    # Collect all energy levels to determine unique level indices
    max_levels = max(len(v) for v in levels_dict.values())
    level_labels = [f"$E_{i}$" for i in range(max_levels)]

    for i, (method, levels) in enumerate(levels_dict.items()):
        for j, energy in enumerate(levels):
            line = ax.hlines(
                energy,
                i - width / 2,
                i + width / 2,
                colors=f"C{j}",
                linewidth=2.5,
                label=level_labels[j] if i == 0 else None,
            )
            ax.text(
                i + width / 2 + 0.08,
                energy,
                f"{energy:.1f}",
                va="center",
                fontsize=8,
            )

    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_wavefunction(
    amplitudes: np.ndarray | list[float],
    labels: list[str],
    title: str = "Wavefunction Components",
    save_path: str | Path | None = None,
):
    """Bar chart of wavefunction amplitudes in the physical basis.

    Args:
        amplitudes: Wavefunction coefficients (may be negative).
        labels: Basis state labels (e.g., ["|0>", "|1>", "|2>"]).
        title: Plot title.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["C0" if a >= 0 else "C3" for a in amplitudes]
    ax.bar(labels, amplitudes, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.axhline(y=0, color="k", linewidth=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_method_comparison(
    results: list[dict],
    title: str = "Ground State Energy Comparison",
    ylabel: str = "Energy (MeV)",
    save_path: str | Path | None = None,
):
    """Bar chart comparing ground state energies from different methods.

    Args:
        results: List of dicts, each with 'method' (str) and 'energy' (float).
            Optionally include 'exact_energy' (float) in the first entry.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = [r["method"] for r in results]
    energies = [r["energy"] for r in results]

    bars = ax.bar(methods, energies, color="steelblue", edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{energy:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    exact = results[0].get("exact_energy")
    if exact is not None:
        ax.axhline(
            y=exact,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Exact: {exact:.1f} MeV",
        )
        ax.legend()

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_zne(
    zne_result,
    exact_energy: float | None = None,
    title: str = "Zero-Noise Extrapolation",
    save_path: str | Path | None = None,
):
    """Plot ZNE: total energy vs noise scale factor with exponential fit.

    Shows raw noisy measurements at each lambda, the extrapolated value
    at lambda=0, and optionally the exact energy for comparison.
    Similar to Fig. 4 of Gallimore & Liao.

    Args:
        zne_result: ZNEResult from run_zne().
        exact_energy: Exact ground state energy for reference.
        title: Plot title.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    lambdas = sorted(zne_result.raw_energies.keys())
    energies = [zne_result.raw_energies[lam] for lam in lambdas]

    # Raw data points
    ax.scatter(lambdas, energies, color="C0", s=60, zorder=5, label="Noisy measurements")

    # Fit curve through data + extrapolated point
    try:
        from scipy.optimize import curve_fit as _cf

        def _exp(x, a, b, c):
            return a * np.exp(b * x) + c

        popt, _ = _cf(
            _exp, lambdas, energies,
            p0=[energies[-1] - energies[0], 0.3, energies[0]],
            maxfev=5000,
        )
        lam_fine = np.linspace(0, max(lambdas) + 0.5, 100)
        ax.plot(lam_fine, _exp(lam_fine, *popt), "C0--", alpha=0.5, label="Exponential fit")
    except (RuntimeError, ValueError):
        pass

    # Extrapolated point at lambda=0
    ax.errorbar(
        0, zne_result.energy, yerr=2 * zne_result.energy_std,
        fmt="s", color="C3", markersize=10, capsize=5, zorder=6,
        label=f"ZNE: {zne_result.energy:.1f} $\\pm$ {2*zne_result.energy_std:.1f} MeV",
    )

    if exact_energy is not None:
        ax.axhline(
            y=exact_energy, color="green", linestyle="--", linewidth=1.5,
            label=f"Exact: {exact_energy:.1f} MeV",
        )

    ax.set_xlabel(r"Noise scale factor $\lambda$")
    ax.set_ylabel("Energy (MeV)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_zne_per_term(
    zne_result,
    exact_term_values: dict[str, float] | None = None,
    title: str = "ZNE Per Pauli Term",
    save_path: str | Path | None = None,
):
    """Plot each Pauli term's decay vs lambda with exponential fits.

    Similar to Fig. 3 of Gallimore & Liao: each subplot shows one
    traceless Pauli term decaying toward zero with increasing noise.
    Noiseless exact values shown as stars (matching the paper's convention).

    Args:
        zne_result: ZNEResult from run_zne().
        exact_term_values: {pauli_label: noiseless <P>} for reference stars.
        title: Plot title.
        save_path: If given, save figure to this path.

    Returns:
        (fig, axes) matplotlib objects.
    """
    terms = {k: v for k, v in zne_result.per_term.items() if not all(c == "I" for c in k)}

    n_terms = len(terms)
    cols = 3
    rows = (n_terms + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes_flat = np.array(axes).flatten()

    lambdas = np.array(zne_result.scale_factors, dtype=float)

    for idx, (label, data) in enumerate(terms.items()):
        ax = axes_flat[idx]
        values = np.array(data["values"])

        ax.scatter(lambdas, values, color="C0", s=30, zorder=5, label="Noisy")

        if "r_fit" in data:
            lam_fine = np.linspace(0, max(lambdas) + 0.5, 50)
            a, r = data["extrapolated"], data["r_fit"]
            ax.plot(lam_fine, a * r ** lam_fine, "C0--", alpha=0.5)
            ax.scatter(0, a, color="C3", marker="s", s=50, zorder=6, label="ZNE")

        if exact_term_values and label in exact_term_values:
            ax.scatter(
                0, exact_term_values[label], color="green", marker="*",
                s=120, zorder=7, label="Exact",
            )

        ax.set_title(label, fontsize=10, fontfamily="monospace")
        ax.set_xlabel(r"$\lambda$", fontsize=9)
        ax.set_ylabel(r"$\langle P \rangle$", fontsize=9)
        ax.tick_params(labelsize=8)

    # Single shared legend from the last subplot with all entries
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=3)

    for idx in range(n_terms, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes
