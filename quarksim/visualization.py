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
            a_plus_c = data["extrapolated"]
            r = data["r_fit"]
            c = data.get("c_fit", 0.0)
            a = a_plus_c - c
            ax.plot(lam_fine, a * r ** lam_fine + c, "C0--", alpha=0.5)
            ax.scatter(0, a_plus_c, color="C3", marker="s", s=50, zorder=6, label="ZNE")

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


# ---------------------------------------------------------------------------
# TTITE (Yi, Huo, Liu, Fan, Zhang & Cao) visualization functions
# ---------------------------------------------------------------------------


def plot_ite_convergence(
    tau_values_dict: dict[str, list[float]],
    energy_dict: dict[str, list[float]],
    fidelity_dict: dict[str, list[float]],
    exact_energy: float | None = None,
    title: str = "ITE Convergence",
    ylabel_energy: str = r"Energy $E_\tau$",
    ylabel_fidelity: str = r"Fidelity $F$",
    save_path: str | Path | None = None,
):
    """Dual-axis plot: energy (left) and fidelity (right) vs imaginary time tau.

    Reproduces the layout of Figures 2 and 4 from the TTITE paper.
    Multiple curves per axis, one per method.

    Args:
        tau_values_dict: {"method_name": [tau_0, tau_1, ...], ...}
        energy_dict: {"method_name": [E_0, E_1, ...], ...}
        fidelity_dict: {"method_name": [F_0, F_1, ...], ...}
        exact_energy: Exact ground state energy for dashed reference line.
        title: Plot title.
        ylabel_energy: Left y-axis label.
        ylabel_fidelity: Right y-axis label.
        save_path: If given, save figure to this path.

    Returns:
        (fig, (ax_energy, ax_fidelity)) matplotlib objects.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Color palette: blues for energy, oranges/reds for fidelity
    energy_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
    fidelity_colors = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22"]
    markers_e = ["v", "^", "s", "D"]
    markers_f = ["v", "^", "s", "D"]

    for i, (method, taus) in enumerate(tau_values_dict.items()):
        ec = energy_colors[i % len(energy_colors)]
        fc = fidelity_colors[i % len(fidelity_colors)]
        mk_e = markers_e[i % len(markers_e)]
        mk_f = markers_f[i % len(markers_f)]

        if method in energy_dict:
            ax1.plot(
                taus, energy_dict[method],
                color=ec, marker=mk_e, markersize=5, linewidth=1.5,
                label=f"E-{method}",
            )
        if method in fidelity_dict:
            ax2.plot(
                taus, fidelity_dict[method],
                color=fc, marker=mk_f, markersize=5, linewidth=1.5,
                label=f"F-{method}",
            )

    if exact_energy is not None:
        ax1.axhline(
            y=exact_energy, color="gray", linestyle="--", linewidth=1.5,
            label="GS energy",
        )

    ax1.set_xlabel(r"Time $\tau$")
    ax1.set_ylabel(ylabel_energy)
    ax2.set_ylabel(ylabel_fidelity)
    ax1.set_title(title)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_potential_energy_surface(
    D_values: np.ndarray | list[float],
    energy_curves: dict[str, list[float]],
    exact_gs_energies: list[float] | None = None,
    initial_energies: list[float] | None = None,
    title: str = r"H$_2$ Potential Energy Surface",
    save_path: str | Path | None = None,
):
    """Plot energy vs interatomic distance D for different (tau, order) combos.

    Reproduces Figure 3 of the TTITE paper.

    Args:
        D_values: Array of interatomic distances.
        energy_curves: {"label": [E(D_0), E(D_1), ...], ...}
        exact_gs_energies: Exact ground state energies at each D.
        initial_energies: Initial state energies at each D (tau=0).
        title: Plot title.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if initial_energies is not None:
        ax.plot(D_values, initial_energies, "k-", linewidth=1.5, label="Initial energy")

    if exact_gs_energies is not None:
        ax.plot(
            D_values, exact_gs_energies, "k--", linewidth=1.5,
            marker="*", markersize=6, label="GS energy",
        )

    markers = ["o", "s", "^", "v", "D", "p"]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(energy_curves)))

    for i, (label, energies) in enumerate(energy_curves.items()):
        ax.plot(
            D_values, energies,
            color=colors[i], marker=markers[i % len(markers)],
            markersize=5, linewidth=1.2, label=label,
        )

    ax.set_xlabel(r"Interatomic distance $D$ ($\AA$)")
    ax.set_ylabel(r"Energy $E_\tau$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_multi_size_convergence(
    tau_values_dict: dict[str, list[float]],
    norm_energy_dict: dict[str, list[float]],
    fidelity_dict: dict[str, list[float]],
    title: str = "Heisenberg Ground State at Different Sizes",
    save_path: str | Path | None = None,
):
    """Plot normalized energy and fidelity vs tau for multiple system sizes.

    Reproduces Figure 5 of the TTITE paper.

    Args:
        tau_values_dict: {"n=3": [tau_0, ...], "n=6": [...], ...}
        norm_energy_dict: {"n=3": [NE_0, ...], ...}
        fidelity_dict: {"n=3": [F_0, ...], ...}
        title: Plot title.
        save_path: If given, save figure to this path.

    Returns:
        (fig, (ax1, ax2)) matplotlib objects.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D"]

    for i, (label, taus) in enumerate(tau_values_dict.items()):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        if label in norm_energy_dict:
            ax1.plot(
                taus, norm_energy_dict[label],
                color=c, marker=m, markersize=4, linewidth=1.2,
                label=f"NE-{label}",
            )
        if label in fidelity_dict:
            ax2.plot(
                taus, fidelity_dict[label],
                color=c, marker=m, markersize=4, linewidth=1.2,
                linestyle="--", alpha=0.7,
                label=f"F-{label}",
            )

    ax1.set_xlabel(r"Time $\tau$")
    ax1.set_ylabel(r"Normalized energy $NE_\tau$")
    ax2.set_ylabel(r"Fidelity $F$")
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_vqite_convergence(
    energy_histories: list[list[float]],
    exact_energies: list[float] | None = None,
    state_labels: list[str] | None = None,
    shot_histories: list[list[float]] | None = None,
    title: str = "VQITE Convergence",
    ylabel: str = r"Energy [fm$^{-1}$]",
    save_path: str | Path | None = None,
):
    """Plot VQITE energy vs imaginary time step for multiple states.

    Reproduces the style of Woloshyn Figs. 4-6:
      - Solid colored lines: wave function simulator (exact statevector)
      - Dashed colored lines: ideal quantum simulation (shot-based)
      - Horizontal thin dashed lines: exact eigenvalues from diagonalization

    Args:
        energy_histories: Exact statevector histories, one list per state.
        exact_energies: Exact eigenvalues for horizontal reference lines.
        state_labels: Labels for each state (e.g., ["1S", "2S", "3S", "4S"]).
        shot_histories: Shot-based histories (same structure). Plotted as
            colored dashed lines when provided.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    colors = ["red", "green", "blue", "magenta"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, hist in enumerate(energy_histories):
        color = colors[i % len(colors)]
        label = state_labels[i] if state_labels else f"State {i+1}"
        ax.plot(hist, color=color, linewidth=1.5, label=label)

    if shot_histories is not None:
        for i, hist in enumerate(shot_histories):
            color = colors[i % len(colors)]
            label = f"{state_labels[i]} (shots)" if state_labels else f"State {i+1} (shots)"
            ax.plot(hist, color=color, linewidth=1.2, linestyle="--",
                    alpha=0.7, label=label)

    if exact_energies is not None:
        for i, e in enumerate(exact_energies):
            ax.axhline(
                y=e, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
            )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_vqite_noisy_convergence(
    ideal_history: list[float],
    noisy_history: list[float],
    exact_energy: float | None = None,
    title: str = "VQITE — Noisy Simulation",
    ylabel: str = r"Energy [fm$^{-1}$]",
    save_path: str | Path | None = None,
):
    """Plot ideal vs noisy VQITE convergence for a single state.

    Reproduces the style of Woloshyn Figs. 9-11: one state at a time,
    comparing ideal (black), noisy (red), and optionally mitigated (green).

    Args:
        ideal_history: Energy vs step from noiseless shot-based simulation.
        noisy_history: Energy vs step from noisy simulation.
        exact_energy: Exact eigenvalue for horizontal reference.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: If given, save figure to this path.

    Returns:
        (fig, ax) matplotlib objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ideal_history, "k-", linewidth=1.5, label="Ideal simulation")
    ax.plot(noisy_history, "r-", linewidth=1.2, alpha=0.8, label="Noisy simulation")

    if exact_energy is not None:
        ax.axhline(
            y=exact_energy, color="gray", linestyle="--", linewidth=1,
            label=f"Exact: {exact_energy:.3f}",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
