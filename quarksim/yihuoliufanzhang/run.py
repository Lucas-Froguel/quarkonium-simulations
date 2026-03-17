"""Run the TTITE simulation for ground state preparation.

Reproduces key figures from:
    Yi, Huo, Liu, Fan, Zhang & Cao,
    "A probabilistic quantum algorithm for imaginary-time evolution
     based on Taylor expansion", EPJ Quantum Technology 12:43 (2025).

Usage:
    uv run python -m quarksim.yihuoliufanzhang.run [OPTIONS]

Options:
    --system {h2,heisenberg}   Test system (default: h2)
    --D FLOAT                  H2 interatomic distance in Angstrom (default: 0.35)
    --n INT                    Heisenberg chain length (default: 6)
    --J FLOAT                  Heisenberg coupling (default: 1.0)
    --field FLOAT              Heisenberg magnetic field h (default: 1.0)
    --tau FLOAT                Total imaginary time
    --dt FLOAT                 Trotter step size
    --order INT                Taylor expansion order R (default: 10)
    --figure {2,3,4,5,all}     Which paper figure to reproduce (default: all)
    --output DIR               Output directory (default: output/yihuoliufanzhang)
    --no-plots                 Skip generating plots
"""

import argparse
from pathlib import Path

import numpy as np

from quarksim.results import SimulationRecord, save_record
from quarksim.simulation import exact_diagonalization
from quarksim.visualization import (
    plot_ite_convergence,
    plot_multi_size_convergence,
    plot_potential_energy_surface,
)
from quarksim.yihuoliufanzhang.evolution import (
    run_ite_exact,
    run_trotter_ite,
    run_ttite,
)
from quarksim.yihuoliufanzhang.hamiltonian import (
    build_h2_hamiltonian,
    build_heisenberg_hamiltonian,
    h2_coefficients,
)


def _make_initial_state_plus(n: int) -> np.ndarray:
    """Create |+>^{otimes n} initial state."""
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    return state


def _make_h2_initial_state_fig3() -> np.ndarray:
    """Initial state for Figure 3: (4|00> + 1|01> + 1|10> + 1|11>) / sqrt(19)."""
    state = np.array([4.0, 1.0, 1.0, 1.0], dtype=complex)
    return state / np.linalg.norm(state)


def reproduce_figure_2(output_dir: Path):
    """Figure 2: H2 at D=0.35, energy/fidelity convergence vs tau.

    5th-order Taylor expansion (expansion to 4th term, i.e., j=0..4 -> order=4).
    tau=3, dt=0.3, initial state |+>|+>.
    """
    print("\n=== Reproducing Figure 2: H2 convergence ===")
    D = 0.35
    tau, dt, order = 3.0, 0.3, 4
    ham = build_h2_hamiltonian(D)
    psi0 = _make_initial_state_plus(2)

    eigenvalues, _ = exact_diagonalization(ham)
    gs_energy = eigenvalues[0]
    print(f"  H2 at D={D}: GS energy = {gs_energy:.6f}")

    # Run all three methods
    print("  Running exact ITE...")
    ite_exact = run_ite_exact(ham, psi0, tau, n_points=50)

    print("  Running Trotter ITE...")
    trotter = run_trotter_ite(ham, psi0, tau, dt)

    print(f"  Running TTITE (order={order})...")
    ttite = run_ttite(ham, psi0, tau, dt, order)

    print(f"  Final TTITE energy: {ttite.final_energy:.6f}  (exact: {gs_energy:.6f})")
    print(f"  Final TTITE fidelity: {ttite.final_fidelity:.6f}")

    # Plot
    tau_dict = {
        "ITE theory": ite_exact.tau_values,
        "Trotter ITE": trotter.tau_values,
        "Proposed": ttite.tau_values,
    }
    energy_dict = {
        "ITE theory": ite_exact.energy_history,
        "Trotter ITE": trotter.energy_history,
        "Proposed": ttite.energy_history,
    }
    fidelity_dict = {
        "ITE theory": ite_exact.fidelity_history,
        "Trotter ITE": trotter.fidelity_history,
        "Proposed": ttite.fidelity_history,
    }

    plot_ite_convergence(
        tau_dict, energy_dict, fidelity_dict,
        exact_energy=gs_energy,
        title=rf"H$_2$ ground state ($D = {D}$ $\AA$)",
        save_path=output_dir / "figure2_h2_convergence.png",
    )
    print(f"  Saved {output_dir / 'figure2_h2_convergence.png'}")

    return ttite


def reproduce_figure_3(output_dir: Path):
    """Figure 3: H2 potential energy surface E(D).

    Vary D from 0.3 to 2.1, run TTITE at tau=0.3, 1.2, 3.0
    with Taylor orders 2 and 10. dt=0.3.
    Initial state: (4|00> + 1|01> + 1|10> + 1|11>) / sqrt(19).
    """
    print("\n=== Reproducing Figure 3: H2 PES ===")
    D_values = np.arange(0.30, 2.15, 0.05)
    dt = 0.3
    tau_order_combos = [
        (0.3, 2), (0.3, 10),
        (1.2, 2), (1.2, 10),
        (3.0, 2), (3.0, 10),
    ]

    # Compute exact GS energies and initial energies at each D
    exact_gs = []
    initial_es = []

    for D in D_values:
        ham = build_h2_hamiltonian(D)
        evals, _ = exact_diagonalization(ham)
        exact_gs.append(evals[0])

        psi0 = _make_h2_initial_state_fig3()
        H_mat = ham.to_matrix()
        e0 = float(np.real(psi0.conj() @ H_mat @ psi0))
        initial_es.append(e0)

    # Run TTITE at each (tau, order) combo for all D values
    energy_curves = {}
    for tau, order in tau_order_combos:
        label = rf"$\tau$={tau}, order_{order}"
        print(f"  Running {label}...")
        energies = []
        for D in D_values:
            ham = build_h2_hamiltonian(D)
            psi0 = _make_h2_initial_state_fig3()
            result = run_ttite(ham, psi0, tau, dt, order)
            energies.append(result.final_energy)
        energy_curves[label] = energies

    plot_potential_energy_surface(
        D_values, energy_curves,
        exact_gs_energies=exact_gs,
        initial_energies=initial_es,
        title=r"H$_2$ energy vs interatomic distance $D$",
        save_path=output_dir / "figure3_h2_pes.png",
    )
    print(f"  Saved {output_dir / 'figure3_h2_pes.png'}")


def reproduce_figure_4(output_dir: Path):
    """Figure 4: Heisenberg n=6, J=h=1, convergence.

    tau=2, dt=0.1. Compare order 2 and order 10 + baselines.
    """
    print("\n=== Reproducing Figure 4: Heisenberg n=6 convergence ===")
    n, J, h_field = 6, 1.0, 1.0
    tau, dt = 2.0, 0.1
    ham = build_heisenberg_hamiltonian(n, J, h_field)
    psi0 = _make_initial_state_plus(n)

    eigenvalues, _ = exact_diagonalization(ham)
    gs_energy = eigenvalues[0]
    print(f"  Heisenberg n={n}, J={J}, h={h_field}: GS energy = {gs_energy:.6f}")

    print("  Running exact ITE...")
    ite_exact = run_ite_exact(ham, psi0, tau, n_points=40)

    print("  Running Trotter ITE...")
    trotter = run_trotter_ite(ham, psi0, tau, dt)

    print("  Running TTITE (order=2)...")
    ttite_2 = run_ttite(ham, psi0, tau, dt, order=2)

    print("  Running TTITE (order=10)...")
    ttite_10 = run_ttite(ham, psi0, tau, dt, order=10)

    print(f"  TTITE order=2  final energy: {ttite_2.final_energy:.6f}")
    print(f"  TTITE order=10 final energy: {ttite_10.final_energy:.6f}")

    tau_dict = {
        "ITE theory": ite_exact.tau_values,
        "Trotter ITE": trotter.tau_values,
        "Proposed_order_2": ttite_2.tau_values,
        "Proposed_order_10": ttite_10.tau_values,
    }
    energy_dict = {
        "ITE theory": ite_exact.energy_history,
        "Trotter ITE": trotter.energy_history,
        "Proposed_order_2": ttite_2.energy_history,
        "Proposed_order_10": ttite_10.energy_history,
    }
    fidelity_dict = {
        "ITE theory": ite_exact.fidelity_history,
        "Trotter ITE": trotter.fidelity_history,
        "Proposed_order_2": ttite_2.fidelity_history,
        "Proposed_order_10": ttite_10.fidelity_history,
    }

    plot_ite_convergence(
        tau_dict, energy_dict, fidelity_dict,
        exact_energy=gs_energy,
        title=f"Heisenberg chain $n={n}$, $J=h={J}$",
        save_path=output_dir / "figure4_heisenberg_convergence.png",
    )
    print(f"  Saved {output_dir / 'figure4_heisenberg_convergence.png'}")

    return ttite_10


def reproduce_figure_5(output_dir: Path):
    """Figure 5: Heisenberg at n=3,6,9,12, J=h=1.

    tau=2, dt=0.1, 5th-order Taylor (order=4).
    Normalized energy NE = (E - E_min) / (E_max - E_min).
    """
    print("\n=== Reproducing Figure 5: Heisenberg multi-size ===")
    sizes = [3, 6, 9, 12]
    J, h_field = 1.0, 1.0
    tau, dt, order = 2.0, 0.1, 4

    tau_dict = {}
    ne_dict = {}
    fid_dict = {}

    for n in sizes:
        label = f"n_{n}"
        print(f"  Running n={n} ({2**n}-dim Hilbert space)...")
        ham = build_heisenberg_hamiltonian(n, J, h_field)
        psi0 = _make_initial_state_plus(n)

        eigenvalues, _ = exact_diagonalization(ham)
        E_min = eigenvalues[0]
        E_max = eigenvalues[-1]

        result = run_ttite(ham, psi0, tau, dt, order)

        # Normalized energy
        denom = E_max - E_min
        if abs(denom) < 1e-15:
            ne_vals = [0.0] * len(result.energy_history)
        else:
            ne_vals = [(e - E_min) / denom for e in result.energy_history]

        tau_dict[label] = result.tau_values
        ne_dict[label] = ne_vals
        fid_dict[label] = result.fidelity_history

        print(f"    GS energy = {E_min:.4f}, final NE = {ne_vals[-1]:.6f}, "
              f"final F = {result.fidelity_history[-1]:.6f}")

    plot_multi_size_convergence(
        tau_dict, ne_dict, fid_dict,
        title=r"Heisenberg ground state, $J = h = 1$",
        save_path=output_dir / "figure5_heisenberg_scaling.png",
    )
    print(f"  Saved {output_dir / 'figure5_heisenberg_scaling.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="TTITE simulation (Yi, Huo, Liu, Fan, Zhang & Cao, EPJ QT 2025)"
    )
    parser.add_argument(
        "--system", choices=["h2", "heisenberg"], default="h2",
        help="Test system",
    )
    parser.add_argument("--D", type=float, default=0.35, help="H2 distance (Angstrom)")
    parser.add_argument("--n", type=int, default=6, help="Heisenberg chain length")
    parser.add_argument("--J", type=float, default=1.0, help="Heisenberg coupling")
    parser.add_argument("--field", type=float, default=1.0, help="Heisenberg field h")
    parser.add_argument("--tau", type=float, default=None, help="Total imaginary time")
    parser.add_argument("--dt", type=float, default=None, help="Trotter step size")
    parser.add_argument("--order", type=int, default=10, help="Taylor expansion order R")
    parser.add_argument(
        "--figure", choices=["2", "3", "4", "5", "all"], default="all",
        help="Which paper figure to reproduce",
    )
    parser.add_argument(
        "--output", type=str, default="output/yihuoliufanzhang",
        help="Output directory",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Set defaults based on system ---
    if args.system == "h2":
        tau = args.tau if args.tau is not None else 3.0
        dt = args.dt if args.dt is not None else 0.3
    else:
        tau = args.tau if args.tau is not None else 2.0
        dt = args.dt if args.dt is not None else 0.1

    # --- Figure reproduction mode ---
    if not args.no_plots:
        if args.figure in ("2", "all"):
            reproduce_figure_2(output_dir)
        if args.figure in ("3", "all"):
            reproduce_figure_3(output_dir)
        if args.figure in ("4", "all"):
            reproduce_figure_4(output_dir)
        if args.figure in ("5", "all"):
            reproduce_figure_5(output_dir)

        # Copy PNGs next to derivation.tex so pdflatex can find them
        import shutil
        pkg_dir = Path(__file__).parent
        for png in output_dir.glob("*.png"):
            shutil.copy2(png, pkg_dir / png.name)

    # --- Single-system run (always) ---
    print(f"\n=== Running TTITE: system={args.system} ===")

    if args.system == "h2":
        ham = build_h2_hamiltonian(args.D)
        psi0 = _make_initial_state_plus(2)
        system_desc = f"H2, D={args.D}"
    else:
        ham = build_heisenberg_hamiltonian(args.n, args.J, args.field)
        psi0 = _make_initial_state_plus(args.n)
        system_desc = f"Heisenberg, n={args.n}, J={args.J}, h={args.field}"

    eigenvalues, _ = exact_diagonalization(ham)
    gs_energy = eigenvalues[0]
    print(f"  {system_desc}")
    print(f"  Exact GS energy: {gs_energy:.6f}")
    print(f"  Parameters: tau={tau}, dt={dt}, order={args.order}")

    result = run_ttite(ham, psi0, tau, dt, args.order)

    print(f"\n  TTITE final energy: {result.final_energy:.6f}")
    print(f"  TTITE final fidelity: {result.final_fidelity:.6f}")
    print(f"  Error vs exact: {result.final_energy - gs_energy:.6f}")

    # --- Save results ---
    record = SimulationRecord(
        paper="yihuoliufanzhang",
        method="ttite",
        ground_state_energy=float(result.final_energy),
        parameters=[],
        wavefunction_amplitudes=np.abs(result.final_state).tolist(),
        energy_levels=[float(result.final_energy)],
        exact_energies=[float(gs_energy)],
        convergence=result.energy_history,
        metadata={
            "system": args.system,
            "tau": tau,
            "dt": dt,
            "order": args.order,
            "fidelity": result.final_fidelity,
            "fidelity_history": result.fidelity_history,
            "tau_values": result.tau_values,
            "success_probabilities": result.success_probabilities,
        },
    )
    result_path = output_dir / "ttite_result.json"
    save_record(record, result_path)
    print(f"\n  Results saved to {result_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
