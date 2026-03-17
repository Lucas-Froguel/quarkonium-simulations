"""Run the Woloshyn VQITE simulation for charmonium spectroscopy.

Usage:
    uv run python -m quarksim.woloshyn.run [OPTIONS]

Options:
    --channel {1S0,3S1,1P1,all}  Channel to simulate (default: all)
    --n-steps N                   VQITE steps (default: 50)
    --dtau F                      Step size (default: 0.02)
    --output DIR                  Output directory (default: output/woloshyn)
    --no-plots                    Skip generating plots
    --seed N                      Random seed
    --transitions                 Compute M1/E1 transition amplitudes
"""

import argparse
from pathlib import Path

import numpy as np

from quarksim.woloshyn.ansatz import build_ansatz, physical_amplitudes
from quarksim.woloshyn.hamiltonian import (
    CHANNELS,
    CHARMONIUM_PARAMS,
    PAPER_EIGENVALUES,
    build_pauli_hamiltonian,
    get_matrix,
    validate_pauli_hamiltonian,
)
from quarksim.woloshyn.transitions import (
    e1_amplitude,
    e1_matrix_elements,
    m1_overlap,
)
from quarksim.woloshyn.vqite import run_vqite, run_vqite_excited
from quarksim.results import SimulationRecord, save_record
from quarksim.visualization import (
    plot_energy_levels,
    plot_vqite_convergence,
    plot_wavefunction,
)


def run_channel(channel: str, ansatz, args) -> dict:
    """Run full VQITE simulation for a single channel."""
    print(f"\n{'='*60}")
    print(f"Channel: {channel}")
    print(f"{'='*60}")

    # --- Build Hamiltonian ---
    h_matrix = get_matrix(channel)
    pauli_ham = build_pauli_hamiltonian(channel)
    validate_pauli_hamiltonian(pauli_ham, h_matrix)

    # --- Exact diagonalization ---
    eigvals = np.linalg.eigvalsh(h_matrix)
    exact_paper = PAPER_EIGENVALUES[channel]
    print(f"\nExact diagonalization (fm^-1):")
    for i, (e, ep) in enumerate(zip(eigvals, exact_paper)):
        print(f"  State {i+1}: {e:.4f}  (paper: {ep:.3f})")

    # --- VQITE ground state ---
    print(f"\nRunning VQITE ground state ({args.n_steps} steps, dtau={args.dtau})...")
    gs = run_vqite(pauli_ham, ansatz, n_steps=args.n_steps, dtau=args.dtau)
    amps_gs = physical_amplitudes(gs.wavefunction)
    print(f"  Energy: {gs.energy:.4f} fm^-1  (exact: {eigvals[0]:.4f})")
    print(f"  Wavefunction: {' + '.join(f'{a:.4f}|{i}>' for i, a in enumerate(amps_gs))}")

    # --- VQITE excited states ---
    print(f"\nRunning VQITE excited states...")
    states = [gs]
    energy_histories = [gs.energy_history]

    for n_exc in range(1, 4):
        lower_svs = [s.wavefunction for s in states]
        exc = run_vqite_excited(
            pauli_ham, ansatz, lower_svs,
            alpha=10.0, n_steps=args.n_steps + 10, dtau=args.dtau,
        )
        states.append(exc)
        energy_histories.append(exc.energy_history)
        print(f"  State {n_exc+1}: {exc.energy:.4f} fm^-1  (exact: {eigvals[n_exc]:.4f})")

    vqite_energies = [s.energy for s in states]

    return {
        "channel": channel,
        "eigvals": eigvals,
        "states": states,
        "vqite_energies": vqite_energies,
        "energy_histories": energy_histories,
    }


def run_transitions(results: dict) -> None:
    """Compute and print M1 and E1 transition amplitudes."""
    print(f"\n{'='*60}")
    print("Transition Amplitudes")
    print(f"{'='*60}")

    channels = {r["channel"]: r for r in results}

    # --- M1 transitions (3S1 <-> 1S0) ---
    if "3S1" in channels and "1S0" in channels:
        print("\nM1 transitions (3S1 -> 1S0):")
        print(f"  {'Transition':<20} {'|overlap|^2':>12}  {'Paper (exact)':>14}")
        print(f"  {'-'*50}")

        s1_states = channels["3S1"]["states"]
        s0_states = channels["1S0"]["states"]

        # Paper Table 4 expected values (exact column)
        paper_m1 = {
            (1, 1): 0.9826,
            (2, 1): 0.0107,
            (2, 2): 0.9781,
            (3, 2): 0.0061,
        }

        for (i3, i1), expected in paper_m1.items():
            if i3 <= len(s1_states) and i1 <= len(s0_states):
                overlap = m1_overlap(s1_states[i3-1].wavefunction, s0_states[i1-1].wavefunction)
                print(f"  {i3}^3S_1 -> {i1}^1S_0  {overlap:12.4f}  {expected:14.4f}")

        # Cross-channel: 1S0 -> 3S1
        paper_m1_cross = {
            (2, 1): 0.0123,
            (3, 1): 0.0025,
        }
        print()
        for (i0, i1), expected in paper_m1_cross.items():
            if i0 <= len(s0_states) and i1 <= len(s1_states):
                overlap = m1_overlap(s0_states[i0-1].wavefunction, s1_states[i1-1].wavefunction)
                print(f"  {i0}^1S_0 -> {i1}^3S_1  {overlap:12.4f}  {expected:14.4f}")

    # --- E1 transitions (1P1 <-> 1S0) ---
    if "1P1" in channels and "1S0" in channels:
        print("\nE1 transitions (radial matrix element <r> in fm):")
        print(f"  {'Transition':<20} {'<r> (fm)':>12}  {'Paper (w.f.)':>14}  {'Paper (exact)':>14}")
        print(f"  {'-'*65}")

        p1_states = channels["1P1"]["states"]
        s0_states = channels["1S0"]["states"]
        r_matrix = e1_matrix_elements()

        # Paper Table 5: (P_index, S_index, label, wf_sim_value, exact_value)
        # <P_state|r|S_state> where indices are 1-based
        e1_transitions = [
            (1, 1, "1^1P_1 -> 1^1S_0", 0.3925, 0.3490),
            (2, 2, "2^1P_1 -> 2^1S_0", 0.7174, 0.5900),
            (1, 2, "2^1S_0 -> 1^1P_1", 0.5406, 0.5757),
            (2, 3, "3^1S_0 -> 2^1P_1", 0.8227, 0.8873),
            (1, 3, "3^1S_0 -> 1^1P_1", 0.0448, 0.0311),
        ]

        for ip, is_, label, wf_sim, exact in e1_transitions:
            if ip <= len(p1_states) and is_ <= len(s0_states):
                amp = e1_amplitude(
                    s0_states[is_-1].wavefunction,
                    p1_states[ip-1].wavefunction,
                    r_matrix,
                )
                print(f"  {label:<20} {abs(amp):12.4f}  {wf_sim:14.4f}  {exact:14.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="VQITE simulation for charmonium spectroscopy (Woloshyn)"
    )
    parser.add_argument(
        "--channel", type=str, default="all",
        choices=["1S0", "3S1", "1P1", "all"],
        help="Channel to simulate",
    )
    parser.add_argument("--n-steps", type=int, default=50, help="VQITE steps")
    parser.add_argument("--dtau", type=float, default=0.02, help="Step size")
    parser.add_argument(
        "--output", type=str, default="output/woloshyn", help="Output directory",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--transitions", action="store_true",
        help="Compute M1/E1 transition amplitudes (requires all channels)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # --- Parameters ---
    print("Woloshyn — Demonstrating quantum computing with the quark model")
    print(f"  Parameters: alpha_s={CHARMONIUM_PARAMS['alpha_s']}, "
          f"b={CHARMONIUM_PARAMS['b']} GeV^2, "
          f"m_c={CHARMONIUM_PARAMS['m_c']} GeV, "
          f"sigma={CHARMONIUM_PARAMS['sigma_smear']} GeV")
    print(f"  omega={CHARMONIUM_PARAMS['omega']} fm^-1, N_basis=4")

    ansatz = build_ansatz()
    channels = CHANNELS if args.channel == "all" else [args.channel]

    # --- Run channels ---
    all_results = []
    for channel in channels:
        result = run_channel(channel, ansatz, args)
        all_results.append(result)

    # --- Transitions ---
    if args.transitions:
        if set(channels) >= {"1S0", "3S1", "1P1"}:
            run_transitions(all_results)
        else:
            print("\nWarning: --transitions requires --channel all")

    # --- Save results ---
    for r in all_results:
        gs = r["states"][0]
        record = SimulationRecord(
            paper="woloshyn",
            method="vqite_statevector",
            ground_state_energy=float(gs.energy),
            parameters=gs.parameters.tolist(),
            wavefunction_amplitudes=physical_amplitudes(gs.wavefunction).tolist(),
            energy_levels=[float(e) for e in r["vqite_energies"]],
            exact_energies=r["eigvals"].tolist(),
            convergence=gs.energy_history,
            metadata={
                "channel": r["channel"],
                "n_steps": args.n_steps,
                "dtau": args.dtau,
                "physics_params": CHARMONIUM_PARAMS,
                "excited_states": {
                    f"state_{i+2}": {
                        "energy": float(r["states"][i+1].energy),
                        "wavefunction": physical_amplitudes(r["states"][i+1].wavefunction).tolist(),
                    }
                    for i in range(len(r["states"]) - 1)
                },
            },
        )
        result_path = output_dir / f"vqite_result_{r['channel']}.json"
        save_record(record, result_path)
        print(f"\n  Results saved to {result_path}")

    # --- Plots ---
    if not args.no_plots:
        print("\nGenerating plots...")
        import matplotlib
        matplotlib.use("Agg")

        for r in all_results:
            ch = r["channel"]
            exact = r["eigvals"].tolist()

            plot_vqite_convergence(
                r["energy_histories"],
                exact_energies=exact,
                state_labels=[f"State {i+1}" for i in range(4)],
                title=f"VQITE Convergence — Charmonium {ch}",
                save_path=output_dir / f"vqite_convergence_{ch}.png",
            )
            print(f"  Saved {output_dir / f'vqite_convergence_{ch}.png'}")

            plot_energy_levels(
                {
                    "Exact diag.": exact,
                    "VQITE": r["vqite_energies"],
                    f"Paper (Table 3)": PAPER_EIGENVALUES[ch],
                },
                title=f"Energy Levels — {ch}",
                ylabel=r"Energy [fm$^{-1}$]",
                save_path=output_dir / f"energy_levels_{ch}.png",
            )
            print(f"  Saved {output_dir / f'energy_levels_{ch}.png'}")

            # Ground state wavefunction
            gs_amps = physical_amplitudes(r["states"][0].wavefunction)
            plot_wavefunction(
                gs_amps,
                labels=[f"|{i}>" for i in range(4)],
                title=f"Ground State Wavefunction — {ch}",
                save_path=output_dir / f"wavefunction_{ch}.png",
            )
            print(f"  Saved {output_dir / f'wavefunction_{ch}.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
