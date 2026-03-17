"""Run the Gallimore VQE simulation for charmonium spectroscopy.

Usage:
    uv run python -m quarksim.gallimore.run [OPTIONS]

Options:
    --maxiter N     Maximum VQE iterations (default: 300)
    --output DIR    Output directory for results and plots (default: output/gallimore)
    --no-plots      Skip generating plots
    --seed N        Random seed for reproducibility
"""

import argparse
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from quarksim.gallimore.ansatz import (
    build_ansatz,
    orthogonal_state,
    physical_amplitudes,
    third_state,
)
from quarksim.gallimore.hamiltonian import (
    CHARMONIUM_PARAMS,
    build_matrix,
    build_pauli_hamiltonian,
    validate_pauli_hamiltonian,
)
from quarksim.results import SimulationRecord, save_record
from quarksim.simulation import (
    VQEResult,
    physical_eigenvalues,
    run_vqe,
    run_vqe_excited,
    run_vqe_noisy,
    run_zne,
)
from quarksim.visualization import (
    plot_convergence,
    plot_energy_levels,
    plot_wavefunction,
    plot_zne,
    plot_zne_per_term,
)


def main():
    parser = argparse.ArgumentParser(
        description="VQE simulation for charmonium spectroscopy (Gallimore & Liao)"
    )
    parser.add_argument(
        "--maxiter", type=int, default=300, help="Max VQE iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/gallimore",
        help="Output directory",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--shots", type=int, default=0,
        help="Shot count per Pauli term (0 = exact statevector, >0 = shot-based)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="Depolarizing error rate per gate (0 = noiseless). Implies --shots if shots=0.",
    )
    parser.add_argument(
        "--zne", action="store_true",
        help="Run zero-noise extrapolation after VQE (requires --noise > 0).",
    )
    args = parser.parse_args()

    # Noise implies shot-based simulation
    if args.noise > 0 and args.shots == 0:
        args.shots = 8192

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # --- Build Hamiltonian ---
    print("Building Hamiltonian...")
    print(f"  Parameters: omega={CHARMONIUM_PARAMS['omega']} MeV, "
          f"kappa={CHARMONIUM_PARAMS['kappa']}, "
          f"sqrt(sigma)={np.sqrt(CHARMONIUM_PARAMS['sigma']):.1f} MeV, "
          f"mu={CHARMONIUM_PARAMS['mu']} MeV")

    h_matrix = build_matrix()
    pauli_ham = build_pauli_hamiltonian()

    # Validate: Pauli Hamiltonian matches physical matrix in 1-particle sector
    validate_pauli_hamiltonian(pauli_ham, h_matrix)
    print("  Pauli Hamiltonian validated against physical matrix.")

    # --- Exact diagonalization ---
    print("\nExact diagonalization (physical subspace):")
    exact_energies = physical_eigenvalues(pauli_ham)
    state_labels = ["1S (ground)", "2S (1st excited)", "3S (2nd excited)"]
    for label, energy in zip(state_labels, exact_energies):
        print(f"  {label}: {energy:.2f} MeV")

    # Also diagonalize the 3x3 matrix directly for comparison
    eigvals_3x3 = np.linalg.eigvalsh(h_matrix)
    print(f"\n  (3x3 matrix eigenvalues: {eigvals_3x3.round(2)})")
    print(f"  Paper expected: ~492.6 MeV (1S), ~1210.8 MeV (2S)")

    # --- VQE optimization ---
    ansatz = build_ansatz()
    if args.shots > 0:
        mode = f"shots={args.shots}"
        if args.noise > 0:
            mode += f", depolarizing_rate={args.noise}"
        print(f"\nRunning VQE ({mode}, maxiter={args.maxiter})...")
        vqe_result: VQEResult = run_vqe_noisy(
            pauli_ham,
            ansatz,
            shots=args.shots,
            depolarizing_rate=args.noise,
            method="cobyla",
            maxiter=args.maxiter,
        )
    else:
        print(f"\nRunning VQE (statevector, maxiter={args.maxiter})...")
        vqe_result: VQEResult = run_vqe(
            pauli_ham, ansatz, method="cobyla", maxiter=args.maxiter
        )

    alpha, beta = vqe_result.parameters
    amps = physical_amplitudes(vqe_result.wavefunction)

    print(f"  VQE energy: {vqe_result.energy:.2f} MeV")
    print(f"  Parameters: alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"  Wavefunction: {amps[0]:.4f}|0> + {amps[1]:.4f}|1> + {amps[2]:.4f}|2>")
    print(f"  Function evaluations: {vqe_result.num_evaluations}")
    print(f"  Error vs exact: {vqe_result.energy - exact_energies[0]:.4f} MeV")

    # --- Excited states via orthogonalization (Section II.D) ---
    print("\nEstimating excited states via orthogonalization...")

    # 1st excited state: minimize over gamma in the subspace orthogonal to ground state
    alpha0, beta0 = vqe_result.parameters

    def build_2s_state(gamma):
        return orthogonal_state(alpha0, beta0, gamma)

    excited_1 = run_vqe_excited(pauli_ham, build_2s_state)
    gamma_opt = excited_1.parameters[0]
    amps_2s = physical_amplitudes(excited_1.wavefunction)

    print(f"  2S energy: {excited_1.energy:.2f} MeV  (exact: {exact_energies[1]:.2f})")
    print(f"  gamma = {gamma_opt:.4f}")
    print(f"  Wavefunction: {amps_2s[0]:.4f}|0> + {amps_2s[1]:.4f}|1> + {amps_2s[2]:.4f}|2>")
    print(f"  Error vs exact: {excited_1.energy - exact_energies[1]:.4f} MeV")

    # 2nd excited state: fully determined as the remaining orthogonal direction
    sv_3s = third_state(vqe_result.wavefunction, excited_1.wavefunction)
    energy_3s = Statevector(sv_3s).expectation_value(pauli_ham).real
    amps_3s = physical_amplitudes(sv_3s)

    print(f"\n  3S energy: {energy_3s:.2f} MeV  (exact: {exact_energies[2]:.2f})")
    print(f"  Wavefunction: {amps_3s[0]:.4f}|0> + {amps_3s[1]:.4f}|1> + {amps_3s[2]:.4f}|2>")
    print(f"  Error vs exact: {energy_3s - exact_energies[2]:.4f} MeV")

    vqe_energies = [vqe_result.energy, excited_1.energy, energy_3s]

    # --- Zero-noise extrapolation (Section II.E) ---
    zne_result = None
    if args.zne:
        if args.noise <= 0:
            print("\nWarning: --zne requires --noise > 0. Skipping ZNE.")
        else:
            # The paper's workflow: VQE finds good params (robust to moderate noise),
            # then ZNE corrects the energy estimate. If the noisy VQE found bad params
            # (high noise), fall back to noiseless-optimal params for the ZNE demo.
            zne_params = vqe_result.parameters
            noiseless_result = run_vqe(pauli_ham, ansatz, method="cobyla", maxiter=300)
            noiseless_energy_at_params = Statevector(
                ansatz.assign_parameters(zne_params)
            ).expectation_value(pauli_ham).real

            # If noisy VQE drifted >5% from exact, use noiseless params instead
            if abs(noiseless_energy_at_params - exact_energies[0]) > 0.05 * abs(exact_energies[0]):
                print(f"\n  Noisy VQE params far from optimum — using noiseless-optimal params for ZNE.")
                zne_params = noiseless_result.parameters

            noise_rate = args.noise
            zne_shots = max(args.shots, 32768)  # ZNE needs good statistics

            print(f"\nRunning zero-noise extrapolation (noise={noise_rate}, shots={zne_shots})...")
            zne_result = run_zne(
                pauli_ham,
                ansatz,
                zne_params,
                shots=zne_shots,
                depolarizing_rate=noise_rate,
            )
            print(f"\n  ZNE extrapolated energy: {zne_result.energy:.1f} +/- {2*zne_result.energy_std:.1f} MeV (2sigma)")
            print(f"  Raw noisy energies: {', '.join(f'lambda={l}: {e:.1f}' for l, e in sorted(zne_result.raw_energies.items()))}")
            print(f"  Exact energy: {exact_energies[0]:.1f} MeV")
            print(f"  ZNE error: {zne_result.energy - exact_energies[0]:.1f} MeV")

    # --- Save results ---
    record = SimulationRecord(
        paper="gallimore",
        method="vqe_statevector",
        ground_state_energy=float(vqe_result.energy),
        parameters=vqe_result.parameters.tolist(),
        wavefunction_amplitudes=amps.tolist(),
        energy_levels=[float(e) for e in vqe_energies],
        exact_energies=exact_energies.tolist(),
        convergence=vqe_result.convergence,
        metadata={
            "maxiter": args.maxiter,
            "num_evaluations": vqe_result.num_evaluations,
            "optimizer_message": vqe_result.optimizer_message,
            "physics_params": CHARMONIUM_PARAMS,
            "excited_states": {
                "2S": {
                    "energy": float(excited_1.energy),
                    "gamma": float(gamma_opt),
                    "wavefunction": amps_2s.tolist(),
                },
                "3S": {
                    "energy": float(energy_3s),
                    "wavefunction": amps_3s.tolist(),
                },
            },
        },
    )
    result_path = output_dir / "vqe_result.json"
    save_record(record, result_path)
    print(f"\n  Results saved to {result_path}")

    # --- Plots ---
    if not args.no_plots:
        print("\nGenerating plots...")

        plot_convergence(
            vqe_result.convergence,
            exact_energy=exact_energies[0],
            title="VQE Convergence — Charmonium Ground State",
            save_path=output_dir / "convergence.png",
        )
        print(f"  Saved {output_dir / 'convergence.png'}")

        plot_energy_levels(
            {
                "Exact diag.": exact_energies.tolist(),
                "VQE": vqe_energies,
                "Paper (expected)": [492.6, 1210.8],
            },
            title="Charmonium Energy Levels",
            save_path=output_dir / "energy_levels.png",
        )
        print(f"  Saved {output_dir / 'energy_levels.png'}")

        plot_wavefunction(
            amps,
            labels=["|0> (1s)", "|1> (2s)", "|2> (3s)"],
            title="Ground State Wavefunction — VQE",
            save_path=output_dir / "wavefunction.png",
        )
        print(f"  Saved {output_dir / 'wavefunction.png'}")

        if zne_result is not None:
            plot_zne(
                zne_result,
                exact_energy=exact_energies[0],
                title="Zero-Noise Extrapolation — Ground State",
                save_path=output_dir / "zne_extrapolation.png",
            )
            print(f"  Saved {output_dir / 'zne_extrapolation.png'}")

            # Compute noiseless exact <P> for each term at the ZNE params
            zne_sv = Statevector(ansatz.assign_parameters(zne_result.params_used))
            exact_term_values = {}
            for pauli in pauli_ham.paulis:
                label = pauli.to_label()
                if not all(c == "I" for c in label):
                    exact_term_values[label] = zne_sv.expectation_value(pauli).real

            plot_zne_per_term(
                zne_result,
                exact_term_values=exact_term_values,
                title="ZNE Per Pauli Term (Eqs. 12-20)",
                save_path=output_dir / "zne_per_term.png",
            )
            print(f"  Saved {output_dir / 'zne_per_term.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
