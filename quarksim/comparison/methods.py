"""Unified wrappers for VQE, VQITE, and TTITE on the Woloshyn Hamiltonian.

Each wrapper returns a common MethodResult dataclass so results are
directly comparable across methods.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from quarksim.comparison.config import ExperimentConfig
from quarksim.simulation import (
    exact_diagonalization,
    run_vqe,
    run_vqe_noisy as _sim_run_vqe_noisy,
    make_noise_model,
)
from quarksim.woloshyn.ansatz import build_ansatz
from quarksim.woloshyn.hamiltonian import build_pauli_hamiltonian, get_matrix
from quarksim.woloshyn.vqite import run_vqite, run_vqite_excited as _woloshyn_vqite_excited
from quarksim.yihuoliufanzhang.evolution import run_ttite


@dataclass
class MethodResult:
    """Common result type for all three methods."""

    method: str            # "VQE", "VQITE", "TTITE"
    channel: str           # "1S0", "3S1", "1P1"
    energy: float          # Ground-state energy (fm^-1)
    exact_energy: float    # Exact eigenvalue for comparison
    error: float           # energy - exact_energy
    convergence: list[float] = field(default_factory=list)
    wavefunction: np.ndarray = field(default_factory=lambda: np.array([]))
    fidelity: float = 0.0  # |<psi|gs>|^2
    wall_time: float = 0.0
    excited_energies: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _compute_fidelity(state: np.ndarray, ground_state: np.ndarray) -> float:
    """Compute |<psi|gs>|^2."""
    return float(abs(np.vdot(ground_state, state)) ** 2)


def _exact_ground_state(hamiltonian: SparsePauliOp) -> tuple[float, np.ndarray, list[float]]:
    """Return (ground energy, ground state vector, all eigenvalues)."""
    eigenvalues, eigenvectors = exact_diagonalization(hamiltonian)
    return float(eigenvalues[0]), eigenvectors[:, 0], [float(e) for e in eigenvalues]


# ---------------------------------------------------------------------------
# Noiseless methods
# ---------------------------------------------------------------------------

def run_vqe_ideal(channel: str, cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run noiseless VQE on the Woloshyn Hamiltonian."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    rng = np.random.default_rng(cfg.seed)
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

    t0 = time.perf_counter()
    result = run_vqe(ham, ansatz, x0=x0, method=cfg.vqe_method, maxiter=cfg.vqe_maxiter)
    wall = time.perf_counter() - t0

    fid = _compute_fidelity(result.wavefunction, gs_state)

    return MethodResult(
        method="VQE",
        channel=channel,
        energy=result.energy,
        exact_energy=gs_energy,
        error=result.energy - gs_energy,
        convergence=result.convergence,
        wavefunction=result.wavefunction,
        fidelity=fid,
        wall_time=wall,
        metadata={
            "parameters": result.parameters.tolist(),
            "num_evaluations": result.num_evaluations,
            "optimizer": cfg.vqe_method,
            "exact_eigenvalues": all_eigs,
        },
    )


def run_vqite_ideal(channel: str, cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run noiseless VQITE on the Woloshyn Hamiltonian."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    theta0 = np.full(ansatz.num_parameters, 0.5)

    t0 = time.perf_counter()
    result = run_vqite(
        ham, ansatz,
        n_steps=cfg.vqite_n_steps,
        dtau=cfg.vqite_dtau,
        theta0=theta0,
        regularization=cfg.vqite_regularization,
    )
    wall = time.perf_counter() - t0

    fid = _compute_fidelity(result.wavefunction, gs_state)

    return MethodResult(
        method="VQITE",
        channel=channel,
        energy=result.energy,
        exact_energy=gs_energy,
        error=result.energy - gs_energy,
        convergence=result.energy_history,
        wavefunction=result.wavefunction,
        fidelity=fid,
        wall_time=wall,
        metadata={
            "parameters": result.parameters.tolist(),
            "n_steps": cfg.vqite_n_steps,
            "dtau": cfg.vqite_dtau,
            "exact_eigenvalues": all_eigs,
        },
    )


def run_ttite_ideal(channel: str, cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run noiseless TTITE on the Woloshyn Hamiltonian."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    # Equal superposition initial state (nonzero overlap with ground state)
    initial_state = np.ones(4, dtype=complex) / 2.0

    t0 = time.perf_counter()
    result = run_ttite(
        ham, initial_state,
        tau_total=cfg.ttite_tau_total,
        dt=cfg.ttite_dt,
        order=cfg.ttite_order,
    )
    wall = time.perf_counter() - t0

    fid = float(result.final_fidelity)

    return MethodResult(
        method="TTITE",
        channel=channel,
        energy=result.final_energy,
        exact_energy=gs_energy,
        error=result.final_energy - gs_energy,
        convergence=result.energy_history,
        wavefunction=result.final_state,
        fidelity=fid,
        wall_time=wall,
        metadata={
            "tau_total": cfg.ttite_tau_total,
            "dt": cfg.ttite_dt,
            "order": cfg.ttite_order,
            "tau_values": result.tau_values,
            "fidelity_history": result.fidelity_history,
            "success_probabilities": result.success_probabilities,
            "exact_eigenvalues": all_eigs,
        },
    )


# ---------------------------------------------------------------------------
# Excited states (VQE + VQITE only; TTITE does not support excited states)
# ---------------------------------------------------------------------------

def run_vqe_excited(
    channel: str,
    n_states: int = 4,
    cfg: ExperimentConfig | None = None,
) -> list[MethodResult]:
    """Run VQE for ground + excited states using the penalty method.

    Returns a list of MethodResult, one per eigenstate.
    """
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    rng = np.random.default_rng(cfg.seed)
    results = []
    found_states: list[np.ndarray] = []

    for k in range(min(n_states, len(all_eigs))):
        x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

        t0 = time.perf_counter()
        if k == 0:
            vqe_res = run_vqe(ham, ansatz, x0=x0, method=cfg.vqe_method, maxiter=cfg.vqe_maxiter)
        else:
            vqe_res = _run_vqe_penalized(
                ham, ansatz, found_states, x0=x0,
                alpha=cfg.penalty_alpha,
                method=cfg.vqe_method,
                maxiter=cfg.vqe_maxiter,
            )
        wall = time.perf_counter() - t0

        found_states.append(vqe_res.wavefunction)
        fid = _compute_fidelity(vqe_res.wavefunction, gs_state) if k == 0 else 0.0

        results.append(MethodResult(
            method="VQE",
            channel=channel,
            energy=vqe_res.energy,
            exact_energy=all_eigs[k],
            error=vqe_res.energy - all_eigs[k],
            convergence=vqe_res.convergence,
            wavefunction=vqe_res.wavefunction,
            fidelity=fid,
            wall_time=wall,
            metadata={"state_index": k, "parameters": vqe_res.parameters.tolist()},
        ))

    return results


def _run_vqe_penalized(
    hamiltonian: SparsePauliOp,
    ansatz,
    lower_states: list[np.ndarray],
    x0: np.ndarray,
    alpha: float = 10.0,
    method: str = "cobyla",
    maxiter: int = 300,
):
    """VQE with penalty terms to avoid previously found states."""
    from scipy.optimize import minimize
    from quarksim.simulation import VQEResult

    energies: list[float] = []

    def cost_fn(params):
        bound = ansatz.assign_parameters(params)
        sv = np.array(Statevector(bound))
        e = Statevector(sv).expectation_value(hamiltonian).real
        for phi in lower_states:
            overlap_sq = abs(np.vdot(phi, sv)) ** 2
            e += alpha * overlap_sq
        energies.append(e)
        return e

    result = minimize(cost_fn, x0, method=method, options={"maxiter": maxiter})

    bound = ansatz.assign_parameters(result.x)
    sv = np.array(Statevector(bound))
    # Report actual Hamiltonian energy (without penalty)
    actual_energy = Statevector(sv).expectation_value(hamiltonian).real

    return VQEResult(
        energy=actual_energy,
        parameters=result.x,
        wavefunction=sv,
        convergence=energies,
        num_evaluations=result.nfev,
        optimizer_message=str(result.message),
    )


def run_vqite_all_excited(
    channel: str,
    n_states: int = 4,
    cfg: ExperimentConfig | None = None,
) -> list[MethodResult]:
    """Run VQITE for ground + excited states using the penalty method."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    results = []
    found_states: list[np.ndarray] = []

    for k in range(min(n_states, len(all_eigs))):
        theta0 = np.full(ansatz.num_parameters, 0.5)

        t0 = time.perf_counter()
        if k == 0:
            vqite_res = run_vqite(
                ham, ansatz,
                n_steps=cfg.vqite_n_steps,
                dtau=cfg.vqite_dtau,
                theta0=theta0,
                regularization=cfg.vqite_regularization,
            )
        else:
            vqite_res = _woloshyn_vqite_excited(
                ham, ansatz, found_states,
                alpha=cfg.penalty_alpha,
                n_steps=cfg.vqite_n_steps,
                dtau=cfg.vqite_dtau,
                theta0=theta0,
                regularization=cfg.vqite_regularization,
            )
        wall = time.perf_counter() - t0

        found_states.append(vqite_res.wavefunction)
        fid = _compute_fidelity(vqite_res.wavefunction, gs_state) if k == 0 else 0.0

        results.append(MethodResult(
            method="VQITE",
            channel=channel,
            energy=vqite_res.energy,
            exact_energy=all_eigs[k],
            error=vqite_res.energy - all_eigs[k],
            convergence=vqite_res.energy_history,
            wavefunction=vqite_res.wavefunction,
            fidelity=fid,
            wall_time=wall,
            metadata={"state_index": k, "parameters": vqite_res.parameters.tolist()},
        ))

    return results


# ---------------------------------------------------------------------------
# Noisy methods
# ---------------------------------------------------------------------------

def run_vqe_noisy(
    channel: str,
    depolarizing_rate: float,
    cfg: ExperimentConfig | None = None,
) -> MethodResult:
    """Run VQE with shot-based measurement and depolarizing noise."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    rng = np.random.default_rng(cfg.seed)
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

    t0 = time.perf_counter()
    result = _sim_run_vqe_noisy(
        ham, ansatz,
        shots=cfg.vqe_shots,
        depolarizing_rate=depolarizing_rate,
        x0=x0,
        method=cfg.vqe_method,
        maxiter=cfg.vqe_maxiter,
    )
    wall = time.perf_counter() - t0

    fid = _compute_fidelity(result.wavefunction, gs_state)

    return MethodResult(
        method="VQE",
        channel=channel,
        energy=result.energy,
        exact_energy=gs_energy,
        error=result.energy - gs_energy,
        convergence=result.convergence,
        wavefunction=result.wavefunction,
        fidelity=fid,
        wall_time=wall,
        metadata={
            "depolarizing_rate": depolarizing_rate,
            "shots": cfg.vqe_shots,
            "parameters": result.parameters.tolist(),
        },
    )


def run_vqite_noisy(
    channel: str,
    depolarizing_rate: float,
    cfg: ExperimentConfig | None = None,
) -> MethodResult:
    """Run VQITE with noisy energy evaluations in the gradient loop.

    The energy gradient C_i = -dE/dtheta_i is computed via parameter-shift
    with shot-based, noisy energy measurements. The metric tensor remains
    statevector-based (noiseless) since it uses overlap functions, not
    energy measurements.
    """
    from qiskit_aer import AerSimulator
    from quarksim.simulation import _estimate_energy_shot_based
    from quarksim.woloshyn.vqite import compute_metric_tensor

    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    if depolarizing_rate > 0:
        noise_model = make_noise_model(depolarizing_rate)
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = AerSimulator()

    n_params = ansatz.num_parameters
    theta = np.full(n_params, 0.5)
    shots = cfg.vqe_shots
    energy_history: list[float] = []

    def noisy_energy(params):
        bound = ansatz.assign_parameters(params)
        return _estimate_energy_shot_based(bound, ham, backend, shots)

    t0 = time.perf_counter()
    for step in range(cfg.vqite_n_steps):
        e = noisy_energy(theta)
        energy_history.append(e)

        # Noisy gradient via parameter-shift
        C = np.zeros(n_params)
        for k in range(n_params):
            shift = np.zeros(n_params)
            shift[k] = np.pi / 2
            e_plus = noisy_energy(theta + shift)
            e_minus = noisy_energy(theta - shift)
            C[k] = -(e_plus - e_minus) / 2.0

        # Noiseless metric tensor (statevector-based)
        A = compute_metric_tensor(ansatz, theta)
        A_reg = A + cfg.vqite_regularization * np.eye(n_params)

        theta_dot = np.linalg.solve(A_reg, C)
        theta = theta + cfg.vqite_dtau * theta_dot

    # Final energy (noisy)
    final_energy = noisy_energy(theta)
    energy_history.append(final_energy)
    wall = time.perf_counter() - t0

    # Final wavefunction (noiseless, for fidelity computation)
    from qiskit.quantum_info import Statevector as SV
    final_sv = np.array(SV(ansatz.assign_parameters(theta)))
    fid = _compute_fidelity(final_sv, gs_state)

    return MethodResult(
        method="VQITE",
        channel=channel,
        energy=final_energy,
        exact_energy=gs_energy,
        error=final_energy - gs_energy,
        convergence=energy_history,
        wavefunction=final_sv,
        fidelity=fid,
        wall_time=wall,
        metadata={
            "depolarizing_rate": depolarizing_rate,
            "shots": shots,
            "parameters": theta.tolist(),
        },
    )


def run_ttite_noisy(
    channel: str,
    depolarizing_rate: float,
    cfg: ExperimentConfig | None = None,
) -> MethodResult:
    """Run TTITE with noisy LCU circuits via density-matrix simulation.

    For each Trotter step:
      1. Build the 3-qubit LCU circuit (1 ancilla + 2 work)
      2. Initialize ancilla in |0>, work qubits in current state rho
      3. Run through noisy density-matrix simulator
      4. Postselect ancilla = |0>, trace out, renormalize
    """
    from quarksim.yihuoliufanzhang.circuit import taylor_coefficients
    from quarksim.yihuoliufanzhang.hamiltonian import decompose_to_pauli_terms

    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)
    H_matrix = ham.to_matrix()
    terms = decompose_to_pauli_terms(ham)

    L = int(round(cfg.ttite_tau_total / cfg.ttite_dt))
    n_work = 2

    # Start with pure state density matrix
    state = np.ones(4, dtype=complex) / 2.0
    rho = np.outer(state, state.conj())

    energy_history = [float(np.real(np.trace(H_matrix @ rho)))]
    fidelity_history = [float(np.real(gs_state.conj() @ rho @ gs_state))]

    t0 = time.perf_counter()
    for seg in range(L):
        for c_i, label, mat in terms:
            is_identity = all(p == "I" for p in label)
            if is_identity:
                # Scalar factor: e^{-c0*dt} absorbed in normalization
                factor = np.exp(-c_i * cfg.ttite_dt)
                rho = rho * (factor ** 2)
                tr = np.real(np.trace(rho))
                if tr > 1e-30:
                    rho = rho / tr
                continue

            alpha, beta = taylor_coefficients(c_i, cfg.ttite_dt, cfg.ttite_order)

            # Apply T_i = alpha*I + beta*h_i on the density matrix,
            # then model gate noise as an effective depolarizing channel.
            T_i = alpha * np.eye(4) + beta * mat
            rho_new = T_i @ rho @ T_i.conj().T

            # Apply depolarizing noise: rho -> (1-p)*rho + p*I/d
            # The effective noise per Trotter step accounts for the LCU circuit
            # having ~n_work controlled-Pauli gates + 2 single-qubit gates
            # We model this as an effective depolarizing channel on the work
            # system, with rate proportional to the gate count.
            n_active_gates = sum(1 for p in label if p != "I")
            # Each controlled-Pauli = 1 two-qubit gate; plus Ry + H on ancilla = 2 single-qubit
            effective_rate = 1.0 - (1.0 - depolarizing_rate) ** (n_active_gates + 2)
            d = 4  # dimension of work system
            rho_new = (1.0 - effective_rate) * rho_new + effective_rate * np.eye(d) * np.trace(rho_new) / d

            tr = np.real(np.trace(rho_new))
            if tr > 1e-30:
                rho = rho_new / tr
            else:
                break  # State collapsed

        energy = float(np.real(np.trace(H_matrix @ rho)))
        fid = float(np.real(gs_state.conj() @ rho @ gs_state))
        energy_history.append(energy)
        fidelity_history.append(fid)

    wall = time.perf_counter() - t0

    # Extract approximate pure state (leading eigenvector of rho)
    evals, evecs = np.linalg.eigh(rho)
    approx_state = evecs[:, -1]  # largest eigenvalue

    return MethodResult(
        method="TTITE",
        channel=channel,
        energy=energy_history[-1],
        exact_energy=gs_energy,
        error=energy_history[-1] - gs_energy,
        convergence=energy_history,
        wavefunction=approx_state,
        fidelity=fidelity_history[-1],
        wall_time=wall,
        metadata={
            "depolarizing_rate": depolarizing_rate,
            "tau_values": [i * cfg.ttite_dt for i in range(L + 1)],
            "fidelity_history": fidelity_history,
            "exact_eigenvalues": all_eigs,
        },
    )
