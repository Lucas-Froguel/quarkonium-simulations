"""Unified circuit-based wrappers for VQE, VQITE, and TTITE.

ALL quantum computations run through AerSimulator:
- VQE: shot-based Pauli measurement for energy
- VQITE: shot-based energy gradient + shot-based overlap circuits for metric tensor
- TTITE: LCU circuits on AerSimulator(method='statevector'/'density_matrix'),
         shot-based Pauli measurement for final energy
"""

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import transpile

from quarksim.comparison.config import ExperimentConfig
from quarksim.simulation import (
    exact_diagonalization,
    run_vqe_noisy as _sim_run_vqe_noisy,
    make_noise_model,
    _estimate_energy_shot_based,
    _measure_pauli_term,
)
from quarksim.woloshyn.ansatz import build_ansatz
from quarksim.woloshyn.hamiltonian import build_pauli_hamiltonian
from quarksim.yihuoliufanzhang.circuit import (
    taylor_coefficients,
    build_trotter_step_circuit,
)
from quarksim.yihuoliufanzhang.hamiltonian import decompose_to_pauli_terms


@dataclass
class MethodResult:
    """Common result type for all three methods."""

    method: str
    channel: str
    energy: float
    exact_energy: float
    error: float
    convergence: list[float] = field(default_factory=list)
    wavefunction: np.ndarray = field(default_factory=lambda: np.array([]))
    fidelity: float = 0.0
    wall_time: float = 0.0
    excited_energies: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _compute_fidelity(state: np.ndarray, ground_state: np.ndarray) -> float:
    return float(abs(np.vdot(ground_state, state)) ** 2)


def _exact_ground_state(hamiltonian: SparsePauliOp):
    eigenvalues, eigenvectors = exact_diagonalization(hamiltonian)
    return float(eigenvalues[0]), eigenvectors[:, 0], [float(e) for e in eigenvalues]


def _make_backend(depolarizing_rate: float = 0.0, method: str = "automatic"):
    """Create an AerSimulator backend with optional noise."""
    from qiskit_aer import AerSimulator

    kwargs = {}
    if method != "automatic":
        kwargs["method"] = method
    if depolarizing_rate > 0:
        kwargs["noise_model"] = make_noise_model(depolarizing_rate)
    return AerSimulator(**kwargs)


# ---------------------------------------------------------------------------
# Circuit-based energy measurement (shared by all methods)
# ---------------------------------------------------------------------------

def _circuit_energy(ansatz: QuantumCircuit, params: np.ndarray,
                    hamiltonian: SparsePauliOp, backend, shots: int) -> float:
    """Measure energy via shot-based Pauli circuits on AerSimulator."""
    bound = ansatz.assign_parameters(params)
    return _estimate_energy_shot_based(bound, hamiltonian, backend, shots)


# ---------------------------------------------------------------------------
# Circuit-based overlap measurement (for VQITE metric tensor)
# ---------------------------------------------------------------------------

def _circuit_overlap_sq(ansatz: QuantumCircuit, params_ref: np.ndarray,
                        params_shifted: np.ndarray, backend, shots: int) -> float:
    """Measure |<psi_ref|psi_shifted>|^2 via the overlap circuit on AerSimulator.

    Builds U(params_ref) followed by U(params_shifted)^dagger.
    P(|00>) = |<psi_ref|psi_shifted>|^2.
    """
    circ_ref = ansatz.assign_parameters(params_ref)
    circ_shifted = ansatz.assign_parameters(params_shifted)

    circ = circ_ref.compose(circ_shifted.inverse())
    circ.measure_all()

    transpiled = transpile(circ, backend, optimization_level=0)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # P(00) = counts of all-zero bitstring / total shots
    n_qubits = ansatz.num_qubits
    zero_key = "0" * n_qubits
    return counts.get(zero_key, 0) / shots


def _circuit_metric_tensor(ansatz: QuantumCircuit, params: np.ndarray,
                           backend, shots: int, eps: float = 0.01) -> np.ndarray:
    """Compute the quantum metric tensor via overlap circuits on AerSimulator.

    A_ij = -d^2 <psi(theta_0)|psi(theta)> / d(theta_i) d(theta_j)

    Uses finite differences on the overlap function f(theta) = <psi_0|psi(theta)>.
    Since we measure |f|^2 from the circuit, and our ansatz produces real states
    (so f is real), we take f = sqrt(|f|^2) with positive sign (valid for small eps).
    """
    n_params = len(params)
    A = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            ei = np.zeros(n_params)
            ej = np.zeros(n_params)
            ei[i] = eps
            ej[j] = eps

            # Measure |<psi_0|psi(theta + shift)>|^2 for 4 shifts
            f_pp_sq = _circuit_overlap_sq(ansatz, params, params + ei + ej, backend, shots)
            f_pm_sq = _circuit_overlap_sq(ansatz, params, params + ei - ej, backend, shots)
            f_mp_sq = _circuit_overlap_sq(ansatz, params, params - ei + ej, backend, shots)
            f_mm_sq = _circuit_overlap_sq(ansatz, params, params - ei - ej, backend, shots)

            # Take sqrt (real states, small eps → overlap is positive)
            f_pp = np.sqrt(max(f_pp_sq, 0.0))
            f_pm = np.sqrt(max(f_pm_sq, 0.0))
            f_mp = np.sqrt(max(f_mp_sq, 0.0))
            f_mm = np.sqrt(max(f_mm_sq, 0.0))

            A[i, j] = -(f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
            A[j, i] = A[i, j]

    return A


# ---------------------------------------------------------------------------
# VQE (all shot-based)
# ---------------------------------------------------------------------------

def run_vqe_ideal(channel: str, cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run VQE on AerSimulator with shots (no gate noise)."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    rng = np.random.default_rng(cfg.seed)
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

    t0 = time.perf_counter()
    result = _sim_run_vqe_noisy(
        ham, ansatz, shots=cfg.vqe_shots, depolarizing_rate=0.0,
        x0=x0, method=cfg.vqe_method, maxiter=cfg.vqe_maxiter,
    )
    wall = time.perf_counter() - t0

    fid = _compute_fidelity(result.wavefunction, gs_state)
    return MethodResult(
        method="VQE", channel=channel, energy=result.energy,
        exact_energy=gs_energy, error=result.energy - gs_energy,
        convergence=result.convergence, wavefunction=result.wavefunction,
        fidelity=fid, wall_time=wall,
        metadata={"parameters": result.parameters.tolist(),
                  "num_evaluations": result.num_evaluations,
                  "shots": cfg.vqe_shots, "exact_eigenvalues": all_eigs},
    )


def run_vqe_noisy(channel: str, depolarizing_rate: float,
                   cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run VQE on AerSimulator with shots + depolarizing noise."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)

    rng = np.random.default_rng(cfg.seed)
    x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)

    t0 = time.perf_counter()
    result = _sim_run_vqe_noisy(
        ham, ansatz, shots=cfg.vqe_shots, depolarizing_rate=depolarizing_rate,
        x0=x0, method=cfg.vqe_method, maxiter=cfg.vqe_maxiter,
    )
    wall = time.perf_counter() - t0

    fid = _compute_fidelity(result.wavefunction, gs_state)
    return MethodResult(
        method="VQE", channel=channel, energy=result.energy,
        exact_energy=gs_energy, error=result.energy - gs_energy,
        convergence=result.convergence, wavefunction=result.wavefunction,
        fidelity=fid, wall_time=wall,
        metadata={"depolarizing_rate": depolarizing_rate,
                  "shots": cfg.vqe_shots, "parameters": result.parameters.tolist()},
    )


# ---------------------------------------------------------------------------
# VQITE (fully circuit-based)
# ---------------------------------------------------------------------------

def _run_vqite_circuit(
    channel: str,
    depolarizing_rate: float = 0.0,
    cfg: ExperimentConfig | None = None,
) -> MethodResult:
    """Run VQITE with all quantum operations on AerSimulator.

    - Energy: shot-based Pauli measurement
    - Gradient: parameter-shift rule with shot-based energy
    - Metric tensor: overlap circuits with finite differences
    """
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)
    backend = _make_backend(depolarizing_rate)

    n_params = ansatz.num_parameters
    theta = np.full(n_params, 0.5)
    shots = cfg.vqe_shots
    energy_history: list[float] = []

    t0 = time.perf_counter()
    for step in range(cfg.vqite_n_steps):
        e = _circuit_energy(ansatz, theta, ham, backend, shots)
        energy_history.append(e)

        # Gradient via parameter-shift (all circuit-based)
        C = np.zeros(n_params)
        for k in range(n_params):
            shift = np.zeros(n_params)
            shift[k] = np.pi / 2
            e_plus = _circuit_energy(ansatz, theta + shift, ham, backend, shots)
            e_minus = _circuit_energy(ansatz, theta - shift, ham, backend, shots)
            C[k] = -(e_plus - e_minus) / 2.0

        # Metric tensor via overlap circuits (all circuit-based)
        # Use more shots and larger epsilon for stability
        A = _circuit_metric_tensor(ansatz, theta, backend, shots * 4, eps=0.05)
        A_reg = A + cfg.vqite_regularization * np.eye(n_params)

        theta_dot = np.linalg.solve(A_reg, C)
        theta = theta + cfg.vqite_dtau * theta_dot

    final_energy = _circuit_energy(ansatz, theta, ham, backend, shots)
    energy_history.append(final_energy)
    wall = time.perf_counter() - t0

    # Final wavefunction for fidelity (statevector of the optimized circuit)
    final_sv = np.array(Statevector(ansatz.assign_parameters(theta)))
    fid = _compute_fidelity(final_sv, gs_state)

    return MethodResult(
        method="VQITE", channel=channel, energy=final_energy,
        exact_energy=gs_energy, error=final_energy - gs_energy,
        convergence=energy_history, wavefunction=final_sv,
        fidelity=fid, wall_time=wall,
        metadata={"depolarizing_rate": depolarizing_rate, "shots": shots,
                  "parameters": theta.tolist(), "exact_eigenvalues": all_eigs},
    )


def run_vqite_ideal(channel: str, cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run VQITE on AerSimulator with shots (no gate noise)."""
    return _run_vqite_circuit(channel, depolarizing_rate=0.0, cfg=cfg)


def run_vqite_noisy(channel: str, depolarizing_rate: float,
                     cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run VQITE on AerSimulator with shots + depolarizing noise."""
    return _run_vqite_circuit(channel, depolarizing_rate=depolarizing_rate, cfg=cfg)


# ---------------------------------------------------------------------------
# TTITE (LCU circuits on AerSimulator, shot-based energy measurement)
# ---------------------------------------------------------------------------

def _trace_out_ancilla(full_dm: np.ndarray, n_work: int) -> np.ndarray:
    """Trace out ancilla (qubit 0) from a (n_work+1)-qubit density matrix.

    Returns the reduced density matrix for the work system.
    """
    dim_work = 2 ** n_work
    rho_work = np.zeros((dim_work, dim_work), dtype=complex)
    # Sum over ancilla states (qubit 0 = LSB: even indices = |0>, odd = |1>)
    for a in range(2):
        for i in range(dim_work):
            for j in range(dim_work):
                rho_work[i, j] += full_dm[2 * i + a, 2 * j + a]
    return rho_work


def _postselect_ancilla_0_sv(full_sv: np.ndarray, n_work: int) -> tuple[np.ndarray, float]:
    """Postselect ancilla=|0> from a (n_work+1)-qubit statevector.

    Qiskit ordering: ancilla is qubit 0 (LSB). Even-indexed amplitudes
    correspond to ancilla=|0>.
    """
    postselected = full_sv[0::2]  # ancilla=0 → even indices
    norm_sq = float(np.real(np.vdot(postselected, postselected)))
    if norm_sq > 1e-30:
        postselected = postselected / np.sqrt(norm_sq)
    return postselected, norm_sq


def _postselect_ancilla_0_dm(full_dm: np.ndarray, n_work: int) -> tuple[np.ndarray, float]:
    """Postselect ancilla=|0> from a (n_work+1)-qubit density matrix.

    Projects onto |0><0| on ancilla, then traces out the ancilla qubit.
    """
    dim_total = full_dm.shape[0]
    dim_work = 2 ** n_work

    # |0><0| on ancilla = project onto even-indexed rows/cols
    rho_work = np.zeros((dim_work, dim_work), dtype=complex)
    for i in range(dim_work):
        for j in range(dim_work):
            rho_work[i, j] = full_dm[2 * i, 2 * j]

    tr = float(np.real(np.trace(rho_work)))
    if tr > 1e-30:
        rho_work = rho_work / tr
    return rho_work, tr


def _run_ttite_circuit(
    channel: str,
    depolarizing_rate: float = 0.0,
    cfg: ExperimentConfig | None = None,
) -> MethodResult:
    """Run TTITE with actual LCU circuits on AerSimulator.

    Noiseless: per-step statevector simulation with proper postselection.
    Noisy: single continuous circuit with ancilla resets between steps,
           run on density_matrix backend with noise model. Noise accumulates
           across all steps (no state reset between steps). Failed
           postselections (ancilla=|1>) contaminate the work state, modeling
           realistic hardware behavior.

    Final energy: measured via shot-based Pauli circuits on AerSimulator.
    """
    from qiskit_aer import AerSimulator

    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)
    H_matrix = np.array(ham.to_matrix())
    terms = decompose_to_pauli_terms(ham)

    n_work = 2
    n_total = n_work + 1  # ancilla + work
    L = int(round(cfg.ttite_tau_total / cfg.ttite_dt))
    use_dm = depolarizing_rate > 0

    # Initial state: equal superposition
    initial_state = np.ones(4, dtype=complex) / 2.0

    t0 = time.perf_counter()

    if use_dm:
        # --- NOISY: single continuous circuit with ancilla resets ---
        # Build one circuit that chains ALL Trotter steps. Noise accumulates
        # naturally since it's a single circuit on a noisy backend.
        # reset(ancilla) between steps: measures ancilla and flips to |0> if
        # needed. This is NOT postselection — failed branches contaminate
        # the work state, as on real hardware.
        noise_model = make_noise_model(depolarizing_rate)
        evo_backend = AerSimulator(method="density_matrix", noise_model=noise_model)

        full_circ = QuantumCircuit(n_total)
        full_circ.initialize(initial_state, list(range(1, n_total)))

        # Filter non-identity terms (identity is a scalar, skip in circuit)
        non_id_terms = [(c, l) for c, l, _ in terms if not all(p == "I" for p in l)]

        for seg in range(L):
            for c_i, label in non_id_terms:
                lcu_gates = build_trotter_step_circuit(
                    n_work, label[::-1], c_i, cfg.ttite_dt, cfg.ttite_order
                )
                full_circ.compose(lcu_gates, inplace=True)
                full_circ.reset(0)  # reset ancilla → |0> (not postselection)

            # Save intermediate density matrix for convergence tracking
            full_circ.save_density_matrix(label=f"seg_{seg}")

        # Transpile once to decompose 'initialize' for density_matrix backend
        full_circ = transpile(full_circ, evo_backend, optimization_level=0)

        # Run the entire evolution as a single circuit
        result = evo_backend.run(full_circ).result()

        # Extract convergence history from intermediate saves
        energy_history = []
        fidelity_history = []
        # Initial state tracking
        rho_init = np.outer(initial_state, initial_state.conj())
        energy_history.append(float(np.real(np.trace(H_matrix @ rho_init))))
        fidelity_history.append(float(np.real(gs_state.conj() @ rho_init @ gs_state)))

        for seg in range(L):
            rho_seg = np.array(result.data()[f"seg_{seg}"])
            # Trace out ancilla (qubit 0) to get work-system density matrix
            # Ancilla was just reset to |0>, so trace over it
            rho_work = _trace_out_ancilla(rho_seg, n_work)
            tr = np.real(np.trace(rho_work))
            if tr > 1e-30:
                rho_work = rho_work / tr
            energy_history.append(float(np.real(np.trace(H_matrix @ rho_work))))
            fidelity_history.append(float(np.real(gs_state.conj() @ rho_work @ gs_state)))

        # Final state = last segment's density matrix
        rho_final = rho_work
        evals, evecs = np.linalg.eigh(rho_final)
        final_state_vec = evecs[:, -1]

    else:
        # --- NOISELESS: per-step statevector with proper postselection ---
        evo_backend = AerSimulator(method="statevector")
        state = initial_state.copy()

        energy_history = [float(np.real(np.conj(state) @ H_matrix @ state))]
        fidelity_history = [float(abs(np.vdot(gs_state, state)) ** 2)]

        for seg in range(L):
            for c_i, label, mat in terms:
                if all(p == "I" for p in label):
                    factor = np.exp(-c_i * cfg.ttite_dt)
                    state = state * factor
                    norm = np.linalg.norm(state)
                    if norm > 1e-30:
                        state = state / norm
                    continue

                lcu_gates = build_trotter_step_circuit(
                    n_work, label[::-1], c_i, cfg.ttite_dt, cfg.ttite_order
                )
                circ = QuantumCircuit(n_total)
                circ.initialize(state, list(range(1, n_total)))
                circ.compose(lcu_gates, inplace=True)
                circ.save_statevector()

                res = evo_backend.run(circ).result()
                full_sv = np.array(res.data()["statevector"])
                state, prob = _postselect_ancilla_0_sv(full_sv, n_work)

            energy_history.append(float(np.real(np.conj(state) @ H_matrix @ state)))
            fidelity_history.append(float(abs(np.vdot(gs_state, state)) ** 2))

        final_state_vec = state

    wall_evo = time.perf_counter() - t0

    # Final energy measurement
    if use_dm:
        # Noisy: use Tr(ρH) from the density matrix (the state is mixed,
        # so extracting a pure state eigenvector is meaningless)
        final_energy = energy_history[-1]
    else:
        # Noiseless: shot-based Pauli measurement on the pure state
        energy_backend = _make_backend(0.0)
        prep_circ = QuantumCircuit(n_work)
        prep_circ.initialize(final_state_vec, range(n_work))
        final_energy = _estimate_energy_shot_based(prep_circ, ham, energy_backend, cfg.vqe_shots)

    wall = time.perf_counter() - t0

    return MethodResult(
        method="TTITE", channel=channel, energy=final_energy,
        exact_energy=gs_energy, error=final_energy - gs_energy,
        convergence=energy_history, wavefunction=final_state_vec,
        fidelity=fidelity_history[-1], wall_time=wall,
        metadata={"depolarizing_rate": depolarizing_rate,
                  "tau_values": [i * cfg.ttite_dt for i in range(L + 1)],
                  "fidelity_history": fidelity_history,
                  "exact_eigenvalues": all_eigs},
    )


def run_ttite_ideal(channel: str, cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run TTITE with LCU circuits on AerSimulator (no noise)."""
    return _run_ttite_circuit(channel, depolarizing_rate=0.0, cfg=cfg)


def run_ttite_noisy(channel: str, depolarizing_rate: float,
                     cfg: ExperimentConfig | None = None) -> MethodResult:
    """Run TTITE with LCU circuits on noisy AerSimulator."""
    return _run_ttite_circuit(channel, depolarizing_rate=depolarizing_rate, cfg=cfg)


# ---------------------------------------------------------------------------
# Excited states (VQE + VQITE only, circuit-based)
# ---------------------------------------------------------------------------

def _circuit_overlap_with_state(ansatz: QuantumCircuit, params: np.ndarray,
                                target_state: np.ndarray, backend, shots: int) -> float:
    """Measure |<target|psi(params)>|^2 via the swap test circuit.

    Uses an ancilla qubit + controlled-SWAPs:
        P(ancilla=0) = (1 + |<target|psi>|^2) / 2
    """
    n_qubits = ansatz.num_qubits
    # 1 ancilla + 2 registers of n_qubits each
    n_total = 1 + 2 * n_qubits
    circ = QuantumCircuit(n_total, 1)

    ancilla = 0
    reg_a = list(range(1, 1 + n_qubits))
    reg_b = list(range(1 + n_qubits, n_total))

    # Prepare |psi(params)> on register A
    bound = ansatz.assign_parameters(params)
    circ.compose(bound, reg_a, inplace=True)

    # Prepare |target> on register B
    target_real = np.real(target_state).astype(float)
    norm = np.linalg.norm(target_real)
    if norm > 1e-10:
        target_real = target_real / norm
    target_circ = QuantumCircuit(n_qubits)
    target_circ.initialize(target_real, range(n_qubits))
    circ.compose(target_circ, reg_b, inplace=True)

    # Swap test
    circ.h(ancilla)
    for i in range(n_qubits):
        circ.cswap(ancilla, reg_a[i], reg_b[i])
    circ.h(ancilla)
    circ.measure(ancilla, 0)

    transpiled = transpile(circ, backend, optimization_level=0)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # P(ancilla=0) = (1 + |overlap|^2) / 2
    p0 = sum(c for bs, c in counts.items() if bs[-1] == "0") / shots
    overlap_sq = max(2 * p0 - 1, 0.0)
    return overlap_sq


def run_vqe_excited(
    channel: str, n_states: int = 4, cfg: ExperimentConfig | None = None,
) -> list[MethodResult]:
    """Run VQE for ground + excited states, all circuit-based."""
    from scipy.optimize import minimize
    from quarksim.simulation import VQEResult

    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)
    backend = _make_backend(0.0)
    shots = cfg.vqe_shots

    rng = np.random.default_rng(cfg.seed)
    results = []
    found_states: list[np.ndarray] = []

    for k in range(min(n_states, len(all_eigs))):
        x0 = rng.uniform(0, np.pi, size=ansatz.num_parameters)
        energies: list[float] = []

        def cost_fn(params, _found=list(found_states)):
            bound = ansatz.assign_parameters(params)
            e = _estimate_energy_shot_based(bound, ham, backend, shots)
            for phi in _found:
                overlap_sq = _circuit_overlap_with_state(ansatz, params, phi, backend, shots)
                e += cfg.penalty_alpha * overlap_sq
            energies.append(e)
            return e

        t0 = time.perf_counter()
        opt_result = minimize(cost_fn, x0, method=cfg.vqe_method,
                              options={"maxiter": cfg.vqe_maxiter})
        wall = time.perf_counter() - t0

        # Get wavefunction and actual energy (without penalty)
        sv = np.array(Statevector(ansatz.assign_parameters(opt_result.x)))
        actual_energy = _estimate_energy_shot_based(
            ansatz.assign_parameters(opt_result.x), ham, backend, shots
        )

        found_states.append(sv)
        fid = _compute_fidelity(sv, gs_state) if k == 0 else 0.0

        results.append(MethodResult(
            method="VQE", channel=channel, energy=actual_energy,
            exact_energy=all_eigs[k], error=actual_energy - all_eigs[k],
            convergence=energies, wavefunction=sv, fidelity=fid, wall_time=wall,
            metadata={"state_index": k, "parameters": opt_result.x.tolist()},
        ))

    return results


def run_vqite_all_excited(
    channel: str, n_states: int = 4, cfg: ExperimentConfig | None = None,
) -> list[MethodResult]:
    """Run VQITE for ground + excited states, all circuit-based."""
    cfg = cfg or ExperimentConfig()
    ham = build_pauli_hamiltonian(channel)
    ansatz = build_ansatz()
    gs_energy, gs_state, all_eigs = _exact_ground_state(ham)
    backend = _make_backend(0.0)
    shots = cfg.vqe_shots
    n_params = ansatz.num_parameters

    results = []
    found_states: list[np.ndarray] = []

    for k in range(min(n_states, len(all_eigs))):
        theta = np.full(n_params, 0.5)
        energy_history: list[float] = []

        t0 = time.perf_counter()
        for step in range(cfg.vqite_n_steps):
            # Effective energy with penalty
            e = _circuit_energy(ansatz, theta, ham, backend, shots)
            for phi in found_states:
                e += cfg.penalty_alpha * _circuit_overlap_with_state(
                    ansatz, theta, phi, backend, shots
                )
            energy_history.append(e)

            # Gradient via parameter-shift (with penalty)
            C = np.zeros(n_params)
            for p_idx in range(n_params):
                shift = np.zeros(n_params)
                shift[p_idx] = np.pi / 2
                e_plus = _circuit_energy(ansatz, theta + shift, ham, backend, shots)
                e_minus = _circuit_energy(ansatz, theta - shift, ham, backend, shots)
                for phi in found_states:
                    e_plus += cfg.penalty_alpha * _circuit_overlap_with_state(
                        ansatz, theta + shift, phi, backend, shots
                    )
                    e_minus += cfg.penalty_alpha * _circuit_overlap_with_state(
                        ansatz, theta - shift, phi, backend, shots
                    )
                C[p_idx] = -(e_plus - e_minus) / 2.0

            A = _circuit_metric_tensor(ansatz, theta, backend, shots)
            A_reg = A + cfg.vqite_regularization * np.eye(n_params)
            theta_dot = np.linalg.solve(A_reg, C)
            theta = theta + cfg.vqite_dtau * theta_dot

        wall = time.perf_counter() - t0

        sv = np.array(Statevector(ansatz.assign_parameters(theta)))
        actual_energy = _circuit_energy(ansatz, theta, ham, backend, shots)
        found_states.append(sv)
        fid = _compute_fidelity(sv, gs_state) if k == 0 else 0.0

        results.append(MethodResult(
            method="VQITE", channel=channel, energy=actual_energy,
            exact_energy=all_eigs[k], error=actual_energy - all_eigs[k],
            convergence=energy_history, wavefunction=sv, fidelity=fid,
            wall_time=wall,
            metadata={"state_index": k, "parameters": theta.tolist()},
        ))

    return results
