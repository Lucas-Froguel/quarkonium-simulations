"""Core TTITE imaginary-time evolution engine.

Implements the Trotter-Taylor Imaginary-Time Evolution algorithm using
direct matrix operations on statevectors. This is how the paper generates
its numerical results (Figures 2-5).

The algorithm:
1. Decompose H = sum_i c_i h_i into Pauli terms
2. For each Trotter segment (L = tau/dt segments total):
   - For each term i: apply T_i(dt) = alpha_i*I + beta_i*h_i to the state
   - Renormalize after each step
3. Track energy E_tau = <psi|H|psi> and fidelity F = |<psi|gs>|^2
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp

from quarksim.simulation import exact_diagonalization
from quarksim.yihuoliufanzhang.circuit import taylor_coefficients


@dataclass
class TTITEResult:
    """Result of a TTITE imaginary-time evolution."""

    final_energy: float
    final_fidelity: float
    final_state: np.ndarray
    energy_history: list[float] = field(default_factory=list)
    fidelity_history: list[float] = field(default_factory=list)
    tau_values: list[float] = field(default_factory=list)
    success_probabilities: list[float] = field(default_factory=list)
    ground_state_energy: float = 0.0
    metadata: dict = field(default_factory=dict)


def _compute_energy(state: np.ndarray, H_matrix: np.ndarray) -> float:
    """Compute <psi|H|psi>."""
    return np.real(state.conj() @ H_matrix @ state)


def _compute_fidelity(state: np.ndarray, ground_state: np.ndarray) -> float:
    """Compute |<psi|gs>|^2."""
    return float(abs(state.conj() @ ground_state) ** 2)


def apply_trotter_step_matrix(
    state: np.ndarray,
    pauli_matrix: np.ndarray,
    c_i: float,
    dt: float,
    order: int,
) -> tuple[np.ndarray, float]:
    """Apply one Trotter step T_i(dt) = alpha*I + beta*h_i to the state.

    Args:
        state: Current statevector (normalized).
        pauli_matrix: Matrix representation of h_i.
        c_i: Coefficient of this term in H.
        dt: Trotter step duration.
        order: Taylor expansion order R.

    Returns:
        (new_state, success_probability)
        new_state is normalized.
        success_probability = ||T_i|psi>||^2 / (2*(alpha^2 + beta^2))  (Eq. 11)
    """
    alpha, beta = taylor_coefficients(c_i, dt, order)
    dim = len(state)

    # Apply T_i = alpha*I + beta*h_i
    new_state = alpha * state + beta * (pauli_matrix @ state)

    # Success probability (Eq. 11)
    norm_sq = np.real(new_state.conj() @ new_state)
    denom = 2.0 * (alpha**2 + beta**2)
    success_prob = norm_sq / denom if denom > 1e-30 else 0.0

    # Normalize
    norm = np.sqrt(norm_sq)
    if norm > 1e-30:
        new_state = new_state / norm

    return new_state, float(success_prob)


def apply_trotter_segment_matrix(
    state: np.ndarray,
    terms: list[tuple[float, str, np.ndarray]],
    dt: float,
    order: int,
) -> tuple[np.ndarray, float]:
    """Apply a full Trotter segment S(dt) = T_m(dt) ... T_1(dt).

    Identity terms contribute a scalar factor e^{-c_0 * dt} which is
    absorbed into the normalization.

    Args:
        state: Current statevector (normalized).
        terms: List of (coefficient, label, matrix) from decompose_to_pauli_terms.
        dt: Trotter step duration.
        order: Taylor expansion order R.

    Returns:
        (new_state, cumulative_success_probability)
    """
    cumulative_prob = 1.0

    for c_i, label, mat in terms:
        is_identity = all(p == "I" for p in label)

        if is_identity:
            # Identity: e^{-c0 * dt} * I = scalar factor, absorbed in normalization
            # In statevector simulation, just multiply and renormalize
            factor = np.exp(-c_i * dt)
            state = state * factor
            norm = np.linalg.norm(state)
            if norm > 1e-30:
                state = state / norm
        else:
            state, prob = apply_trotter_step_matrix(state, mat, c_i, dt, order)
            cumulative_prob *= prob

    return state, cumulative_prob


def run_ttite(
    hamiltonian: SparsePauliOp,
    initial_state: np.ndarray,
    tau_total: float,
    dt: float,
    order: int,
) -> TTITEResult:
    """Run the full TTITE imaginary-time evolution.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian.
        initial_state: Initial statevector (normalized).
        tau_total: Total imaginary time.
        dt: Trotter segment duration (Delta tau).
        order: Taylor expansion order R.

    Returns:
        TTITEResult with energy/fidelity histories.
    """
    from quarksim.yihuoliufanzhang.hamiltonian import decompose_to_pauli_terms

    # Exact diagonalization for reference
    eigenvalues, eigenvectors = exact_diagonalization(hamiltonian)
    gs_energy = eigenvalues[0]
    gs_state = eigenvectors[:, 0]

    # Decompose Hamiltonian
    H_matrix = hamiltonian.to_matrix()
    terms = decompose_to_pauli_terms(hamiltonian)

    # Number of Trotter segments
    L = int(round(tau_total / dt))

    state = initial_state.copy().astype(complex)
    norm = np.linalg.norm(state)
    if norm > 1e-30:
        state = state / norm

    # Track at tau=0
    energy_history = [float(_compute_energy(state, H_matrix))]
    fidelity_history = [float(_compute_fidelity(state, gs_state))]
    tau_values = [0.0]
    success_probs = []

    for step in range(1, L + 1):
        state, prob = apply_trotter_segment_matrix(state, terms, dt, order)
        success_probs.append(prob)

        tau = step * dt
        energy = _compute_energy(state, H_matrix)
        fidelity = _compute_fidelity(state, gs_state)

        energy_history.append(float(energy))
        fidelity_history.append(float(fidelity))
        tau_values.append(float(tau))

    return TTITEResult(
        final_energy=energy_history[-1],
        final_fidelity=fidelity_history[-1],
        final_state=state,
        energy_history=energy_history,
        fidelity_history=fidelity_history,
        tau_values=tau_values,
        success_probabilities=success_probs,
        ground_state_energy=float(gs_energy),
        metadata={
            "tau_total": tau_total,
            "dt": dt,
            "order": order,
            "n_segments": L,
            "n_qubits": hamiltonian.num_qubits,
        },
    )


def run_ite_exact(
    hamiltonian: SparsePauliOp,
    initial_state: np.ndarray,
    tau_total: float,
    n_points: int = 100,
) -> TTITEResult:
    """Compute exact imaginary-time evolution for comparison.

    |psi(tau)> = e^{-H*tau} |psi(0)> / ||e^{-H*tau} |psi(0)>||

    This is the "E-ITE theory" curve in the paper's figures.
    """
    eigenvalues, eigenvectors = exact_diagonalization(hamiltonian)
    gs_energy = eigenvalues[0]
    gs_state = eigenvectors[:, 0]
    H_matrix = hamiltonian.to_matrix()

    state0 = initial_state.copy().astype(complex)
    norm = np.linalg.norm(state0)
    if norm > 1e-30:
        state0 = state0 / norm

    tau_values = np.linspace(0, tau_total, n_points + 1)
    energy_history = []
    fidelity_history = []

    for tau in tau_values:
        # e^{-H*tau} |psi0>
        propagator = expm(-np.array(H_matrix) * tau)
        state = propagator @ state0
        norm = np.linalg.norm(state)
        if norm > 1e-30:
            state = state / norm

        energy_history.append(float(_compute_energy(state, H_matrix)))
        fidelity_history.append(float(_compute_fidelity(state, gs_state)))

    return TTITEResult(
        final_energy=energy_history[-1],
        final_fidelity=fidelity_history[-1],
        final_state=state,
        energy_history=energy_history,
        fidelity_history=fidelity_history,
        tau_values=tau_values.tolist(),
        ground_state_energy=float(gs_energy),
        metadata={"method": "exact_ite", "n_points": n_points},
    )


def run_trotter_ite(
    hamiltonian: SparsePauliOp,
    initial_state: np.ndarray,
    tau_total: float,
    dt: float,
) -> TTITEResult:
    """Compute Trotter-decomposed ITE without Taylor truncation.

    Each step applies exact e^{-c_i h_i dt} per term (no Taylor error).
    This is the "E-Trotter ITE" curve that isolates Trotter error.
    """
    from quarksim.yihuoliufanzhang.hamiltonian import decompose_to_pauli_terms

    eigenvalues, eigenvectors = exact_diagonalization(hamiltonian)
    gs_energy = eigenvalues[0]
    gs_state = eigenvectors[:, 0]
    H_matrix = hamiltonian.to_matrix()
    terms = decompose_to_pauli_terms(hamiltonian)

    L = int(round(tau_total / dt))
    state = initial_state.copy().astype(complex)
    norm = np.linalg.norm(state)
    if norm > 1e-30:
        state = state / norm

    energy_history = [float(_compute_energy(state, H_matrix))]
    fidelity_history = [float(_compute_fidelity(state, gs_state))]
    tau_values = [0.0]

    for step in range(1, L + 1):
        for c_i, label, mat in terms:
            # Exact e^{-c_i h_i dt}
            propagator = expm(-c_i * mat * dt)
            state = propagator @ state
            norm_val = np.linalg.norm(state)
            if norm_val > 1e-30:
                state = state / norm_val

        tau = step * dt
        energy_history.append(float(_compute_energy(state, H_matrix)))
        fidelity_history.append(float(_compute_fidelity(state, gs_state)))
        tau_values.append(float(tau))

    return TTITEResult(
        final_energy=energy_history[-1],
        final_fidelity=fidelity_history[-1],
        final_state=state,
        energy_history=energy_history,
        fidelity_history=fidelity_history,
        tau_values=tau_values,
        ground_state_energy=float(gs_energy),
        metadata={"method": "trotter_ite", "dt": dt, "n_segments": L},
    )
