"""Variational Quantum Imaginary Time Evolution (VQITE).

Implements the VQITE algorithm from Woloshyn (arXiv:2301.10828v2), Section 3.

Imaginary time evolution e^{-H*tau}|psi_0> converges to the ground state for
large tau. Since this is non-unitary, we use a variational approach: at each
time step, find a unitary update U(Delta_tau) such that

    |psi(tau + dtau)> = U(dtau)|psi(tau)> ≈ e^{-H*dtau}|psi(tau)> / norm

The variational parameters theta evolve as:
    A @ theta_dot = C           (Eq. 9)
    theta(tau + dtau) = theta(tau) + dtau * theta_dot

where:
    C_i = -dE/d(theta_i)       (Eq. 10, energy gradient)
    A_ij = -d^2 <psi|psi(theta)> / d(theta_i) d(theta_j)   (Eq. 11, metric tensor)
"""

from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass
class VQITEResult:
    """Result of a VQITE simulation."""

    energy: float
    parameters: np.ndarray
    wavefunction: np.ndarray
    energy_history: list[float] = field(default_factory=list)
    param_history: list[list[float]] = field(default_factory=list)
    n_steps: int = 0


def _get_statevector(ansatz: QuantumCircuit, params: np.ndarray) -> np.ndarray:
    """Get the statevector for given parameters."""
    bound = ansatz.assign_parameters(params)
    return np.array(Statevector(bound))


def _get_energy(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
) -> float:
    """Compute <psi(params)|H|psi(params)>."""
    sv = Statevector(_get_statevector(ansatz, params))
    return sv.expectation_value(hamiltonian).real


def compute_energy_gradient(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
) -> np.ndarray:
    """Compute C_i = -dE/d(theta_i) using the parameter-shift rule.

    For Ry gates, the exact gradient is:
        dE/d(theta_k) = [E(theta + pi/2 * e_k) - E(theta - pi/2 * e_k)] / 2

    Returns:
        C vector of shape (n_params,). Note the sign: C = -grad(E).
    """
    n_params = len(params)
    C = np.zeros(n_params)

    for k in range(n_params):
        shift = np.zeros(n_params)
        shift[k] = np.pi / 2

        e_plus = _get_energy(ansatz, hamiltonian, params + shift)
        e_minus = _get_energy(ansatz, hamiltonian, params - shift)

        C[k] = -(e_plus - e_minus) / 2.0

    return C


def compute_metric_tensor(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute the quantum metric tensor A_ij (Eq. 11).

    A_ij = -d^2 <psi(theta_fixed)|psi(theta)> / d(theta_i) d(theta_j)

    where derivatives act only on the ket (right side).

    We compute this via finite differences on the overlap function
    f(theta) = <psi_0|psi(theta)> where psi_0 = psi(theta_current).

    A_ij = -[f(+ei+ej) - f(+ei-ej) - f(-ei+ej) + f(-ei-ej)] / (4*eps^2)

    Returns:
        A matrix of shape (n_params, n_params), real and symmetric.
    """
    n_params = len(params)
    psi_0 = _get_statevector(ansatz, params)

    def overlap(theta):
        """<psi_0|psi(theta)>"""
        psi = _get_statevector(ansatz, theta)
        return np.vdot(psi_0, psi).real

    A = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            ei = np.zeros(n_params)
            ej = np.zeros(n_params)
            ei[i] = eps
            ej[j] = eps

            f_pp = overlap(params + ei + ej)
            f_pm = overlap(params + ei - ej)
            f_mp = overlap(params - ei + ej)
            f_mm = overlap(params - ei - ej)

            A[i, j] = -(f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)
            A[j, i] = A[i, j]

    return A


def run_vqite(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    n_steps: int = 50,
    dtau: float = 0.02,
    theta0: np.ndarray | None = None,
    regularization: float = 1e-4,
) -> VQITEResult:
    """Run VQITE to find the ground state.

    At each step:
        1. Compute metric tensor A and energy gradient C
        2. Solve A @ theta_dot = C (with regularization)
        3. Update theta += dtau * theta_dot

    Args:
        hamiltonian: Pauli Hamiltonian (SparsePauliOp).
        ansatz: Parameterized quantum circuit.
        n_steps: Number of imaginary time steps.
        dtau: Step size (learning rate). Paper uses 0.02.
        theta0: Initial parameters. Default: all 0.5 (as in paper).
        regularization: Tikhonov regularization lambda for A + lambda*I.

    Returns:
        VQITEResult with final energy, parameters, and convergence history.
    """
    n_params = ansatz.num_parameters
    if theta0 is None:
        theta0 = np.full(n_params, 0.5)
    theta = theta0.copy()

    energy_history = []
    param_history = []

    for step in range(n_steps):
        energy = _get_energy(ansatz, hamiltonian, theta)
        energy_history.append(energy)
        param_history.append(theta.tolist())

        # Compute gradient and metric
        C = compute_energy_gradient(ansatz, hamiltonian, theta)
        A = compute_metric_tensor(ansatz, theta)

        # Regularize A for numerical stability
        A_reg = A + regularization * np.eye(n_params)

        # Solve for theta_dot
        theta_dot = np.linalg.solve(A_reg, C)

        # Update parameters
        theta = theta + dtau * theta_dot

    # Final energy
    final_energy = _get_energy(ansatz, hamiltonian, theta)
    energy_history.append(final_energy)
    param_history.append(theta.tolist())

    final_sv = _get_statevector(ansatz, theta)

    return VQITEResult(
        energy=final_energy,
        parameters=theta,
        wavefunction=final_sv,
        energy_history=energy_history,
        param_history=param_history,
        n_steps=n_steps,
    )


def run_vqite_excited(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    lower_states: list[np.ndarray],
    alpha: float = 10.0,
    n_steps: int = 50,
    dtau: float = 0.02,
    theta0: np.ndarray | None = None,
    regularization: float = 1e-4,
) -> VQITEResult:
    """Run VQITE for an excited state using the penalty method (Eq. 12).

    The effective energy is:
        E_eff = <H> + alpha * sum_k |<phi_k|psi(theta)>|^2

    where |phi_k> are the already-determined lower-lying states.
    The penalty term pushes the variational state away from them.

    Args:
        hamiltonian: Pauli Hamiltonian.
        ansatz: Parameterized ansatz circuit.
        lower_states: List of statevectors for lower-lying states.
        alpha: Penalty strength. Must be large enough to enforce orthogonality.
        n_steps: Number of VQITE steps.
        dtau: Step size.
        theta0: Initial parameters. Default: all 0.5.
        regularization: Tikhonov regularization for A.

    Returns:
        VQITEResult for the excited state.
    """
    n_params = ansatz.num_parameters
    if theta0 is None:
        theta0 = np.full(n_params, 0.5)
    theta = theta0.copy()

    energy_history = []
    param_history = []

    def effective_energy(params):
        sv = _get_statevector(ansatz, params)
        sv_qiskit = Statevector(sv)
        e = sv_qiskit.expectation_value(hamiltonian).real
        for phi in lower_states:
            overlap_sq = abs(np.vdot(phi, sv)) ** 2
            e += alpha * overlap_sq
        return e

    def effective_gradient(params):
        """C_i = -dE_eff/d(theta_i) via parameter-shift rule."""
        n = len(params)
        C = np.zeros(n)
        for k in range(n):
            shift = np.zeros(n)
            shift[k] = np.pi / 2
            e_plus = effective_energy(params + shift)
            e_minus = effective_energy(params - shift)
            C[k] = -(e_plus - e_minus) / 2.0
        return C

    for step in range(n_steps):
        e_eff = effective_energy(theta)
        energy_history.append(e_eff)
        param_history.append(theta.tolist())

        C = effective_gradient(theta)
        A = compute_metric_tensor(ansatz, theta)
        A_reg = A + regularization * np.eye(n_params)

        theta_dot = np.linalg.solve(A_reg, C)
        theta = theta + dtau * theta_dot

    # Final state — report the actual Hamiltonian energy (without penalty)
    final_sv = _get_statevector(ansatz, theta)
    final_energy = Statevector(final_sv).expectation_value(hamiltonian).real
    energy_history.append(effective_energy(theta))
    param_history.append(theta.tolist())

    return VQITEResult(
        energy=final_energy,
        parameters=theta,
        wavefunction=final_sv,
        energy_history=energy_history,
        param_history=param_history,
        n_steps=n_steps,
    )
