"""LCU quantum circuits for the TTITE algorithm.

Implements Algorithm 1 from the paper: for each Trotter step T_i(dt),
the non-unitary operator e^{-c_i h_i dt} is approximated by a Taylor
expansion and implemented via a linear combination of unitaries (LCU)
using one ancilla qubit.

Key insight: since h_i is a Pauli product, h_i^2 = I, so
    e^{-c_i h_i dt} = alpha * I + beta * h_i
where alpha sums even-order Taylor terms and beta sums odd-order terms.
"""

import math

import numpy as np
from qiskit.circuit import QuantumCircuit


def taylor_coefficients(c_i: float, dt: float, order: int) -> tuple[float, float]:
    """Compute Taylor expansion coefficients (Eqs. 4-5).

    For a Pauli product h_i with h_i^2 = I:
        T_i^R(dt) = alpha * I + beta * h_i

    where:
        alpha = sum_{j=0, j even}^{R} (-c_i * dt)^j / j!
        beta  = sum_{k=0, k odd}^{R}  (-c_i * dt)^k / k!

    Args:
        c_i: Coefficient of the Pauli term in the Hamiltonian.
        dt: Trotter step duration (Delta tau).
        order: Taylor expansion order R.

    Returns:
        (alpha, beta) coefficients.
    """
    x = -c_i * dt
    alpha = 0.0
    beta = 0.0
    for j in range(order + 1):
        term = x**j / math.factorial(j)
        if j % 2 == 0:
            alpha += term
        else:
            beta += term
    return alpha, beta


def build_trotter_step_circuit(
    n_work_qubits: int,
    pauli_label: str,
    c_i: float,
    dt: float,
    order: int,
) -> QuantumCircuit:
    """Build the LCU circuit for one Trotter step (Algorithm 1).

    Circuit layout: qubit 0 = ancilla, qubits 1..n = work system.

    Step 1: Ry(theta) on ancilla, where theta = 2*arccos(alpha / sqrt(alpha^2 + beta^2))
    Step 2: Controlled-h_i (ancilla controls each Pauli gate on work qubits)
    Step 3: Hadamard on ancilla
    Step 4: Measure ancilla (handled by caller)

    On postselecting ancilla = |0>, the work system is in state
    (alpha*I + beta*h_i)|psi> / ||(alpha*I + beta*h_i)|psi>||.

    Args:
        n_work_qubits: Number of qubits in the work system.
        pauli_label: Pauli string label for h_i (Qiskit convention: 'XYZ' = X on q2, Y on q1, Z on q0).
        c_i: Coefficient of this Pauli term.
        dt: Trotter step duration.
        order: Taylor expansion order R.

    Returns:
        QuantumCircuit with n_work_qubits + 1 qubits (ancilla at index 0).
    """
    alpha, beta = taylor_coefficients(c_i, dt, order)
    norm = math.sqrt(alpha**2 + beta**2)

    n_total = n_work_qubits + 1
    qc = QuantumCircuit(n_total)

    # Step 1: Encode alpha, beta on ancilla via Ry
    if norm > 1e-15:
        theta = 2 * math.acos(np.clip(alpha / norm, -1, 1))
        qc.ry(theta, 0)

    # Step 2: Controlled-h_i
    # pauli_label[j] corresponds to qubit (n_work - 1 - j) in Qiskit convention
    for j, p in enumerate(pauli_label):
        work_qubit = 1 + j  # map label position j to circuit qubit 1+j
        if p == "X":
            qc.cx(0, work_qubit)
        elif p == "Y":
            qc.cy(0, work_qubit)
        elif p == "Z":
            qc.cz(0, work_qubit)
        # 'I' -> no gate

    # Step 3: Hadamard on ancilla
    qc.h(0)

    return qc


def build_improved_trotter_step_circuit(
    n_work_qubits: int,
    pauli_label: str,
    c_i: float,
    dt: float,
    order: int,
) -> QuantumCircuit:
    """Build the improved LCU circuit (Section 5, Fig. 7).

    Replaces Ry-CU-H with Ry(theta')-CU-Ry(theta')^dagger for higher
    success probability.

    theta' = 2*arccos(sqrt(alpha / (alpha + beta)))   (Eq. 18)

    Success probability: P'_s = ||T_i(dt)|psi>||^2 / (alpha + beta)^2 >= P_s   (Eq. 20)
    """
    alpha, beta = taylor_coefficients(c_i, dt, order)

    n_total = n_work_qubits + 1
    qc = QuantumCircuit(n_total)

    # Step 1: Ry(theta') on ancilla
    ab_sum = alpha + beta
    if abs(ab_sum) > 1e-15:
        ratio = np.clip(alpha / ab_sum, 0, 1)
        theta_prime = 2 * math.acos(math.sqrt(ratio))
        qc.ry(theta_prime, 0)

    # Step 2: Controlled-h_i (same as standard)
    for j, p in enumerate(pauli_label):
        work_qubit = 1 + j
        if p == "X":
            qc.cx(0, work_qubit)
        elif p == "Y":
            qc.cy(0, work_qubit)
        elif p == "Z":
            qc.cz(0, work_qubit)

    # Step 3: Ry(theta')^dagger on ancilla (instead of Hadamard)
    if abs(ab_sum) > 1e-15:
        ratio = np.clip(alpha / ab_sum, 0, 1)
        theta_prime = 2 * math.acos(math.sqrt(ratio))
        qc.ry(-theta_prime, 0)

    return qc
