"""Variational ansatz circuit for the 2-qubit quarkonium system.

Implements the ansatz from Woloshyn (arXiv:2301.10828v2), Fig. 2.

The circuit uses 3 parameters (theta_0, theta_1, theta_2) on 2 qubits
to produce any real superposition of 4 basis states:

    q0: ──Ry(θ₀)──●──────────
                   │
    q1: ──Ry(θ₁)──⊕──Ry(θ₂)──

Unlike Gallimore's UCC ansatz (3 qubits, 1-particle sector), here all
4 computational states are physical and directly encode the 4 HO basis
states.
"""

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit


def build_ansatz() -> QuantumCircuit:
    """Build the 3-parameter variational ansatz circuit.

    The circuit applies:
        1. Ry(theta_0) on qubit 0
        2. Ry(theta_1) on qubit 1
        3. CNOT from qubit 0 to qubit 1
        4. Ry(theta_2) on qubit 1

    Starting from |00>, this produces a general real 4-state superposition
    with 3 free parameters (matching the 3 degrees of freedom: 4 real
    amplitudes minus 1 normalization constraint).

    Returns:
        QuantumCircuit with parameters 'theta_0', 'theta_1', 'theta_2'.
    """
    theta_0 = Parameter("theta_0")
    theta_1 = Parameter("theta_1")
    theta_2 = Parameter("theta_2")

    circ = QuantumCircuit(2)
    circ.ry(theta_0, 0)
    circ.ry(theta_1, 1)
    circ.cx(0, 1)
    circ.ry(theta_2, 1)

    return circ


def ansatz_state(theta_0: float, theta_1: float, theta_2: float) -> np.ndarray:
    """Compute the ansatz state vector analytically.

    Traces through the circuit gates starting from |00>:

    After Ry(θ₀) on q0 and Ry(θ₁) on q1:
        |ψ> = (cos(θ₀/2)|0> + sin(θ₀/2)|1>) ⊗ (cos(θ₁/2)|0> + sin(θ₁/2)|1>)

    After CNOT(0→1):
        |00> → |00>, |01> → |01>, |10> → |11>, |11> → |10>

    After Ry(θ₂) on q1:
        Applies rotation to qubit 1 in each q0 branch.

    Returns:
        4-component real statevector [c_00, c_01, c_10, c_11].
    """
    c0 = np.cos(theta_0 / 2)
    s0 = np.sin(theta_0 / 2)
    c1 = np.cos(theta_1 / 2)
    s1 = np.sin(theta_1 / 2)
    c2 = np.cos(theta_2 / 2)
    s2 = np.sin(theta_2 / 2)

    # Qiskit state ordering: |q1 q0>, index = 2*q1 + q0
    #
    # After Ry(θ₀)⊗Ry(θ₁) in |q1 q0> basis:
    # |00>: c0*c1, |01>: s0*c1, |10>: c0*s1, |11>: s0*s1
    #
    # After CNOT(q0→q1): flips q1 when q0=1
    # |00>→|00>: c0*c1, |01>→|11>: s0*c1, |10>→|10>: c0*s1, |11>→|01>: s0*s1
    #
    # After Ry(θ₂) on q1: mixes |X0>↔|X1> for each q0 value
    # q0=0 branch (indices 00,10): before: |00>=c0*c1, |10>=c0*s1
    #   |00>: c0*(c1*c2 - s1*s2)
    #   |10>: c0*(c1*s2 + s1*c2)
    # q0=1 branch (indices 01,11): before: |01>=s0*s1, |11>=s0*c1
    #   |01>: s0*(s1*c2 - c1*s2)
    #   |11>: s0*(s1*s2 + c1*c2)

    sv = np.zeros(4)
    sv[0b00] = c0 * (c1 * c2 - s1 * s2)  # |00>
    sv[0b01] = s0 * (s1 * c2 - c1 * s2)  # |01>
    sv[0b10] = c0 * (c1 * s2 + s1 * c2)  # |10>
    sv[0b11] = s0 * (s1 * s2 + c1 * c2)  # |11>
    return sv


def physical_amplitudes(statevector: np.ndarray) -> np.ndarray:
    """Extract the 4 basis state amplitudes from the statevector.

    In the direct encoding, all 4 computational basis states are physical
    (unlike Gallimore where only 3 of 8 are in the 1-particle sector).

    Returns:
        [c_0, c_1, c_2, c_3] — real amplitudes for HO states n=0,1,2,3.
    """
    return np.array([statevector[i].real for i in range(4)])
