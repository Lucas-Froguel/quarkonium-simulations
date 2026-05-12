"""Three-parameter UCC ansatz on 4 qubits for the spin-dependent N=4 case.

Extension of the Gallimore & Liao UCC ansatz from N=3 to N=4 orbitals.
Starting from the reference state |0001> (one excitation on qubit 0 =
orbital 0), the ansatz applies a chain of three Givens rotations between
adjacent orbital pairs:

    |0001>  -- G_{0,1}(theta_3) -->
       cos(theta_3)|0001> + sin(theta_3)|0010>
    -- G_{1,2}(theta_2) -->
       cos(theta_3)|0001> + sin(theta_3)cos(theta_2)|0010>
                          + sin(theta_3)sin(theta_2)|0100>
    -- G_{2,3}(theta_1) -->
       cos(theta_3)|0001> + sin(theta_3)cos(theta_2)|0010>
                          + sin(theta_3)sin(theta_2)cos(theta_1)|0100>
                          + sin(theta_3)sin(theta_2)sin(theta_1)|1000>

producing the spherical-coordinate single-particle superposition. Each
Givens rotation between qubits j and j+1 is implemented as
    CNOT(j+1, j) -- CRY(2 theta)(j -> j+1) -- CNOT(j+1, j)
(standard hardware decomposition).
"""

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit


def _givens(circ: QuantumCircuit, qj: int, qj1: int, theta) -> None:
    """In-place Givens rotation between qubits j and j+1 on the 1-particle
    subspace {|01>, |10>}.

    Implements the rotation
        |01> -> cos(theta)|01> + sin(theta)|10>
        |10> -> -sin(theta)|01> + cos(theta)|10>
    via CNOT(qj1, qj) -- CRY(2 theta)(qj -> qj1) -- CNOT(qj1, qj).
    """
    circ.cx(qj1, qj)
    circ.cry(2.0 * theta, qj, qj1)
    circ.cx(qj1, qj)


def build_ucc4() -> QuantumCircuit:
    """Build the parameterised 3-parameter UCC ansatz on 4 qubits.

    Parameters (in circuit order, matching the spherical formula):
        theta_3 = first Givens angle  (between qubits 0 and 1)
        theta_2 = second Givens angle (between qubits 1 and 2)
        theta_1 = third Givens angle  (between qubits 2 and 3)

    Returns:
        QuantumCircuit on 4 qubits with three parameters.
    """
    t3 = Parameter("theta_3")
    t2 = Parameter("theta_2")
    t1 = Parameter("theta_1")

    circ = QuantumCircuit(4)

    # Reference state |0001>: occupy qubit 0
    circ.x(0)

    # Chain of Givens rotations
    _givens(circ, 0, 1, t3)
    _givens(circ, 1, 2, t2)
    _givens(circ, 2, 3, t1)

    return circ


def ucc4_state(theta_3: float, theta_2: float, theta_1: float) -> np.ndarray:
    """Analytical 16-component statevector produced by the UCC ansatz."""
    sv = np.zeros(16)
    c3, s3 = np.cos(theta_3), np.sin(theta_3)
    c2, s2 = np.cos(theta_2), np.sin(theta_2)
    c1, s1 = np.cos(theta_1), np.sin(theta_1)
    # Indices (decimal) of the four physical basis states (Qiskit big-endian):
    #   |0001>=1, |0010>=2, |0100>=4, |1000>=8
    sv[1] = c3
    sv[2] = s3 * c2
    sv[4] = s3 * s2 * c1
    sv[8] = s3 * s2 * s1
    return sv
