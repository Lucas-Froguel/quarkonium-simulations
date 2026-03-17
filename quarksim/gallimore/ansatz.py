"""UCC ansatz circuit for the 3-orbital quarkonium system.

Implements the unitary coupled cluster (UCC) ansatz from Gallimore & Liao
(arXiv:2202.03333v2), Section II.C and Fig. 1.

The ansatz prepares the state:
    |psi(alpha, beta)> = cos(alpha)|001> + sin(alpha)sin(beta)|010>
                         + sin(alpha)cos(beta)|100>

where |001> = orbital 0 occupied, |010> = orbital 1, |100> = orbital 2.
This is the most general single-particle state in 3 orbitals (up to phase).
"""

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit


def build_ansatz() -> QuantumCircuit:
    """Build the parameterized UCC ansatz circuit.

    The circuit uses two parameters (alpha, beta) and implements
    a low-depth decomposition of the UCC operator (Eq. 22):
        U(theta, phi) = exp{ theta(a†_1 a_0 - h.c.) + phi(a†_2 a_0 - h.c.) }

    with alpha = sqrt(theta^2 + phi^2) and sin(beta) = theta/alpha.

    Qubit ordering: qubit j = orbital j.  The Qiskit state |q2 q1 q0>
    maps to occupation numbers |f2 f1 f0>.

    Returns:
        QuantumCircuit with parameters 'alpha' and 'beta'.
    """
    alpha = Parameter("alpha")
    beta = Parameter("beta")

    circ = QuantumCircuit(3)

    circ.ry(beta, 1)
    circ.ry(2 * alpha, 0)
    circ.cx(0, 2)
    circ.cx(2, 1)
    circ.x(0)
    circ.ry(-beta, 1)
    circ.cx(2, 1)
    circ.cx(1, 2)

    return circ


def ansatz_state(alpha: float, beta: float) -> np.ndarray:
    """Compute the UCC ansatz state vector analytically.

    Returns the 8-component statevector in the computational basis.
    Only states |001>, |010>, |100> have nonzero amplitude (Eq. 23).
    """
    sv = np.zeros(8)
    sv[0b001] = np.cos(alpha)            # |001> = orbital 0
    sv[0b010] = np.sin(alpha) * np.sin(beta)  # |010> = orbital 1
    sv[0b100] = np.sin(alpha) * np.cos(beta)  # |100> = orbital 2
    return sv


def orthogonal_state(alpha0: float, beta0: float, gamma: float) -> np.ndarray:
    """Build a state orthogonal to |psi(alpha0, beta0)>, parametrized by gamma.

    Implements the orthogonalization from Section II.D (Eqs. 37-39):
        cos(alpha1)          = -sin(alpha0) cos(gamma)
        sin(alpha1) sin(beta1) = cos(alpha0) sin(beta0) cos(gamma) + cos(beta0) sin(gamma)
        sin(alpha1) cos(beta1) = cos(alpha0) cos(beta0) cos(gamma) - sin(beta0) sin(gamma)

    The resulting state is guaranteed orthogonal to |psi(alpha0, beta0)>
    for any value of gamma. Sweeping gamma scans the full orthogonal subspace.

    Returns an 8-component statevector.
    """
    c0 = -np.sin(alpha0) * np.cos(gamma)
    c1 = np.cos(alpha0) * np.sin(beta0) * np.cos(gamma) + np.cos(beta0) * np.sin(gamma)
    c2 = np.cos(alpha0) * np.cos(beta0) * np.cos(gamma) - np.sin(beta0) * np.sin(gamma)

    sv = np.zeros(8)
    sv[0b001] = c0
    sv[0b010] = c1
    sv[0b100] = c2
    return sv


def third_state(sv0: np.ndarray, sv1: np.ndarray) -> np.ndarray:
    """Compute the third orthogonal state from the first two.

    With 3 orbitals, once 2 orthonormal states are known the third
    is fully determined (up to sign) as the cross product in the
    3D amplitude space.

    Returns an 8-component statevector.
    """
    a0 = physical_amplitudes(sv0)
    a1 = physical_amplitudes(sv1)
    a2 = np.cross(a0, a1)
    a2 /= np.linalg.norm(a2)

    sv = np.zeros(8)
    sv[0b001] = a2[0]
    sv[0b010] = a2[1]
    sv[0b100] = a2[2]
    return sv


def physical_amplitudes(statevector: np.ndarray) -> np.ndarray:
    """Extract 1-particle amplitudes from a full 8-component statevector.

    Returns [c_0, c_1, c_2] where c_j is the real amplitude for orbital j.
    (Imaginary parts are zero for the UCC ansatz with real parameters.)
    """
    return np.array([
        statevector[0b001].real,  # orbital 0
        statevector[0b010].real,  # orbital 1
        statevector[0b100].real,  # orbital 2
    ])
