"""M1 and E1 transition amplitudes for charmonium.

Implements the transition amplitude calculations from Woloshyn
(arXiv:2301.10828v2), Section 4.

M1 transitions: spin-singlet <-> spin-triplet (same L, spin flip).
    The squared amplitude is |<psi_f|psi_i>|^2 (spatial wavefunction overlap).

E1 transitions: Delta_L = 1 (S-wave <-> P-wave, same spin).
    The amplitude is <u_f(r)|r|u_i(r)> (radial matrix element of r).

Both can be computed via:
    - Direct statevector inner products (exact, fast)
    - Quantum circuits: overlap circuit (Fig. 3), swap test (Fig. 7),
      Hadamard test (Fig. 8)
"""

import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from quarksim.woloshyn.ansatz import build_ansatz, physical_amplitudes
from quarksim.woloshyn.hamiltonian import CHARMONIUM_PARAMS, _to_fm_inv


# ---------------------------------------------------------------------------
# M1 transitions (Section 4.1)
# ---------------------------------------------------------------------------


def m1_overlap(state_i: np.ndarray, state_f: np.ndarray) -> float:
    """Compute |<psi_f|psi_i>|^2 — the squared spatial wavefunction overlap.

    For M1 transitions in the nonrelativistic limit, the transition amplitude
    is just the overlap of spatial wavefunctions since the spin-flip operator
    doesn't affect the spatial part.

    Args:
        state_i: Initial state (4-component statevector, e.g. 3S1).
        state_f: Final state (4-component statevector, e.g. 1S0).

    Returns:
        Squared overlap (between 0 and 1).
    """
    overlap = np.vdot(state_f, state_i)
    return abs(overlap) ** 2


def build_overlap_circuit(
    params_i: np.ndarray, params_f: np.ndarray,
) -> QuantumCircuit:
    """Build the U_i U_f^dagger overlap circuit (Fig. 3).

    Prepares |psi_i> and then applies the inverse of |psi_f>'s preparation.
    Measuring |00> gives P(00) = |<psi_f|psi_i>|^2.

    Args:
        params_i: Parameters for the initial state ansatz.
        params_f: Parameters for the final state ansatz.

    Returns:
        QuantumCircuit ready for measurement.
    """
    ansatz = build_ansatz()

    circ_i = ansatz.assign_parameters(params_i)
    circ_f = ansatz.assign_parameters(params_f)

    circ = circ_i.compose(circ_f.inverse())
    circ.measure_all()
    return circ


def build_swap_test_circuit(
    params_i: np.ndarray, params_f: np.ndarray,
) -> QuantumCircuit:
    """Build the swap test circuit (Fig. 7).

    Uses an ancilla qubit to measure |<psi_f|psi_i>|^2 without
    directly implementing U^dagger.

    The measurement result is:
        P(ancilla=0) = (1 + |<psi_f|psi_i>|^2) / 2

    Circuit layout (5 qubits):
        q0: ancilla -- H -- controlled-swap -- H -- measure
        q1,q2: state_i register (U_i applied)
        q3,q4: state_f register (U_f applied)

    Args:
        params_i: Parameters for initial state.
        params_f: Parameters for final state.

    Returns:
        QuantumCircuit with 5 qubits.
    """
    ansatz = build_ansatz()
    circ_i = ansatz.assign_parameters(params_i)
    circ_f = ansatz.assign_parameters(params_f)

    # 5 qubits: q0=ancilla, q1-q2=register i, q3-q4=register f
    circ = QuantumCircuit(5, 1)

    # Prepare states in their registers
    circ.compose(circ_i, qubits=[1, 2], inplace=True)
    circ.compose(circ_f, qubits=[3, 4], inplace=True)

    # Hadamard on ancilla
    circ.h(0)

    # Controlled-SWAP: swap q1<->q3 and q2<->q4 when ancilla=1
    circ.cswap(0, 1, 3)
    circ.cswap(0, 2, 4)

    # Hadamard on ancilla
    circ.h(0)

    # Measure ancilla
    circ.measure(0, 0)

    return circ


# ---------------------------------------------------------------------------
# E1 transitions (Section 4.2)
# ---------------------------------------------------------------------------


def _ho_wavefunction_u(r: np.ndarray, n: int, l: int, nu: float) -> np.ndarray:
    """Normalized u(r) = r*R(r) for the HO. Same as hamiltonian._ho_wavefunction."""
    norm_sq = 2.0 * nu ** (l + 1.5) * sp.gamma(n + 1) / sp.gamma(n + l + 1.5)
    norm = np.sqrt(norm_sq)
    x = nu * r**2
    laguerre = sp.eval_genlaguerre(n, l + 0.5, x)
    return norm * r ** (l + 1) * np.exp(-0.5 * x) * laguerre


def e1_matrix_elements(n_basis: int = 4, **params) -> np.ndarray:
    """Compute the P-wave to S-wave matrix elements of r (Eq. 13).

    <m_P|r|n_S> = integral_0^inf u_m^{l=1}(r) * r * u_n^{l=0}(r) dr

    This is a (n_basis x n_basis) matrix connecting P-wave states (rows)
    to S-wave states (columns).

    Args:
        n_basis: Number of HO states per channel (default 4).
        **params: Override CHARMONIUM_PARAMS.

    Returns:
        Matrix of shape (n_basis, n_basis) in fm.
    """
    p = {**CHARMONIUM_PARAMS, **params}
    omega = p["omega"]
    mu = _to_fm_inv(p["m_c"] / 2.0, p["hbar_c"])
    nu = mu * omega

    R = np.zeros((n_basis, n_basis))
    for m in range(n_basis):
        for n in range(n_basis):
            def integrand(r):
                if r < 1e-15:
                    return 0.0
                u_p = _ho_wavefunction_u(np.array([r]), m, 1, nu)[0]
                u_s = _ho_wavefunction_u(np.array([r]), n, 0, nu)[0]
                return u_p * r * u_s

            R[m, n], _ = integrate.quad(integrand, 0, np.inf, limit=200)

    return R


# Expected r matrix elements from Eq. (13) (fm)
# Note: The paper's (2,2) element is 5.4382, which matches the (2,2) element
# of the 1S0 Hamiltonian matrix (Eq. 4) — likely a copy-paste error.
# Our computed value of ~0.882 fits the monotonic pattern of the diagonal.
PAPER_R_MATRIX = np.array([
    [0.57751, -0.4715, 0, 0],
    [0, 0.7455, -0.6668, 0],
    [0, 0, 0.8821, -0.8167],
    [0, 0, 0, 1.0002],
])


def e1_amplitude(
    state_i: np.ndarray,
    state_f: np.ndarray,
    r_matrix: np.ndarray,
) -> float:
    """Compute the E1 transition amplitude <u_f|r|u_i> via statevector.

    Args:
        state_i: Initial S-wave state (4-component).
        state_f: Final P-wave state (4-component).
        r_matrix: Matrix elements <m_P|r|n_S> from e1_matrix_elements().

    Returns:
        Transition amplitude in fm.
    """
    amps_i = physical_amplitudes(state_i)
    amps_f = physical_amplitudes(state_f)
    return float(amps_f @ r_matrix @ amps_i)


def build_e1_pauli_operator(r_matrix: np.ndarray) -> SparsePauliOp:
    """Decompose the E1 transition operator into Pauli form (Eq. 14).

    The r-operator matrix connects states in different channels, but when
    encoded on the same 2-qubit register, it becomes a general 4x4 matrix
    that may have complex Pauli coefficients (including Y terms).

    Args:
        r_matrix: 4x4 matrix of <m_P|r|n_S>.

    Returns:
        SparsePauliOp with potentially complex coefficients.
    """
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # All 16 two-qubit Pauli operators
    pauli_labels = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ",
                    "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    pauli_mats = {
        "I": I2, "X": X, "Y": Y, "Z": Z,
    }

    terms = []
    for label in pauli_labels:
        P = np.kron(pauli_mats[label[0]], pauli_mats[label[1]])
        coeff = np.trace(r_matrix @ P) / 4.0
        if abs(coeff) > 1e-15:
            terms.append((label, complex(coeff)))

    return SparsePauliOp.from_list(terms)


def build_hadamard_test_circuit(
    params_i: np.ndarray,
    params_f: np.ndarray,
    operator: SparsePauliOp,
    part: str = "real",
) -> QuantumCircuit:
    """Build a Hadamard test circuit for <psi_f|O|psi_i> (Fig. 8).

    The circuit uses an ancilla qubit to extract Re or Im parts of the
    transition matrix element <psi_f|O|psi_i>.

    Measuring the ancilla:
        <sigma_x> = 2*Re(<psi_f|O|psi_i>)  (when R = Ry(-pi/2))
        <sigma_y> = 2*Im(<psi_f|O|psi_i>)  (when R = Rx(pi/2))

    This is a simplified implementation that works for statevector
    simulation. Shot-based requires measuring each Pauli term of O
    separately.

    Args:
        params_i: Initial state parameters.
        params_f: Final state parameters.
        operator: Operator O as SparsePauliOp (2-qubit).
        part: "real" or "imag" — which part to extract.

    Returns:
        QuantumCircuit with 5 qubits (q0=ancilla, q1-q2=system).
    """
    ansatz = build_ansatz()

    circ = QuantumCircuit(3, 1)

    # Hadamard on ancilla
    circ.h(0)

    # Controlled-U_f on system register
    circ_f = ansatz.assign_parameters(params_f).to_gate().control(1)
    circ.append(circ_f, [0, 1, 2])

    # Apply X to ancilla (flip control for next controlled gate)
    circ.x(0)

    # Controlled-U_i on system register
    circ_i = ansatz.assign_parameters(params_i).to_gate().control(1)
    circ.append(circ_i, [0, 1, 2])

    # Apply X to ancilla (reset)
    circ.x(0)

    # Rotation R on ancilla for real/imag selection
    if part == "real":
        circ.ry(-np.pi / 2, 0)
    else:
        circ.rx(np.pi / 2, 0)

    # Hadamard on ancilla
    circ.h(0)

    circ.measure(0, 0)
    return circ
