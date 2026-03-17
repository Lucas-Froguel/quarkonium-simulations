"""Cornell potential Hamiltonian in the quantum harmonic oscillator basis.

Implements the Hamiltonian from Gallimore & Liao, "Quantum Computing for
Heavy Quarkonium Spectroscopy" (arXiv:2202.03333v2), Sections II.A and II.B.

The charmonium system uses the Cornell potential V(r) = -kappa/r + sigma*r
expanded in QHO s-wave states, then mapped to qubit operators via the
Jordan-Wigner transformation.
"""

import numpy as np
import scipy.special as sp
import mpmath
from qiskit.quantum_info import SparsePauliOp

# Default charmonium parameters from the paper (Section II.A)
CHARMONIUM_PARAMS = {
    "omega": 562.9,       # MeV — oscillator frequency
    "kappa": 0.4063,      # dimensionless — color Coulomb coupling
    "sigma": 441.6**2,    # MeV^2 — string tension (sqrt(sigma) = 441.6 MeV)
    "mu": 637.5,          # MeV — reduced mass of c-cbar pair
}


def oscillator_length(mu: float, omega: float) -> float:
    """Oscillator length b = 1/sqrt(mu * omega).  Eq. (4)."""
    return 1.0 / np.sqrt(mu * omega)


def kinetic_element(m: int, n: int, omega: float) -> float:
    """Kinetic energy matrix element <m|T|n> in the QHO s-wave basis.

    Eq. (5):
        <m|T|n> = (omega/2) * [ (2n + 3/2) delta_{mn}
                                - sqrt(n(n+1/2)) delta_{m, n-1}
                                - sqrt((n+1)(n+3/2)) delta_{m, n+1} ]

    The kinetic energy operator is tridiagonal: it only couples
    adjacent oscillator states n and n+/-1.
    """
    if m == n:
        return 0.5 * omega * (2 * n + 1.5)
    if m == n - 1:
        return -0.5 * omega * np.sqrt(n * (n + 0.5))
    if m == n + 1:
        return -0.5 * omega * np.sqrt((n + 1) * (n + 1.5))
    return 0.0


def potential_element(m: int, n: int, b: float, kappa: float, sigma: float) -> float:
    """Potential energy matrix element <m|V|n> for the Cornell potential.

    V(r) = -kappa/r + sigma*r

    Uses Eqs. (6)-(7) with hypergeometric functions:
        <m|r^{-1}|n> via 3F2  — Eq. (7)
        <m|r|n>      via 2F1  — Eq. (6)
    """
    prefactor = np.sqrt(
        sp.gamma(m + 1.5) * sp.gamma(n + 1.5)
        / (sp.gamma(m + 1) * sp.gamma(n + 1))
    )
    sign = (-1) ** (m + n)

    # <m|r^{-1}|n>, Eq. (7)
    r_inv = (
        sign
        * 4.0
        / (b * np.pi * (1 + 2 * n))
        * prefactor
        * float(mpmath.hyp3f2(0.5, 1, -m, 1.5, 0.5 - n, 1))
    )

    # <m|r|n>, Eq. (6)
    r_lin = (
        sign
        * 4.0
        * b
        / (np.pi * (1 - 4 * n**2))
        * prefactor
        * float(mpmath.hyp2f1(2, -m, 1.5 - n, 1))
    )

    return -kappa * r_inv + sigma * r_lin


def build_matrix(n_orbitals: int = 3, **params) -> np.ndarray:
    """Build the n_orbitals x n_orbitals Hamiltonian matrix h_{mn} = <m|T+V|n>.

    This is the physical Hamiltonian in the QHO basis before
    mapping to qubits. Eigenvalues give the energy levels.

    Args:
        n_orbitals: Number of QHO basis states (default 3).
        **params: Override any of CHARMONIUM_PARAMS.

    Returns:
        Real symmetric matrix of shape (n_orbitals, n_orbitals).
    """
    p = {**CHARMONIUM_PARAMS, **params}
    omega = p["omega"]
    b = oscillator_length(p["mu"], omega)

    H = np.zeros((n_orbitals, n_orbitals))
    for m in range(n_orbitals):
        for n in range(n_orbitals):
            H[m, n] = kinetic_element(m, n, omega) + potential_element(
                m, n, b, p["kappa"], p["sigma"]
            )
    return H


def build_pauli_hamiltonian(n_orbitals: int = 3, **params) -> SparsePauliOp:
    """Build the Pauli Hamiltonian via Jordan-Wigner transformation.

    Maps the second-quantized Hamiltonian H = sum_{m,n} h_{mn} a†_m a_n
    onto qubit operators using the Jordan-Wigner transformation (Eqs. 8-9).

    For N=3 orbitals this produces 10 Pauli terms (Eqs. 11-20):
        - 1 identity (constant energy offset)
        - 3 single-Z terms (diagonal energies)
        - 4 XX/YY terms (adjacent orbital couplings)
        - 2 XZX/YZY terms (non-adjacent orbital coupling via Z string)

    Qiskit convention: in a Pauli string 'ABC', A acts on qubit 2,
    B on qubit 1, C on qubit 0.  Qubit j = orbital j.

    Args:
        n_orbitals: Number of orbitals (only 3 supported).
        **params: Override any of CHARMONIUM_PARAMS.

    Returns:
        SparsePauliOp representing the qubit Hamiltonian.
    """
    if n_orbitals != 3:
        raise NotImplementedError("Only n_orbitals=3 is currently supported.")

    h = build_matrix(n_orbitals, **params)

    # Jordan-Wigner mapping for 3 orbitals:
    #   Diagonal:      a†_n a_n = (I - Z_n) / 2
    #   Adjacent:      a†_m a_n + h.c. = (X_m X_n + Y_m Y_n) / 2   (|m-n|=1)
    #   Non-adjacent:  a†_0 a_2 + h.c. = (X_0 Z_1 X_2 + Y_0 Z_1 Y_2) / 2

    terms = []

    # Eq. (11): Identity — constant offset from diagonal h_{nn}(I - Z_n)/2
    c_I = 0.5 * (h[0, 0] + h[1, 1] + h[2, 2])
    terms.append(("III", c_I))

    # Eqs. (12)-(14): Single-Z operators
    terms.append(("IIZ", -0.5 * h[0, 0]))  # -h_{00}/2 · Z_0
    terms.append(("IZI", -0.5 * h[1, 1]))  # -h_{11}/2 · Z_1
    terms.append(("ZII", -0.5 * h[2, 2]))  # -h_{22}/2 · Z_2

    # Eqs. (15)-(16): X_m X_n for adjacent orbitals
    terms.append(("IXX", 0.5 * h[0, 1]))  # h_{01}/2 · X_0 X_1
    terms.append(("XXI", 0.5 * h[1, 2]))  # h_{12}/2 · X_1 X_2

    # Eqs. (17)-(18): Y_m Y_n for adjacent orbitals
    terms.append(("IYY", 0.5 * h[0, 1]))  # h_{01}/2 · Y_0 Y_1
    terms.append(("YYI", 0.5 * h[1, 2]))  # h_{12}/2 · Y_1 Y_2

    # Eqs. (19)-(20): X_0 Z_1 X_2 and Y_0 Z_1 Y_2 (non-adjacent)
    terms.append(("XZX", 0.5 * h[0, 2]))  # h_{02}/2 · X_0 Z_1 X_2
    terms.append(("YZY", 0.5 * h[0, 2]))  # h_{02}/2 · Y_0 Z_1 Y_2

    return SparsePauliOp.from_list(terms)


def validate_pauli_hamiltonian(pauli_ham: SparsePauliOp, phys_matrix: np.ndarray):
    """Check that the Pauli Hamiltonian matches the physical matrix.

    Projects the 8x8 Pauli Hamiltonian onto the 1-particle subspace
    and compares with the 3x3 physical Hamiltonian matrix.

    Raises AssertionError if the matrices disagree beyond numerical tolerance.
    """
    full = pauli_ham.to_matrix()
    n = phys_matrix.shape[0]

    # 1-particle states: |001>=1, |010>=2, |100>=4
    indices = [1 << i for i in range(n)]
    projected = np.array([[full[i, j] for j in indices] for i in indices]).real

    if not np.allclose(projected, phys_matrix, atol=1e-10):
        diff = np.max(np.abs(projected - phys_matrix))
        raise AssertionError(
            f"Pauli Hamiltonian does not match physical matrix. Max diff: {diff:.2e}\n"
            f"Projected:\n{projected}\n"
            f"Physical:\n{phys_matrix}"
        )
