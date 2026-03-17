"""Hamiltonian construction for the nonrelativistic quark model.

Implements the charmonium Hamiltonian from Woloshyn, "Demonstrating quantum
computing with the quark model" (arXiv:2301.10828v2), Sections 2.1--2.2.

The potential is the Cornell potential with a spin-dependent contact term:
    V(r) = -a/r + b*r + V_s(r) * S_c . S_cbar

Three channels are supported:
    - 1S0 (eta_c):  l=0, spin singlet (S.S = -3/4)
    - 3S1 (J/psi):  l=0, spin triplet (S.S = +1/4)
    - 1P1 (h_c):    l=1, spin singlet (S.S = -3/4, but no contact term for l>0)

The Hamiltonian is expanded in a harmonic oscillator basis with N=4 states
and encoded on 2 qubits via direct state mapping (not Jordan-Wigner).

Two modes of matrix construction:
  1. Paper matrices (default): Hardcoded from Eqs. (4), (15), (16).
     Guaranteed to reproduce the paper's Pauli coefficients and eigenvalues.
  2. Computed matrices: Numerical integration from physics parameters.
     Useful for parameter variation and validation.
"""

import numpy as np
import scipy.special as sp
import scipy.integrate as integrate
from qiskit.quantum_info import SparsePauliOp

# Parameters from Barnes et al. via Woloshyn Table 1
CHARMONIUM_PARAMS = {
    "alpha_s": 0.5461,        # Strong coupling constant
    "b": 0.1425,              # GeV^2 — string tension
    "m_c": 1.4794,            # GeV — charm quark mass
    "sigma_smear": 1.0946,    # GeV — Gaussian smearing parameter for delta(r)
    "omega": 1.2,             # fm^-1 — HO frequency (chosen from Fig. 1)
    "hbar_c": 0.197326,       # GeV*fm — conversion factor
}

# Spin expectation values: S_c . S_cbar
SPIN_PRODUCT = {
    "1S0": -3.0 / 4.0,   # Singlet
    "3S1": 1.0 / 4.0,    # Triplet
    "1P1": -3.0 / 4.0,   # Singlet (contact term vanishes for l>0)
}

CHANNEL_L = {
    "1S0": 0,
    "3S1": 0,
    "1P1": 1,
}

# Hardcoded Hamiltonian matrices from the paper (fm^-1, omega = 1.2 fm^-1)
# Note: The paper's Eq. (16) for 1P1 has a probable typo in the (3,3)
# element (7.5104 = same as 3S1), which does not reproduce Table 3
# eigenvalues. We use the corrected value 8.6034 computed to match
# the Table 3 trace.
PAPER_MATRICES = {
    "1S0": np.array([   # Eq. (4)
        [0.9431, -0.8733, -0.7690, -0.5601],
        [-0.8733, 3.33652, -0.5646, -0.8648],
        [-0.7690, -0.5646, 5.4382, -0.1566],
        [-0.5601, -0.8648, -0.1566, 7.3451],
    ]),
    "3S1": np.array([   # Eq. (15) — Appendix
        [1.0946, -0.7114, -0.6111, -0.4112],
        [-0.7114, 3.5406, -0.3910, -0.6989],
        [-0.6111, -0.3910, 5.6122, -0.0119],
        [-0.4112, -0.6989, -0.0119, 7.5104],
    ]),
    "1P1": np.array([   # Eq. (16) — Appendix (corrected (3,3) element)
        [2.8561, -0.2395, -0.3827, -0.2282],
        [-0.2395, 4.919, -0.0373, -0.5097],
        [-0.3827, -0.0373, 6.8114, 0.4058],
        [-0.2282, -0.5097, 0.4058, 8.6034],
    ]),
}

# Pauli coefficients from the paper (Section 2.2.1) for validation
# Order: c0(II), c1(IZ), c2(ZI), c3(ZZ), c4(IX), c5(XI), c6(ZX), c7(XZ), c8(XX), c9(YY)
PAPER_PAULI_COEFFICIENTS = {
    "1S0": [4.273, -2.119, -1.082, -0.129, -0.817, -0.515, -0.358, 0.048, -0.562, -0.002],
    "3S1": [4.439, -2.122, -1.086, -0.137, -0.655, -0.350, -0.361, 0.044, -0.401, 0.010],
    "1P1": [5.798, -1.910, -0.964, -0.067, -0.446, 0.083, -0.323, -0.064, -0.095, 0.133],
}

# Expected eigenvalues from Table 3 (fm^-1)
PAPER_EIGENVALUES = {
    "1S0": [0.395, 3.506, 5.664, 7.546],
    "3S1": [0.753, 3.634, 5.723, 7.648],
    "1P1": [2.783, 4.875, 6.765, 8.767],
}

CHANNELS = ["1S0", "3S1", "1P1"]


def get_matrix(channel: str) -> np.ndarray:
    """Return the paper's Hamiltonian matrix for a given channel.

    This is the primary interface for the simulation — uses the paper's
    explicit matrices to guarantee reproducibility.

    Args:
        channel: One of "1S0", "3S1", "1P1".

    Returns:
        4x4 real symmetric matrix in fm^-1.
    """
    return PAPER_MATRICES[channel].copy()


def build_pauli_hamiltonian(channel: str) -> SparsePauliOp:
    """Build the 2-qubit Pauli Hamiltonian for a given channel.

    Uses the paper's matrix for the given channel. Maps the 4x4 matrix
    to Pauli operators using direct encoding:
        |00> = state 0, |01> = state 1, |10> = state 2, |11> = state 3.

    For a real symmetric 4x4 matrix, 10 Pauli terms contribute.

    Args:
        channel: One of "1S0", "3S1", "1P1".

    Returns:
        SparsePauliOp representing the 2-qubit Hamiltonian.
    """
    H = get_matrix(channel)
    return matrix_to_pauli(H)


def matrix_to_pauli(H: np.ndarray) -> SparsePauliOp:
    """Decompose a 4x4 real symmetric matrix into a 2-qubit Pauli sum.

    Uses c_k = Tr(H @ P_k) / 4 for each of the 10 real Pauli operators.

    Qiskit convention: in Pauli string 'AB', A acts on qubit 1, B on qubit 0.

    Args:
        H: 4x4 real symmetric matrix.

    Returns:
        SparsePauliOp with up to 10 terms.
    """
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    pauli_ops = {
        "II": np.kron(I2, I2),
        "IZ": np.kron(I2, Z),    # Z on qubit 0
        "ZI": np.kron(Z, I2),    # Z on qubit 1
        "ZZ": np.kron(Z, Z),
        "IX": np.kron(I2, X),    # X on qubit 0
        "XI": np.kron(X, I2),    # X on qubit 1
        "ZX": np.kron(Z, X),     # Z on qubit 1, X on qubit 0
        "XZ": np.kron(X, Z),     # X on qubit 1, Z on qubit 0
        "XX": np.kron(X, X),
        "YY": np.kron(Y, Y),
    }

    terms = []
    for label, P in pauli_ops.items():
        coeff = np.trace(H @ P).real / 4.0
        if abs(coeff) > 1e-15:
            terms.append((label, coeff))

    if not terms:
        terms.append(("II", 0.0))

    return SparsePauliOp.from_list(terms)


def validate_pauli_hamiltonian(
    pauli_ham: SparsePauliOp,
    phys_matrix: np.ndarray,
    atol: float = 1e-6,
):
    """Check that the Pauli Hamiltonian reproduces the physical matrix.

    Unlike Gallimore (which projects onto a 1-particle subspace), here all
    4 computational basis states are physical, so the full 4x4 Pauli matrix
    should match directly.

    Raises AssertionError if matrices disagree.
    """
    reconstructed = pauli_ham.to_matrix().real
    if not np.allclose(reconstructed, phys_matrix, atol=atol):
        diff = np.max(np.abs(reconstructed - phys_matrix))
        raise AssertionError(
            f"Pauli Hamiltonian does not match physical matrix. Max diff: {diff:.2e}\n"
            f"Reconstructed:\n{reconstructed}\n"
            f"Physical:\n{phys_matrix}"
        )


# ---------------------------------------------------------------------------
# From-scratch matrix computation (for reference and parameter variation)
# ---------------------------------------------------------------------------

def _to_fm_inv(value_gev: float, hbar_c: float = 0.197326) -> float:
    """Convert a quantity in GeV to fm^-1."""
    return value_gev / hbar_c


def _ho_wavefunction(r: np.ndarray, n: int, l: int, nu: float) -> np.ndarray:
    """Normalized radial function u(r) = r * R(r) for the 3D HO.

    Satisfies integral_0^inf |u(r)|^2 dr = 1.

    Args:
        r: Radial coordinate array (fm).
        n: Radial quantum number (0, 1, 2, ...).
        l: Orbital angular momentum.
        nu: mu * omega (fm^-2).
    """
    norm_sq = 2.0 * nu ** (l + 1.5) * sp.gamma(n + 1) / sp.gamma(n + l + 1.5)
    norm = np.sqrt(norm_sq)
    x = nu * r**2
    laguerre = sp.eval_genlaguerre(n, l + 0.5, x)
    return norm * r ** (l + 1) * np.exp(-0.5 * x) * laguerre


def _compute_matrix_element(m: int, n: int, l: int, nu: float, func) -> float:
    """Compute <m|f(r)|n> = integral_0^inf u_m(r) f(r) u_n(r) dr."""
    def integrand(r):
        if r < 1e-15:
            return 0.0
        um = _ho_wavefunction(np.array([r]), m, l, nu)[0]
        un = _ho_wavefunction(np.array([r]), n, l, nu)[0]
        return um * func(r) * un

    result, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    return result


def compute_matrix(channel: str, n_basis: int = 4, **params) -> np.ndarray:
    """Compute Hamiltonian matrix from physics parameters (numerical integration).

    Implements H_qm = H_HO + V(r) - (1/2)*mu*omega^2*r^2  (Eq. 3)

    This function computes from scratch rather than using the paper's
    hardcoded matrices. Useful for parameter studies.

    Args:
        channel: One of "1S0", "3S1", "1P1".
        n_basis: Number of HO basis states (default 4).
        **params: Override any CHARMONIUM_PARAMS.

    Returns:
        Real symmetric matrix of shape (n_basis, n_basis) in fm^-1.
    """
    p = {**CHARMONIUM_PARAMS, **params}
    l = CHANNEL_L[channel]
    omega = p["omega"]
    hbar_c = p["hbar_c"]
    mu = _to_fm_inv(p["m_c"] / 2.0, hbar_c)
    nu = mu * omega
    a_coulomb = 4.0 * p["alpha_s"] / 3.0
    b_string = p["b"] / hbar_c**2
    sigma_fm = _to_fm_inv(p["sigma_smear"], hbar_c)

    spin_ss = SPIN_PRODUCT[channel]
    mc_fm = _to_fm_inv(p["m_c"], hbar_c)
    vs_prefactor = 32.0 * np.pi * p["alpha_s"] / (9.0 * mc_fm**2) * spin_ss

    H = np.zeros((n_basis, n_basis))
    for m_idx in range(n_basis):
        for n_idx in range(m_idx, n_basis):
            ho_energy = omega * (2 * n_idx + l + 1.5) if m_idx == n_idx else 0.0

            def potential(r):
                v = -a_coulomb / r + b_string * r - 0.5 * mu * omega**2 * r**2
                if l == 0:
                    delta_r = (sigma_fm / np.sqrt(np.pi)) ** 3 * np.exp(
                        -sigma_fm**2 * r**2
                    )
                    v += vs_prefactor * delta_r
                return v

            v_mn = _compute_matrix_element(m_idx, n_idx, l, nu, potential)
            H[m_idx, n_idx] = ho_energy + v_mn
            if m_idx != n_idx:
                H[n_idx, m_idx] = H[m_idx, n_idx]

    return H
