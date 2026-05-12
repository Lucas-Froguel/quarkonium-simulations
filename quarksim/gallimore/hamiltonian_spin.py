"""Jordan-Wigner Hamiltonian for the spin-dependent N=4 charmonium matrices.

Extension of the Gallimore & Liao construction (N=3, spinless) to the
N=4 spin-dependent charmonium channels of Woloshyn. The physical 4x4
matrix elements h_{mn} are taken from the Woloshyn paper (Eqs. 4, 15, 16);
the Jordan-Wigner mapping then expresses the second-quantised operator

    H_hat = sum_{m,n} h_{mn} a_m^dagger a_n

as a 17-term Pauli sum on 4 qubits. Physical states live in the
single-particle sector, i.e. {|0001>, |0010>, |0100>, |1000>}.

Qiskit convention: in a Pauli string 'ABCD', A acts on qubit 3, D on qubit 0.
Orbital n is mapped to qubit n.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quarksim.woloshyn.hamiltonian import get_matrix as get_woloshyn_matrix


# -----------------------------------------------------------------------
# Single-particle basis (in Qiskit big-endian convention "q3 q2 q1 q0")
# -----------------------------------------------------------------------
# Orbital n occupied => bitstring with '1' on position n (qubit n = LSB n).
# In Qiskit string indexing (leftmost = q3), the bitstring is reversed:
#   orbital 0 -> '0001'  (q0 = 1)
#   orbital 1 -> '0010'  (q1 = 1)
#   orbital 2 -> '0100'  (q2 = 1)
#   orbital 3 -> '1000'  (q3 = 1)
_SINGLE_PARTICLE_INDEX = [1, 2, 4, 8]  # decimal indices of |0001>, |0010>, |0100>, |1000>


def _pauli_string(pattern: dict[int, str], n_qubits: int = 4) -> str:
    """Build an n_qubits-character Pauli string in Qiskit big-endian order.

    `pattern[q]` is the Pauli letter on qubit q in {'I','X','Y','Z'}.
    Missing entries default to 'I'.

    The returned string has qubit (n_qubits-1) at index 0 and qubit 0
    at index (n_qubits-1) — i.e. SparsePauliOp's expected convention.
    """
    letters = [pattern.get(q, "I") for q in range(n_qubits)]
    # Reverse so the leftmost character is the highest-index qubit
    return "".join(reversed(letters))


def build_jw_hamiltonian(channel: str) -> SparsePauliOp:
    """Build the 4-qubit Jordan-Wigner Pauli Hamiltonian for a channel.

    Uses the Woloshyn paper matrix for the channel as the physical 4x4
    Hamiltonian, then applies the JW transformation to produce 17 Pauli
    terms (1 identity + 4 single-Z + 6 XX+YY pairs).

    Args:
        channel: One of "1S0", "3S1", "1P1".

    Returns:
        SparsePauliOp on 4 qubits with up to 17 terms.
    """
    h = get_woloshyn_matrix(channel)  # 4x4 real symmetric, fm^-1
    return matrix_to_jw_pauli(h)


def matrix_to_jw_pauli(h: np.ndarray) -> SparsePauliOp:
    """JW-encode a 4x4 real symmetric one-body Hamiltonian on 4 qubits.

    Implements

        H_hat = sum_{m,n} h_{mn} a_m^dagger a_n
              = sum_n h_{nn} (I - Z_n)/2
              + sum_{m<n} h_{mn} (a_m^dagger a_n + h.c.)

    with the Jordan-Wigner forms

        a_n^dagger a_n              = (I - Z_n) / 2
        a_n^dagger a_{n+1} + h.c.   = (X_n X_{n+1} + Y_n Y_{n+1}) / 2
        a_n^dagger a_{n+2} + h.c.   = (X_n Z_{n+1} X_{n+2} + Y_n Z_{n+1} Y_{n+2}) / 2
        a_0^dagger a_3 + h.c.       = (X_0 Z_1 Z_2 X_3 + Y_0 Z_1 Z_2 Y_3) / 2.
    """
    assert h.shape == (4, 4)

    terms: list[tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Identity offset: (1/2) (h_00 + h_11 + h_22 + h_33) * I
    # ------------------------------------------------------------------
    c_I = 0.5 * (h[0, 0] + h[1, 1] + h[2, 2] + h[3, 3])
    terms.append((_pauli_string({}), c_I))

    # ------------------------------------------------------------------
    # Single-Z terms: -h_{nn}/2 * Z_n  (from -h_{nn}/2 * Z_n in (I - Z_n)/2)
    # ------------------------------------------------------------------
    for n in range(4):
        c = -0.5 * h[n, n]
        terms.append((_pauli_string({n: "Z"}), c))

    # ------------------------------------------------------------------
    # Off-diagonal hops -- h_{mn}(a_m^dagger a_n + h.c.)
    # Real symmetric => one coeff per unordered pair (m<n)
    # ------------------------------------------------------------------
    # Adjacent: |m-n|=1
    for n in range(3):  # pairs (0,1), (1,2), (2,3)
        c = 0.5 * h[n, n + 1]
        terms.append((_pauli_string({n: "X", n + 1: "X"}), c))
        terms.append((_pauli_string({n: "Y", n + 1: "Y"}), c))

    # Skip-1: |m-n|=2 -- with a Z on the in-between qubit
    for n in range(2):  # pairs (0,2), (1,3)
        c = 0.5 * h[n, n + 2]
        terms.append((_pauli_string({n: "X", n + 1: "Z", n + 2: "X"}), c))
        terms.append((_pauli_string({n: "Y", n + 1: "Z", n + 2: "Y"}), c))

    # Skip-2: (0,3) -- Z string on qubits 1, 2
    c = 0.5 * h[0, 3]
    terms.append((_pauli_string({0: "X", 1: "Z", 2: "Z", 3: "X"}), c))
    terms.append((_pauli_string({0: "Y", 1: "Z", 2: "Z", 3: "Y"}), c))

    # Drop numerically negligible terms
    filtered = [(p, float(c)) for p, c in terms if abs(c) > 1e-15]
    if not filtered:
        filtered = [(_pauli_string({}), 0.0)]

    return SparsePauliOp.from_list(filtered)


def validate_jw_hamiltonian(channel: str, atol: float = 1e-10) -> None:
    """Verify that the JW Pauli Hamiltonian restricted to the
    single-particle sector matches the original physical 4x4 matrix.

    Raises AssertionError on mismatch.
    """
    h = get_woloshyn_matrix(channel)
    pauli_h = build_jw_hamiltonian(channel)
    full = np.array(pauli_h.to_matrix())

    idx = _SINGLE_PARTICLE_INDEX
    projected = np.array([[full[i, j] for j in idx] for i in idx]).real
    if not np.allclose(projected, h, atol=atol):
        diff = float(np.max(np.abs(projected - h)))
        raise AssertionError(
            f"JW Hamiltonian for {channel} fails the 1-particle restriction "
            f"check (max diff {diff:.2e})."
        )
