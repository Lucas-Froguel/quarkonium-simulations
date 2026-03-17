"""Hamiltonians for the TTITE paper: H2 molecule and Heisenberg spin chain.

H2 molecule:
    H = c0*I + c1*Z1 + c2*Z2 + c3*Z1Z2 + c4*X1X2   (Eq. 12)
    Coefficients from Kandala et al., Nature 549 (2017), STO-3G basis,
    Jordan-Wigner transformation, reduced to 2 qubits.

Heisenberg spin-1/2 chain:
    H = -J * sum(XX + YY + ZZ)_{j,j+1} - h * sum(Z_j)   (Eq. 13)
    Open boundary conditions.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# H2 molecule coefficients (STO-3G, Jordan-Wigner, 2-qubit reduction)
# From Kandala et al., Nature 549, 242 (2017), supplementary Table S1.
# Columns: D (Angstrom), c0, c1, c2, c3, c4
# ---------------------------------------------------------------------------

_H2_TABLE = np.array([
    [0.20,  2.01152820, -0.93048853, -0.93048853,  0.01362387, 0.15797271],
    [0.25,  1.42282789, -0.87064596, -0.87064596,  0.01346349, 0.15927658],
    [0.30,  1.01018208, -0.80864891, -0.80864891,  0.01328798, 0.16081852],
    [0.35,  0.70127311, -0.74741582, -0.74741582,  0.01310364, 0.16257322],
    [0.40,  0.46036350, -0.68881943, -0.68881943,  0.01291397, 0.16451542],
    [0.45,  0.26754722, -0.63388978, -0.63388978,  0.01271920, 0.16662140],
    [0.50,  0.11064654, -0.58307963, -0.58307963,  0.01251643, 0.16887023],
    [0.55, -0.01837352, -0.53648878, -0.53648878,  0.01230035, 0.17124452],
    [0.60, -0.12516506, -0.49401379, -0.49401379,  0.01206439, 0.17373064],
    [0.65, -0.21393163, -0.45543342, -0.45543342,  0.01180192, 0.17631845],
    [0.70, -0.28794508, -0.42045568, -0.42045568,  0.01150740, 0.17900058],
    [0.75, -0.34983342, -0.38874759, -0.38874759,  0.01117714, 0.18177154],
    [0.80, -0.40174128, -0.35995942, -0.35995942,  0.01080973, 0.18462678],
    [0.85, -0.44542363, -0.33374649, -0.33374649,  0.01040607, 0.18756185],
    [0.90, -0.48230859, -0.30978728, -0.30978728,  0.00996911, 0.19057169],
    [0.95, -0.51354842, -0.28779599, -0.28779599,  0.00950347, 0.19365032],
    [1.00, -0.54006628, -0.26752865, -0.26752865,  0.00901493, 0.19679058],
    [1.05, -0.56260011, -0.24878329, -0.24878329,  0.00850994, 0.19998427],
    [1.10, -0.58174230, -0.23139588, -0.23139588,  0.00799518, 0.20322223],
    [1.15, -0.59797347, -0.21523394, -0.21523394,  0.00747720, 0.20649467],
    [1.20, -0.61168972, -0.20018958, -0.20018958,  0.00696216, 0.20979147],
    [1.25, -0.62322320, -0.18617310, -0.18617310,  0.00645559, 0.21310240],
    [1.30, -0.63285720, -0.17310785, -0.17310785,  0.00596229, 0.21641746],
    [1.35, -0.64083661, -0.16092639, -0.16092639,  0.00548622, 0.21972704],
    [1.40, -0.64737531, -0.14956794, -0.14956794,  0.00503054, 0.22302209],
    [1.45, -0.65266120, -0.13897678, -0.13897678,  0.00459759, 0.22629426],
    [1.50, -0.65685989, -0.12910131, -0.12910131,  0.00418896, 0.22953594],
    [1.55, -0.66011746, -0.11989354, -0.11989354,  0.00380558, 0.23274029],
    [1.60, -0.66256274, -0.11130875, -0.11130875,  0.00344779, 0.23590129],
    [1.65, -0.66430918, -0.10330533, -0.10330533,  0.00311546, 0.23901365],
    [1.70, -0.66545652, -0.09584459, -0.09584459,  0.00280807, 0.24207284],
    [1.75, -0.66609231, -0.08889055, -0.08889055,  0.00252482, 0.24507502],
    [1.80, -0.66629324, -0.08240979, -0.08240979,  0.00226467, 0.24801699],
    [1.85, -0.66612638, -0.07637120, -0.07637120,  0.00202649, 0.25089615],
    [1.90, -0.66565029, -0.07074579, -0.07074579,  0.00180902, 0.25371043],
    [1.95, -0.66491596, -0.06550650, -0.06550650,  0.00161098, 0.25645825],
    [2.00, -0.66396774, -0.06062801, -0.06062801,  0.00143110, 0.25913847],
    [2.05, -0.66284410, -0.05608661, -0.05608661,  0.00126812, 0.26175037],
    [2.10, -0.66157832, -0.05186007, -0.05186007,  0.00112080, 0.26429357],
    [2.15, -0.66019913, -0.04792750, -0.04792750,  0.00098799, 0.26676799],
    [2.20, -0.65873122, -0.04426934, -0.04426934,  0.00086855, 0.26917386],
    [2.25, -0.65719580, -0.04086723, -0.04086723,  0.00076144, 0.27151165],
    [2.30, -0.65561095, -0.03770401, -0.03770401,  0.00066564, 0.27378205],
    [2.35, -0.65399206, -0.03476361, -0.03476361,  0.00058020, 0.27598597],
    [2.40, -0.65235219, -0.03203106, -0.03203106,  0.00050424, 0.27812444],
    [2.45, -0.65070231, -0.02949241, -0.02949241,  0.00043691, 0.28019869],
    [2.50, -0.64905162, -0.02713470, -0.02713470,  0.00037742, 0.28221005],
])


def h2_coefficients(D: float) -> tuple[float, float, float, float, float]:
    """Return (c0, c1, c2, c3, c4) for the H2 Hamiltonian at distance D (Angstrom).

    Interpolates from the STO-3G / Jordan-Wigner coefficient table.
    """
    distances = _H2_TABLE[:, 0]
    if D < distances[0] or D > distances[-1]:
        raise ValueError(
            f"D={D} out of range [{distances[0]}, {distances[-1]}] Angstrom"
        )
    c0 = np.interp(D, distances, _H2_TABLE[:, 1])
    c1 = np.interp(D, distances, _H2_TABLE[:, 2])
    c2 = np.interp(D, distances, _H2_TABLE[:, 3])
    c3 = np.interp(D, distances, _H2_TABLE[:, 4])
    c4 = np.interp(D, distances, _H2_TABLE[:, 5])
    return float(c0), float(c1), float(c2), float(c3), float(c4)


def build_h2_hamiltonian(D: float = 0.35) -> SparsePauliOp:
    """Build the 2-qubit H2 Hamiltonian (Eq. 12).

    H = c0*II + c1*ZI + c2*IZ + c3*ZZ + c4*XX

    Convention: in Qiskit SparsePauliOp, 'AB' means A on qubit 1, B on qubit 0.
    sigma_z^1 in the paper = Z on qubit 0 = 'IZ'... but the paper uses
    sigma^1 for the first qubit. We follow Qiskit convention:
      sigma_z^1 -> 'ZI' (Z on qubit 1)
      sigma_z^2 -> 'IZ' (Z on qubit 0)
      sigma_z^1 sigma_z^2 -> 'ZZ'
      sigma_x^1 sigma_x^2 -> 'XX'
    """
    c0, c1, c2, c3, c4 = h2_coefficients(D)
    return SparsePauliOp.from_list([
        ("II", c0),
        ("ZI", c1),
        ("IZ", c2),
        ("ZZ", c3),
        ("XX", c4),
    ])


def build_heisenberg_hamiltonian(
    n: int, J: float = 1.0, h: float = 0.23
) -> SparsePauliOp:
    """Build the Heisenberg spin-1/2 chain Hamiltonian (Eq. 13).

    H = -J * sum_{j=1}^{n-1} (X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1})
        - h * sum_{j=1}^{n} Z_j

    Open boundary conditions. Uses 0-based qubit indexing internally.

    Args:
        n: Number of qubits (spins).
        J: Coupling strength.
        h: Static magnetic field.
    """
    terms = []

    # Interaction terms: -J * (XX + YY + ZZ) for each pair (j, j+1)
    for j in range(n - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            label = ["I"] * n
            label[n - 1 - j] = pauli[0]       # qubit j
            label[n - 1 - (j + 1)] = pauli[1]  # qubit j+1
            terms.append(("".join(label), -J))

    # Field terms: -h * Z_j for each qubit
    for j in range(n):
        label = ["I"] * n
        label[n - 1 - j] = "Z"
        terms.append(("".join(label), -h))

    return SparsePauliOp.from_list(terms).simplify()


def decompose_to_pauli_terms(
    hamiltonian: SparsePauliOp,
) -> list[tuple[float, str, np.ndarray]]:
    """Decompose a Hamiltonian into individual Pauli terms.

    Returns list of (coefficient, pauli_label, pauli_matrix) triples.
    Identity terms are included but can be handled specially by the caller.
    """
    n_qubits = hamiltonian.num_qubits
    dim = 2**n_qubits
    terms = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        # Build the Pauli matrix
        mat = SparsePauliOp.from_list([(label, 1.0)]).to_matrix()
        terms.append((float(coeff.real), label, np.array(mat)))
    return terms
