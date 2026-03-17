"""VQE simulation runner and exact diagonalization.

Provides VQE optimization using:
  - Exact statevector simulation (noiseless, fast)
  - Shot-based simulation with optional depolarizing noise (realistic)
Plus exact diagonalization for validation.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize, minimize_scalar, curve_fit
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


@dataclass
class VQEResult:
    """Result of a VQE simulation."""

    energy: float
    parameters: np.ndarray
    wavefunction: np.ndarray
    convergence: list[float] = field(default_factory=list)
    num_evaluations: int = 0
    optimizer_message: str = ""
    metadata: dict = field(default_factory=dict)


def exact_diagonalization(hamiltonian: SparsePauliOp) -> tuple[np.ndarray, np.ndarray]:
    """Exact diagonalization of a Pauli Hamiltonian.

    Args:
        hamiltonian: Hamiltonian as SparsePauliOp.

    Returns:
        (eigenvalues, eigenvectors) sorted by ascending energy.
        eigenvalues has shape (2^n,), eigenvectors has shape (2^n, 2^n)
        where column j is the eigenvector for eigenvalue j.
    """
    matrix = hamiltonian.to_matrix().real
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def physical_eigenvalues(
    hamiltonian: SparsePauliOp, n_orbitals: int = 3
) -> np.ndarray:
    """Extract eigenvalues from the 1-particle subspace.

    The full 2^n Hilbert space includes unphysical states (0 or >1 particles).
    This projects onto the 1-particle sector and diagonalizes there.

    Args:
        hamiltonian: Pauli Hamiltonian.
        n_orbitals: Number of orbitals (qubits).

    Returns:
        Eigenvalues in the 1-particle subspace, sorted ascending.
    """
    full_matrix = hamiltonian.to_matrix()

    # 1-particle basis states: |001>, |010>, |100> (one qubit set to 1)
    indices = [1 << i for i in range(n_orbitals)]
    submatrix = np.array(
        [[full_matrix[i, j] for j in indices] for i in indices]
    ).real

    eigenvalues = np.linalg.eigvalsh(submatrix)
    return eigenvalues


def run_vqe(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    x0: np.ndarray | None = None,
    method: str = "cobyla",
    maxiter: int = 300,
) -> VQEResult:
    """Run VQE optimization with exact statevector simulation.

    Args:
        hamiltonian: Pauli Hamiltonian to minimize.
        ansatz: Parameterized quantum circuit.
        x0: Initial parameter values. Random in [0, pi) if None.
        method: Scipy minimizer method.
        maxiter: Maximum optimizer iterations.

    Returns:
        VQEResult with optimized energy, parameters, and convergence history.
    """
    n_params = ansatz.num_parameters
    if x0 is None:
        rng = np.random.default_rng()
        x0 = rng.uniform(0, np.pi, size=n_params)

    energies: list[float] = []

    def cost_fn(params):
        bound = ansatz.assign_parameters(params)
        sv = Statevector(bound)
        energy = sv.expectation_value(hamiltonian).real
        energies.append(energy)
        return energy

    result = minimize(cost_fn, x0, method=method, options={"maxiter": maxiter})

    # Final wavefunction
    bound = ansatz.assign_parameters(result.x)
    sv = Statevector(bound)

    return VQEResult(
        energy=result.fun,
        parameters=result.x,
        wavefunction=np.array(sv),
        convergence=energies,
        num_evaluations=result.nfev,
        optimizer_message=str(result.message),
    )


def run_vqe_excited(
    hamiltonian: SparsePauliOp,
    state_builder: Callable[[float], np.ndarray],
    method: str = "bounded",
    bounds: tuple[float, float] = (0, 2 * np.pi),
) -> VQEResult:
    """Run VQE for an excited state over a 1D parameter space.

    This is used with orthogonalization-based excited state methods where
    the ansatz is constrained to a 1-parameter family of states orthogonal
    to lower-lying states.

    Args:
        hamiltonian: Pauli Hamiltonian.
        state_builder: Function gamma -> 8-component statevector.
        method: Scipy scalar minimizer method.
        bounds: Search bounds for the parameter.

    Returns:
        VQEResult with optimized energy and parameter.
    """
    energies: list[float] = []

    def cost_fn(gamma):
        sv = Statevector(state_builder(gamma))
        energy = sv.expectation_value(hamiltonian).real
        energies.append(energy)
        return energy

    result = minimize_scalar(cost_fn, bounds=bounds, method=method)

    final_sv = state_builder(result.x)

    return VQEResult(
        energy=result.fun,
        parameters=np.array([result.x]),
        wavefunction=final_sv,
        convergence=energies,
        num_evaluations=result.nfev,
        optimizer_message="converged" if result.success else str(result.message),
    )


# ---------------------------------------------------------------------------
# Shot-based / noisy simulation
# ---------------------------------------------------------------------------


def make_noise_model(depolarizing_rate: float):
    """Create a simple depolarizing noise model.

    Adds depolarizing error to every gate:
      - 1-qubit gates (ry, x, h, sdg): error rate = depolarizing_rate
      - 2-qubit gates (cx): error rate = depolarizing_rate (applied to 2-qubit space)

    This is the same global depolarizing channel model used in the paper
    (Section II.E). On real hardware, each gate layer has a success rate r_i,
    and the total noise scales as r^lambda.

    Args:
        depolarizing_rate: Error probability per gate (0 = noiseless, 1 = fully mixed).
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()
    error_1q = depolarizing_error(depolarizing_rate, 1)
    error_2q = depolarizing_error(depolarizing_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_1q, ["ry", "x", "h", "sdg"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
    return noise_model


def _measure_pauli_term(
    bound_ansatz: QuantumCircuit,
    pauli_label: str,
    backend,
    shots: int,
) -> float:
    """Measure the expectation value of a single Pauli string.

    Appends basis-rotation gates (H for X, S†H for Y) then measures
    in the computational basis. Returns <P> from the shot statistics.
    """
    n_qubits = bound_ansatz.num_qubits

    # Identity: <I⊗I⊗...> = 1 always
    if all(p == "I" for p in pauli_label):
        return 1.0

    circ = bound_ansatz.copy()

    # Basis rotations: Qiskit Pauli string 'ABC' → A=qubit n-1, ..., C=qubit 0
    for i, pauli_char in enumerate(reversed(pauli_label)):  # i = qubit index
        if pauli_char == "X":
            circ.h(i)
        elif pauli_char == "Y":
            circ.sdg(i)
            circ.h(i)
        # Z and I: measure directly in computational basis

    circ.measure_all()
    # optimization_level=0: do NOT simplify the circuit.
    # This is critical for ZNE — folded circuits (U·U†·U) must NOT be
    # collapsed back to U, or the noise amplification won't work.
    transpiled = transpile(circ, backend, optimization_level=0)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # <P> = sum_s (-1)^{parity of active bits} * count(s) / shots
    exp_val = 0.0
    for bitstring, count in counts.items():
        # bitstring: '010' where leftmost = highest qubit
        parity = 0
        for j, pauli_char in enumerate(pauli_label):
            if pauli_char != "I":
                parity += int(bitstring[j])
        exp_val += (-1) ** (parity % 2) * count / shots

    return exp_val


def _estimate_energy_shot_based(
    bound_ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    backend,
    shots: int,
) -> float:
    """Estimate <H> by measuring each Pauli term separately with finite shots."""
    energy = 0.0
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        exp_val = _measure_pauli_term(bound_ansatz, label, backend, shots)
        energy += coeff.real * exp_val
    return energy


def run_vqe_noisy(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    shots: int = 8192,
    depolarizing_rate: float = 0.0,
    x0: np.ndarray | None = None,
    method: str = "cobyla",
    maxiter: int = 200,
) -> VQEResult:
    """Run VQE with shot-based measurement and optional depolarizing noise.

    Each cost function evaluation measures every Pauli term in the
    Hamiltonian separately using `shots` samples per term.

    Args:
        hamiltonian: Pauli Hamiltonian to minimize.
        ansatz: Parameterized quantum circuit.
        shots: Number of measurement shots per Pauli term.
        depolarizing_rate: Depolarizing error rate per gate (0 = shot noise only).
        x0: Initial parameter values. Random in [0, pi) if None.
        method: Scipy minimizer method.
        maxiter: Maximum optimizer iterations.

    Returns:
        VQEResult with optimized energy and convergence history.
    """
    from qiskit_aer import AerSimulator

    if depolarizing_rate > 0:
        noise_model = make_noise_model(depolarizing_rate)
        backend = AerSimulator(noise_model=noise_model)
    else:
        backend = AerSimulator()

    n_params = ansatz.num_parameters
    if x0 is None:
        rng = np.random.default_rng()
        x0 = rng.uniform(0, np.pi, size=n_params)

    energies: list[float] = []

    def cost_fn(params):
        bound = ansatz.assign_parameters(params)
        energy = _estimate_energy_shot_based(bound, hamiltonian, backend, shots)
        energies.append(energy)
        return energy

    result = minimize(cost_fn, x0, method=method, options={"maxiter": maxiter})

    # Get final wavefunction from noiseless statevector (for analysis)
    bound = ansatz.assign_parameters(result.x)
    sv = Statevector(bound)

    return VQEResult(
        energy=result.fun,
        parameters=result.x,
        wavefunction=np.array(sv),
        convergence=energies,
        num_evaluations=result.nfev,
        optimizer_message=str(result.message),
        metadata={
            "shots": shots,
            "depolarizing_rate": depolarizing_rate,
            "backend": "aer_simulator",
        },
    )


# ---------------------------------------------------------------------------
# Zero-noise extrapolation (Section II.E of Gallimore & Liao)
# ---------------------------------------------------------------------------


@dataclass
class ZNEResult:
    """Result of zero-noise extrapolation."""

    energy: float
    energy_std: float
    raw_energies: dict[int, float]  # {scale_factor: energy}
    per_term: dict[str, dict]       # {pauli_label: {lambdas, values, fit_a, fit_r}}
    scale_factors: list[int]
    params_used: np.ndarray


def _fold_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    """Global unitary folding: U -> U·(U†·U)^n for scale_factor = 2n+1.

    The folded circuit is logically equivalent to the original but has
    (2n+1)x the depth, causing proportionally more noise on real hardware.

    This is the standard approach from Giurgica-Tiron et al. (2020),
    cited in the paper as Ref. [31].

    Args:
        circuit: Bound (no free parameters) quantum circuit without measurements.
        scale_factor: Odd positive integer (1, 3, 5, 7, ...).

    Returns:
        Folded circuit with depth = scale_factor * original_depth.
    """
    if scale_factor < 1 or scale_factor % 2 == 0:
        raise ValueError(f"scale_factor must be odd positive integer, got {scale_factor}")
    if scale_factor == 1:
        return circuit.copy()

    n_folds = (scale_factor - 1) // 2
    folded = circuit.copy()
    for _ in range(n_folds):
        folded = folded.compose(circuit.inverse())
        folded = folded.compose(circuit)
    return folded


def _measure_all_terms(
    bound_ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    backend,
    shots: int,
) -> dict[str, float]:
    """Measure expectation value of every Pauli term. Returns {label: <P>}."""
    results = {}
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        results[label] = _measure_pauli_term(bound_ansatz, label, backend, shots)
    return results


def run_zne(
    hamiltonian: SparsePauliOp,
    ansatz: QuantumCircuit,
    params: np.ndarray,
    shots: int = 16384,
    depolarizing_rate: float = 0.01,
    scale_factors: list[int] | None = None,
) -> ZNEResult:
    """Run zero-noise extrapolation (Section II.E of the paper).

    After VQE finds optimal parameters at base noise (lambda=1), ZNE:
    1. Folds the circuit at increasing noise levels (lambda = 1, 3, 5, 7)
    2. Measures each Pauli term at each noise level
    3. Fits exponential decay <P>(lambda) = A * r^lambda per term
    4. Extrapolates to lambda=0 to recover the noiseless value

    Args:
        hamiltonian: Pauli Hamiltonian.
        ansatz: Parameterized ansatz circuit (unbound).
        params: Optimized VQE parameters.
        shots: Shots per Pauli term per noise level.
        depolarizing_rate: Base depolarizing error rate per gate.
        scale_factors: Noise scale factors (odd integers). Default [1, 3, 5, 7].

    Returns:
        ZNEResult with extrapolated energy and per-term fit data.
    """
    from qiskit_aer import AerSimulator

    if scale_factors is None:
        scale_factors = [1, 3, 5, 7]

    noise_model = make_noise_model(depolarizing_rate)
    backend = AerSimulator(noise_model=noise_model)

    bound = ansatz.assign_parameters(params)

    # --- Step 1: Measure each Pauli term at each noise level ---
    # all_data[label][lambda] = <P>(lambda)
    all_data: dict[str, dict[int, float]] = {}
    raw_energies: dict[int, float] = {}

    for lam in scale_factors:
        print(f"    Measuring at lambda={lam} (circuit depth x{lam})...")
        folded = _fold_circuit(bound, lam)
        term_values = _measure_all_terms(folded, hamiltonian, backend, shots)

        for label, val in term_values.items():
            if label not in all_data:
                all_data[label] = {}
            all_data[label][lam] = val

        # Raw energy at this scale factor
        energy = sum(
            coeff.real * term_values[pauli.to_label()]
            for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs)
        )
        raw_energies[lam] = energy

    # --- Step 2: Fit exponential per traceless term and extrapolate ---
    lambdas = np.array(scale_factors, dtype=float)
    per_term: dict[str, dict] = {}
    extrapolated_energy = 0.0
    extrapolated_var = 0.0

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = pauli.to_label()
        values = np.array([all_data[label][lam] for lam in scale_factors])

        if all(p == "I" for p in label):
            # Identity: always 1, no noise correction needed
            per_term[label] = {"extrapolated": 1.0, "std": 0.0, "values": values.tolist()}
            extrapolated_energy += coeff.real * 1.0
            continue

        # Fit y(lambda) = a * r^lambda + c
        # The constant c accounts for the noise floor from measurement
        # basis-rotation gates (H for X, S†H for Y), which creates a
        # bias independent of the ansatz noise level.
        # At lambda=0: y(0) = a + c  (noiseless value)
        #
        # The paper's simpler model y = a * r^lambda (no offset) fails
        # for terms with small noiseless values (XXI, YYI) where the
        # measurement bias dominates — the same issue noted in Section III.A.
        try:
            def exp_model(lam, a, r, c):
                return a * np.power(r, lam) + c

            # Initial guesses from the data trend
            a0 = values[0] - values[-1]  # decaying part
            r0 = 0.8
            c0 = values[-1]              # asymptotic offset
            popt, pcov = curve_fit(
                exp_model, lambdas, values,
                p0=[a0, r0, c0],
                bounds=([-np.inf, 0.01, -np.inf], [np.inf, 0.9999, np.inf]),
                maxfev=5000,
            )
            a_fit, r_fit, c_fit = popt
            extrapolated_val = a_fit + c_fit  # y(0) = a + c
            # Propagate uncertainty: var(a+c) = var(a) + var(c) + 2*cov(a,c)
            a_var = max(pcov[0, 0], 0)
            c_var = max(pcov[2, 2], 0)
            ac_cov = pcov[0, 2]
            extrap_std = np.sqrt(a_var + c_var + 2 * ac_cov) if (a_var + c_var + 2 * ac_cov) >= 0 else 0.0

            per_term[label] = {
                "extrapolated": float(extrapolated_val),
                "std": float(extrap_std),
                "r_fit": float(r_fit),
                "c_fit": float(c_fit),
                "values": values.tolist(),
            }
            extrapolated_energy += coeff.real * extrapolated_val
            extrapolated_var += (coeff.real * extrap_std) ** 2

        except (RuntimeError, ValueError):
            # Fit failed — fall back to lambda=1 value
            per_term[label] = {
                "extrapolated": float(values[0]),
                "std": float(np.std(values)),
                "fit_failed": True,
                "values": values.tolist(),
            }
            extrapolated_energy += coeff.real * values[0]

    return ZNEResult(
        energy=float(extrapolated_energy),
        energy_std=float(np.sqrt(extrapolated_var)),
        raw_energies=raw_energies,
        per_term=per_term,
        scale_factors=scale_factors,
        params_used=params,
    )
