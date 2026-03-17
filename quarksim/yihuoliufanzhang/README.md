# Yi, Huo, Liu, Fan, Zhang & Cao — Trotter-Taylor Imaginary-Time Evolution

**Paper:** Yi, Huo, Liu, Fan, Zhang & Cao, EPJ Quantum Technology **12**:43 (2025)

## Overview

Imaginary-time evolution (ITE) algorithm using Trotter decomposition and Taylor
expansion to prepare ground states on quantum computers. The non-unitary
imaginary-time operator is implemented via a linear combination of unitaries
(LCU) with an ancilla qubit.

Two test systems are simulated:
- **H₂ molecule** (2 qubits): STO-3G basis, Jordan-Wigner mapping
- **Heisenberg spin-1/2 chain** (3–12 qubits): open boundary conditions

## Detailed Derivation

See **[derivation.pdf](derivation.pdf)** for a full equation-by-equation walkthrough
of the paper, including ITE theory, Trotter decomposition, Taylor expansion,
LCU circuits, and code mapping.

To rebuild the PDF: `pdflatex derivation.tex` (requires LaTeX).

## Files

| File | Description |
|------|-------------|
| `hamiltonian.py` | H₂ and Heisenberg Hamiltonians as Pauli operators |
| `circuit.py` | LCU quantum circuits for TTITE steps (Algorithm 1) |
| `evolution.py` | Core TTITE time-stepping engine (matrix-based simulation) |
| `run.py` | Entry point: exact diag + TTITE evolution + figure reproduction |
| `derivation.tex` | LaTeX source for the detailed derivation |
| `derivation.pdf` | Compiled derivation document |

## Quick Start

```bash
# Default: H₂ at D=0.35, reproduce all figures
python -m quarksim.yihuoliufanzhang.run

# Specific figure
python -m quarksim.yihuoliufanzhang.run --figure 2

# Heisenberg chain
python -m quarksim.yihuoliufanzhang.run --system heisenberg --n 6
```

## Expected Results

### H₂ molecule (D = 0.35 Å)

| Quantity | Our code | Paper |
|----------|----------|-------|
| E₀ (exact) | ≈ −1.05 Ha | ≈ −1.05 Ha |
| E_τ (TTITE, τ=3) | ≈ E₀ | ≈ E₀ |
| Fidelity (τ=3) | ≈ 1.0 | ≈ 1.0 |

### Heisenberg chain (n=6, J=h=1)

| Quantity | Our code | Paper |
|----------|----------|-------|
| Normalized energy | → 0 | → 0 |
| Fidelity | → 1.0 | → 1.0 |
