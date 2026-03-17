# Gallimore & Liao — Quantum Computing for Heavy Quarkonium Spectroscopy

**Paper:** D. Gallimore and J. Liao, [arXiv:2202.03333v2](https://arxiv.org/abs/2202.03333) (2023)

## Overview

VQE simulation of the charmonium (c–c̄) system using the Cornell potential
V(r) = −κ/r + σr, expanded in a 3-state QHO basis and mapped to 3 qubits
via the Jordan-Wigner transformation.

## Detailed Derivation

See **[derivation.pdf](derivation.pdf)** for a full equation-by-equation walkthrough
of the paper, including the physics, matrix elements, Jordan-Wigner mapping,
UCC ansatz, VQE algorithm, and a record of bugs fixed from the original code.

To rebuild the PDF: `pdflatex derivation.tex` (requires LaTeX).

## Files

| File | Description |
|------|-------------|
| `hamiltonian.py` | Cornell potential Hamiltonian: matrix elements (Eqs. 5-7), Pauli decomposition (Eqs. 11-20) |
| `ansatz.py` | UCC ansatz circuit (Fig. 1, Eq. 23) |
| `run.py` | Entry point: exact diag + VQE optimization + plots |
| `derivation.tex` | LaTeX source for the detailed derivation |
| `derivation.pdf` | Compiled derivation document |

## Quick Start

```bash
python -m quarksim.gallimore.run --seed 42
```

## Expected Results

| Quantity | Our code | Paper |
|----------|----------|-------|
| E₀ (ground state) | 492.63 MeV | 492.6 MeV |
| E₁ (1st excited) | 1210.84 MeV | 1210.8 MeV |
| α₀ | 3.311 | 3.31 |
| β₀ | 0.955 | 0.95 |
| Wavefunction | −0.986\|0⟩ − 0.137\|1⟩ − 0.097\|2⟩ | −0.986\|0⟩ − 0.137\|1⟩ − 0.097\|2⟩ |
