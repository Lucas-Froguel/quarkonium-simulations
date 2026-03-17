# Quarkonium Simulations

Quantum computing simulations for heavy quarkonium spectroscopy, reproducing
results from published papers using the VQE algorithm on quantum simulators.

## Installation

Requires Python 3.10+.

```bash
# Set up environment
pyenv local 3.12
uv venv && source .venv/bin/activate

# Install the package
uv pip install -e .

# (Optional) For IBM Quantum hardware support:
uv pip install -e ".[ibm]"
```

## Running Simulations

Each paper has its own subpackage under `quarksim/`. Run a simulation with:

```bash
# Gallimore & Liao (Cornell potential charmonium, VQE)
python -m quarksim.gallimore.run

# With options
python -m quarksim.gallimore.run --seed 42 --maxiter 500 --output results/gallimore
```

Results (JSON data + PNG plots) are saved to `output/<paper>/` by default.

## Project Structure

```
quarksim/                       # Python package
├── simulation.py               # VQE runner, exact diagonalization
├── visualization.py            # Convergence, energy level, wavefunction plots
├── results.py                  # Result storage (JSON) and comparison
└── gallimore/                  # Gallimore & Liao (arXiv:2202.03333)
    ├── README.md               # Overview and expected results
    ├── derivation.tex/.pdf     # Detailed paper derivation (LaTeX)
    ├── hamiltonian.py          # Cornell potential Hamiltonian
    ├── ansatz.py               # UCC ansatz circuit
    └── run.py                  # Entry point

articles/                       # Reference PDFs
output/                         # Simulation outputs (gitignored)
```

## Common Tools (`quarksim/`)

The shared tooling is designed for reuse across different papers and methods:

- **`simulation.py`** — `run_vqe()` for statevector VQE optimization,
  `exact_diagonalization()` and `physical_eigenvalues()` for reference values.
- **`visualization.py`** — `plot_convergence()`, `plot_energy_levels()`,
  `plot_wavefunction()`, `plot_method_comparison()`.
- **`results.py`** — `SimulationRecord` dataclass, `save_record()`/`load_record()`,
  `comparison_table()` for cross-method comparison.

## Adding a New Paper

1. Create `quarksim/<author>/` with `__init__.py`
2. Implement the paper's Hamiltonian and ansatz
3. Create a `run.py` that uses `quarksim.simulation.run_vqe()` and
   `quarksim.visualization` for plotting
4. Add `derivation.tex` with a detailed walkthrough
5. Save results via `quarksim.results.save_record()` for cross-paper comparison

## Papers Implemented

| Paper | arXiv | Method | System | Status |
|-------|-------|--------|--------|--------|
| Gallimore & Liao (2023) | [2202.03333](https://arxiv.org/abs/2202.03333) | VQE + UCC | Charmonium (Cornell potential) | Done |
