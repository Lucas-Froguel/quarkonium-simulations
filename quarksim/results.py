"""Simulation result storage and loading.

Provides a SimulationRecord dataclass for persisting results to JSON,
and utilities for loading and comparing results across methods.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class SimulationRecord:
    """Persistent record of a simulation run."""

    paper: str
    method: str
    ground_state_energy: float
    parameters: list[float]
    wavefunction_amplitudes: list[float]
    energy_levels: list[float] = field(default_factory=list)
    exact_energies: list[float] = field(default_factory=list)
    convergence: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return _make_serializable(obj.tolist())
    if isinstance(obj, complex):
        return float(obj.real)
    if isinstance(obj, (np.complexfloating,)):
        return float(obj.real)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def save_record(record: SimulationRecord, path: str | Path):
    """Save a simulation record to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _make_serializable(asdict(record))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_record(path: str | Path) -> SimulationRecord:
    """Load a simulation record from JSON."""
    with open(path) as f:
        data = json.load(f)
    return SimulationRecord(**data)


def load_all_records(directory: str | Path) -> list[SimulationRecord]:
    """Load all simulation records from a directory."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(load_record(path))
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def comparison_table(records: list[SimulationRecord]) -> str:
    """Format a comparison table of simulation results.

    Returns a string table comparing ground state energies across methods.
    """
    lines = []
    header = f"{'Paper':<15} {'Method':<25} {'E_ground (MeV)':>15} {'E_exact (MeV)':>15} {'Error (MeV)':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in records:
        exact = r.exact_energies[0] if r.exact_energies else float("nan")
        error = r.ground_state_energy - exact if r.exact_energies else float("nan")
        lines.append(
            f"{r.paper:<15} {r.method:<25} {r.ground_state_energy:>15.2f} {exact:>15.2f} {error:>12.2f}"
        )

    return "\n".join(lines)
