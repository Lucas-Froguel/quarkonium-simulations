from qiskit import *
from qiskit.quantum_info import *
from qiskit.visualization import *
import numpy as np


qb, cb = 3, 3
shots = 4096
alpha = 0.2
beta = 0.3
terms = {
    "0": {
        "op": "",
        "val": 1.0
    },
    "1": {
        "op": "circ.z(0)",
        "val": 1.0
    },
    "2": {
        "op": "circ.z(1)",
        "val": 1.0
    },
    "3": {
        "op": "circ.z(2)",
        "val": 1.0
    },
    "4": {
        "op": "circ.x(0), circ.x(1)",
        "val": 1.0
    },
    "5": {
        "op": "circ.x(1), circ.x(2)",
        "val": 1.0
    },
    "6": {
        "op": "circ.y(0), circ.y(1)",
        "val": 1.0
    },
    "7": {
        "op": "circ.y(1), circ.y(2)",
        "val": 1.0
    },
    "8": {
        "op": "circ.x(0), circ.z(1), circ.x(2)",
        "val": 1.0
    },
    "9": {
        "op": "circ.y(0), circ.z(1), circ.y(2)",
        "val": 1.0
    }
}

def apply_ansatz(circ, alpha: float = 0, beta: float = 0):
    circ.ry(beta, 1)
    circ.ry(2 * alpha, 2)
    circ.cx(2, 0)
    circ.cx(0, 1)
    circ.x(2)
    circ.ry( -beta, 1)
    circ.cx(0, 1)
    circ.cx(1, 0)

def measure(circ, num_qbits: int = 0):
    for i in range(num_qbits):
        circ.measure(i, i)

def calculate_mean_value(probs: dict, alpha: int = None, beta: int = None) -> float:
    exp = 0.0
    ansatz_ops = {
        "100": np.cos(alpha),
        "010": np.sin(alpha) * np.sin(beta),
        "001": np.sin(alpha) * np.cos(beta)
    }
    
    # implements <A> = \sum_k a_k sqrt(p_k)
    for key, prob in probs.items():
        if key in ansatz_ops:
            exp += ansatz_ops[key] * np.sqrt(prob)

    return exp

def get_probabilities(circ, shots: int = shots):
    counts = execute(circ, Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
    probs = { _id: val / shots for _id, val in counts.items() }
    return probs

def hamiltonian_terms(index, circ, terms):
    if terms[str(index)]["op"]:
        eval(terms[str(index)]["op"])

def calculate_full_hamiltonian(shots: int = None, alpha: int = None, beta: int = None, terms: dict = None):
    H = 0
    for index in range(10):
        circ = QuantumCircuit(qb, cb)
        apply_ansatz(circ, alpha=alpha, beta=beta)
        hamiltonian_terms(index, circ, terms)
        measure(circ, num_qbits=3)
        probs = get_probabilities(circ, shots=shots)
        exp_val = calculate_mean_value(probs, alpha=alpha, beta=beta)
        
        H += exp_val * terms[str(index)]["val"]

        print(f"Results for index {index}:\n - probs: {probs}\n - exp_val: {exp_val}")

    return H

H = calculate_full_hamiltonian(shots=shots, alpha=alpha, beta=beta, terms=terms)

print(f"H={H}")
