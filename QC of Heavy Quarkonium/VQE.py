from qiskit import *
from qiskit.quantum_info import *
from qiskit.visualization import *


qb, cb = 3, 3
shots = 1024
circ = QuantumCircuit(qb, cb)


def apply_ansatz(circ, alpha: float = 0, beta: float = 0):
    circ.ry(beta, 1)
    circ.ry(2 * alpha, 2)
    circ.cx(2, 0)
    circ.cx(0, 1)
    circ.x(2)
    circ.ry( -beta, 1)
    circ.cx(0, 1)
    circ.cx(1, 0)


apply_ansatz(circ, alpha=0.2, beta=0.4)

#circ.z(2)
#circ.x(0)
#circ.z(1)
#circ.x(2)
circ.h(1)

def measure(circ, num_qbits: int = 0):
    for i in range(num_qbits):
        circ.measure(i, i)

measure(circ, num_qbits=3)

counts = execute(circ, Aer.get_backend('qasm_simulator'), shots=shots).result().get_counts()
print(counts)

probs = { _id: val / shots for _id, val in counts.items() }
print(probs)
plot_histogram(counts)


def calculate_mean_value(probs: dict) -> float:
    exp = 0.0

    for key, prob in probs.items():
        sign = sum([int(k) for k in key])
        exp += (-1)**sign * prob 

    return exp


exp_val = calculate_mean_value(probs)
print(exp_val)