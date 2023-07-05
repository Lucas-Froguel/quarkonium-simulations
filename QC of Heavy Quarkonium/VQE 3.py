import qiskit
import json
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.quantum_info import *
from qiskit.visualization import *
from qiskit import execute, Aer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Session
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp


N = 3
shots = 2048
service = QiskitRuntimeService(channel="ibm_quantum", token="0269a959c178542551fbe79731ceecb4685b8fb2c26fbaa03a8450cccef257f33f346fac7f0cca39ede87c0e7cb0312df6cd73f0fc8f4ae9435058af8bffdcd4")
backend = service.get_backend("ibmq_qasm_simulator")

def get_H():
    with open("H_data.json", "r+") as file:
        H = json.load(file)

    Ham = SparsePauliOp.from_list([(Hk["operator"], Hk["coeff"]) for Hk in H.values()])

    return Ham


def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    energy = (
        estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
    )
    return energy


def build_callback(ansatz, hamiltonian, estimator, callback_dict):
    """Return callback function that uses Estimator instance,
    and stores intermediate values into a dictionary.

    Parameters:
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance
        callback_dict (dict): Mutable dict for storing values

    Returns:
        Callable: Callback function object
    """

    def callback(current_vector):
        """Callback function storing previous solution vector,
        computing the intermediate cost value, and displaying number
        of completed iterations and average time per iteration.

        Values are stored in pre-defined 'callback_dict' dictionary.

        Parameters:
            current_vector (ndarray): Current vector of parameters
                                      returned by optimizer
        """
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"] = current_vector
        # Compute the value of the cost function at the current vector
        callback_dict["cost_history"].append(
            estimator.run(ansatz, hamiltonian, parameter_values=current_vector)
            .result()
            .values[0]
        )
        # Grab the current time
        current_time = time.perf_counter()
        # Find the total time of the execute (after the 1st iteration)
        if callback_dict["iters"] > 1:
            callback_dict["_total_time"] += current_time - callback_dict["_prev_time"]
        # Set the previous time to the current time
        callback_dict["_prev_time"] = current_time
        # Compute the average time per iteration and round it
        time_str = (
            round(callback_dict["_total_time"] / (callback_dict["iters"] - 1), 2)
            if callback_dict["_total_time"]
            else "-"
        )
        # Print to screen on single line
        print(
            "Iters. done: {} [Avg. time per iter: {}]".format(
                callback_dict["iters"], time_str
            ),
            end="\r",
            flush=True,
        )

    return callback


def get_ansatz(N):
    a, b = Parameter("a"), Parameter("b")
    circ = QuantumCircuit(N)

    circ.ry(b, 1)
    circ.ry(2 * a, 2)
    circ.cx(2, 0)
    circ.cx(0, 1)
    circ.x(2)
    circ.ry( -b, 1)
    circ.cx(0, 1)
    circ.cx(1, 0)

    return circ


def save_json(data: dict, filename: str = "run_final_data.json"):
    with open(filename, "w+") as file:
        json.dump(data, file)


def save_pickle(data: dict, filename: str = "run_final_data.pickle"):
    with open(filename, "wb+") as file:
        pickle.dump(data, file)

ansatz = get_ansatz(N)
hamiltonian = get_H()
x0 = np.random.rand(2)
callback_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
    "_total_time": 0,
    "_prev_time": None,
}

with Session(backend=backend):
    estimator = Estimator(options={"shots": shots})
    callback = build_callback(ansatz, hamiltonian, estimator, callback_dict)
    res = minimize(
        cost_func,
        x0,
        args=(ansatz, hamiltonian, estimator),
        method="cobyla",
        callback=callback,
    )

save_pickle(res)
save_pickle(callback_dict, filename="callback_dict.json")
print(res)

fig, ax = plt.subplots()
ax.plot(range(callback_dict["iters"]), callback_dict["cost_history"])
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
plt.savefig("VQE - quarkonium - test 2")
plt.show()