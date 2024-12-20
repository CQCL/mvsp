"""Contains a function for generating the Ising model Hamiltonian."""

from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit


def ising_model(n_qubits: int, h: float, j: float) -> QubitPauliOperator:
    """Return a QubitPauliOperator representing the Ising model Hamiltonian.

    Args:
    ----
        n_qubits (int): The number of qubits in the system.
        h (float): The transverse field strength.
        j (float): The coupling strength.

    Returns:
    -------
        QubitPauliOperator representing the Ising model Hamiltonian

    """
    qpo = QubitPauliOperator({})
    for i in range(n_qubits - 1):
        qpo = qpo + QubitPauliOperator(
            {QubitPauliString([Qubit(i), Qubit(i + 1)], [Pauli.Z, Pauli.Z]): j}
        )
    for i in range(n_qubits):
        qpo = qpo + QubitPauliOperator({QubitPauliString([Qubit(i)], [Pauli.X]): h})
    return qpo
