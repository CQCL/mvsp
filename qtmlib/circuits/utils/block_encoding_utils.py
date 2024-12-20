"""Utils for block encodings."""

from pytket.circuit import Qubit
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import QubitPauliOperator

from qtmlib._types._core_types import CoeffType


def generate_diagonal_block_encoding(
    n_qubits: int,
) -> QubitPauliOperator:
    """Generate a block encoding of a diagonal operator.

    The diagonal of the block encoded matrix defines a uniform disretisation of the
    interval [-1,1] into 2^n_qubits points.

    Args:
    ----
        n_qubits (int): Number of qubits.

    Returns:
    -------
        QubitPauliOperator: QubitPauliOperator of the block encoding.

    """
    ham_dict: dict[QubitPauliString, CoeffType] = {}
    for j in range(n_qubits - 1, -1, -1):
        pauli_string = [Pauli.I] * n_qubits
        pauli_string[n_qubits - 1 - j] = Pauli.Z
        ham_dict[
            QubitPauliString([Qubit(i) for i in range(n_qubits)], pauli_string)
        ] = -(2.0**j) / (2.0**n_qubits - 1)
    return QubitPauliOperator(ham_dict)
