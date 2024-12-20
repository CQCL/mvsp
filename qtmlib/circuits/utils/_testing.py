from qtmlib.circuits.core import RegisterBox, QRegMap
from numpy.typing import NDArray
import numpy as np
from qtmlib.measurement.utils import circuit_unitary_postselect, unitary_postselect
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import QubitPauliString, Pauli
from pytket.circuit import Qubit


def get_controlled_circ_u_postselect_ancilla(
    register_box: RegisterBox, rotation: float
) -> NDArray[np.complex128]:
    """Get circuit unitary for a postselected QControlRegisterBox.

    Takes in a registerbox and a rotation angle and returns a controlled
    circuit unitary for the QControlRegisterBox with the ancilla postselected to 0.
    To be used for testing the QControlRegisterBox.

    Args:
    ----
        register_box (RegisterBox): The RegisterBox.
        rotation (float): The rotation angle.

    Returns:
    -------
        NDArray[np.complex128]: The circuit unitary.

    """
    qc_box = register_box.qcontrol(1)
    circ = qc_box.initialise_circuit()
    circ.Ry(rotation, qc_box.qreg.control[0])
    circ.X(qc_box.qreg.control[0])
    qreg_map = QRegMap(qc_box.q_registers, circ.q_registers)
    circ.add_registerbox(qc_box, qreg_map)
    circ.X(qc_box.qreg.control[0])
    circ.Ry(rotation, qc_box.qreg.control[0]).dagger()
    post_select_dict = qc_box.register_box.postselect
    post_select_dict[qc_box.qreg.control[0]] = 0
    circ_u = circuit_unitary_postselect(circ, post_select_dict)
    return circ_u


def get_controlled_scipy_u(
    register_box: RegisterBox, rotation: float
) -> NDArray[np.complex128]:
    """Get scipy unitary for a postselected QControlRegisterBox.

    Takes in a registerbox and a rotation angle and returns a scipy unitary
    for the QControlRegisterBox with the ancilla postselected to 0.
    To be used for testing the QControlRegisterBox.

    Args:
    ----
        register_box (RegisterBox): The RegisterBox.
        rotation (float): The rotation angle.

    Returns:
    -------
        NDArray[np.complex128]: The scipy unitary.

    """
    scipy_h = register_box.get_unitary()
    if register_box.postselect != {}:
        scipy_h = unitary_postselect(
            register_box.qubits, scipy_h, register_box.postselect.copy()
        )
    factor = np.cos(rotation * np.pi / 2) ** 2
    scipy_u = (
        factor * scipy_h
        - (1 - factor)
        * QubitPauliOperator({QubitPauliString([Qubit(0)], [Pauli.I]): 1})
        .to_sparse_matrix(int(np.log2(scipy_h.shape[0])))
        .todense()
    )
    return scipy_u


def qcontrol_test(register_box: RegisterBox, atol: float):
    rotation: float = 0.1
    scipy_u = get_controlled_scipy_u(register_box, rotation)
    circ_u = get_controlled_circ_u_postselect_ancilla(register_box, rotation)
    np.testing.assert_allclose(circ_u, scipy_u, atol=atol)


def qft_unitary(n_qubits: int) -> NDArray[np.complex128]:
    """Return the unitary matrix for the n qubit Quantum Fourier transform."""
    dim = 2**n_qubits
    list_of_rows: list[list[np.complex128]] = []
    for u in range(dim):
        row = [
            1 / np.sqrt(dim) * np.exp(2 * np.pi * 1j * u * v / dim) for v in range(dim)
        ]
        list_of_rows.append(row)

    qft_arr = np.array(list_of_rows)

    return qft_arr
