"""Tests for circuits/lcu_state_preparation/block_encoding.py."""

import numpy as np
import pytest
from numpy.polynomial import Chebyshev

from mvsp.circuits.core import RegisterCircuit
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation import (
    LCUStatePreparationQReg,
)
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation_block_encoding import (
    ChebychevBlockEncoding,
    FourierBlockEncoding,
)
from mvsp.utils.linalg_utils import (
    get_projector_matrix,
)


@pytest.mark.parametrize(
    ("index"),
    list(range(10)),
)
def test_FourierBlockEncoding(index: int):
    """Test FourierBlockEncoding.

    Args:
    ----
        index (int): Index of the Fourier basis function.

    """
    n_qubits = 5
    W = FourierBlockEncoding(n_block_qubits=n_qubits)
    W_box = W.generate_basis_element(basis_index=index)
    unitary = W_box.get_circuit().get_unitary()

    h = unitary - np.diag(unitary.diagonal())
    assert np.abs(h).max() < 1e-12

    # Compare to exact Fourier basis functions
    x_vals = np.linspace(0, 1, len(unitary.diagonal()))
    y_vals = np.exp(index * 1j * np.pi * x_vals)

    np.testing.assert_allclose(unitary.diagonal().real, y_vals.real, atol=1e-12)
    np.testing.assert_allclose(unitary.diagonal().imag, y_vals.imag, atol=1e-12)


@pytest.mark.parametrize(
    ("power"),
    list(range(10)),
)
def test_FourierBlockEncoding_power(power: int):
    """Test FourierBlockEncoding.

    Args:
    ----
        power (int): Index of the Fourier basis function.

    """
    n_qubits = 5
    W = FourierBlockEncoding(n_block_qubits=n_qubits)
    W_box = W.generate_basis_element(basis_index=power)

    unitary = W_box.get_circuit().get_unitary()

    h = unitary - np.diag(unitary.diagonal())
    assert np.abs(h).max() < 1e-12

    # Compare to exact Fourier basis functions
    x_vals = np.linspace(0, 1, len(unitary.diagonal()))
    y_vals = np.exp(power * 1j * np.pi * x_vals)
    np.testing.assert_allclose(unitary.diagonal().real, y_vals.real, atol=1e-12)
    np.testing.assert_allclose(unitary.diagonal().imag, y_vals.imag, atol=1e-12)


@pytest.mark.parametrize(
    ("index"),
    list(range(10)),
)
def test_ChebychevBlockEncoding(index: int):
    """Test ChebychevBlockEncoding.

    Args:
    ----
        index (int): Index of the Chebychev basis function.

    """
    n_qubits = 5
    W = ChebychevBlockEncoding(n_block_qubits=n_qubits)
    W_box = W.generate_basis_element(basis_index=1)

    anc_qubits = W_box.q_registers[0]
    state_qubits = W_box.q_registers[1]

    n_anc_qubits = len(anc_qubits)
    n_state_qubits = len(state_qubits)

    W_box = W.generate_basis_element(basis_index=index)
    unitary = W_box.get_circuit().get_unitary()

    # Compare to exact matrix
    projector = np.array(
        get_projector_matrix(
            list(range(n_anc_qubits)),
            list(range(n_anc_qubits, n_anc_qubits + n_state_qubits)),
        )
    )
    unitary_projected = projector.transpose() @ unitary @ projector
    h = unitary_projected - np.diag(unitary_projected.diagonal())
    assert np.abs(h).max() < 1e-12

    # Compare to exact Chebyshev polynomial
    x_vals = np.linspace(-1, 1, len(unitary_projected.diagonal()))
    coef = [0 for _ in range(index + 1)]
    coef[-1] = 1
    coef = np.array(coef)
    T = Chebyshev(coef=coef)
    y_vals = T(x_vals)
    np.testing.assert_allclose(unitary_projected.diagonal(), y_vals, atol=1e-12)


@pytest.mark.parametrize(
    ("power"),
    list(range(10)),
)
def test_ChebychevBlockEncoding_power(power: int):
    """Test ChebychevBlockEncoding.

    Args:
    ----
        power (int): Index of the Chebychev basis function.

    """
    n_qubits = 5
    W = ChebychevBlockEncoding(n_block_qubits=n_qubits)
    W_box = W.generate_basis_element(basis_index=1)

    anc_qubits = W_box.q_registers[0]
    state_qubits = W_box.q_registers[1]

    n_anc_qubits = len(anc_qubits)
    n_state_qubits = len(state_qubits)

    W_box = W.generate_basis_element(basis_index=1)
    W_box = W_box.power(power)
    unitary = W_box.get_circuit().get_unitary()

    # Compare to exact matrix
    projector = np.array(
        get_projector_matrix(
            list(range(n_anc_qubits)),
            list(range(n_anc_qubits, n_anc_qubits + n_state_qubits)),
        )
    )
    unitary_projected = projector.transpose() @ unitary @ projector
    h = unitary_projected - np.diag(unitary_projected.diagonal())
    assert np.abs(h).max() < 1e-12

    # Compare to exact Chebyshev polynomial
    x_vals = np.linspace(-1, 1, len(unitary_projected.diagonal()))
    coef = [0 for _ in range(power + 1)]
    coef[-1] = 1
    coef = np.array(coef)
    T = Chebyshev(coef=coef)
    y_vals = T(x_vals)
    np.testing.assert_allclose(unitary_projected.diagonal(), y_vals, atol=1e-12)


@pytest.mark.parametrize("basis_element", ["fourier", "chebyshev"])
@pytest.mark.parametrize(
    "index",
    [
        (0, "00", [1, 0, 0, 0]),
        (1, "01", [0, 1, 0, 0]),
        (2, "10", [0, 0, 1, 0]),
        (3, "11", [0, 0, 0, 1]),
    ],
)
def test_BaseLCUStatePreparationBlockEncoding_add_controlled_block_encoding_sequence(
    basis_element: str, index: tuple[int, str, list[int]]
):
    """Test BaseLCUStatePreparationBlockEncoding.add_controlled_block_encoding_sequence.

    Args:
    ----
        basis_element (str): _description_
        index (tuple[int, str, list[int]]): _description_

    """

    def initialise_bitstring(
        circ: RegisterCircuit, qreg: LCUStatePreparationQReg, bitstring: str
    ):
        assert len(bitstring) == len(qreg.coeffs[0])
        for idx, c in enumerate(bitstring):
            if c == "1":
                circ.X(qreg.coeffs[0][idx])

    # Define setting
    idx = index[0]
    bitstring = index[1]
    coeff = index[2]

    n_qubits_coeffs = [2]
    n_qubits = [5]
    basis_indices = [list(range(2 ** n_qubits_coeffs[0]))]
    if basis_element == "fourier":
        block_encoding = FourierBlockEncoding(n_block_qubits=n_qubits[0])
        n_qubits_block_encoding = [0]
    else:
        block_encoding = ChebychevBlockEncoding(n_block_qubits=n_qubits[0])
        n_qubits_block_encoding = [block_encoding.n_ancilla_qubits]

    # Build circuit
    circ = RegisterCircuit("test_circuit")

    coeffs_qreg = [
        circ.add_q_register(f"Coeffs_register_{i}", n)
        for i, n in enumerate(n_qubits_coeffs)
    ]
    state_qreg = [
        circ.add_q_register(f"State_register_{i}", n) for i, n in enumerate(n_qubits)
    ]
    block_qreg = [
        circ.add_q_register(f"Block_encoding_register_{i}", n_qubits_block_encoding[i])
        for i in range(len(n_qubits))
        if n_qubits_block_encoding[i] > 0
    ]
    qreg = LCUStatePreparationQReg(
        coeffs=coeffs_qreg, state=state_qreg, block=block_qreg
    )

    for q in [x for y in qreg.state for x in y]:
        circ.H(q)
    initialise_bitstring(circ, qreg, bitstring)
    block_encoding.add_controlled_block_encoding_sequence(
        n_qubits_coeffs=n_qubits_coeffs[0],
        basis_indices=basis_indices[0],
        circ=circ,
        qreg=qreg,
        control_idx=0,
    )
    initialise_bitstring(circ, qreg, bitstring)

    # Define projector and compute projected state vector
    if basis_element == "fourier":
        projection_qubits = list(range(len(qreg.coeffs[0])))
        identity_qubits = list(range(len(qreg.coeffs[0]), circ.n_qubits))
    else:
        projection_qubits = list(range(len(qreg.block[0]) + len(qreg.coeffs[0])))
        identity_qubits = list(
            range(len(qreg.block[0]) + len(qreg.coeffs[0]), circ.n_qubits)
        )
    vec = circ.get_statevector()
    projector = np.array(
        get_projector_matrix(
            projection_qubits,
            identity_qubits,
        )
    )
    vec_projected = projector.transpose() @ vec
    y_vals_circ = vec_projected
    factor = 1 / y_vals_circ.real.max()
    y_vals_circ = factor * y_vals_circ

    # Compare to exact function
    if basis_element == "fourier":
        x_vals = np.linspace(0, 1, 2 ** n_qubits[0])
        y_vals_exact = np.exp(idx * 1j * np.pi * x_vals)
    else:
        x_vals = np.linspace(-1, 1, 2 ** n_qubits[0])
        T = Chebyshev(coef=np.array(coeff))
        y_vals_exact = T(x_vals)

    np.testing.assert_allclose(y_vals_exact, y_vals_circ, atol=1e-12)
