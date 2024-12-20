"""utils functions for LCU circuits."""

import string
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pytket.circuit import Op, OpType


def Rz_jkn(j: int, k: int, n: int) -> Op:
    """Create Z rotation operator.

    A single qubit Z rotation operator around an angle specified by parameters
    j, k and n. Needed to build a block encoding of the Fourier series basis elements
    for the LCUStatePreparationBox.

    Args:
    ----
        j (int): Index j.
        k (int): Index k.
        n (int): Index n.

    Returns:
    -------
        Op: Z rotation operator.

    """
    rz = Op.create(OpType.Rz, k * 2.0**j / (2.0**n - 1))
    return rz


def generate_diagonal_entries(
    phi_basis_k: list[int | float | np.float64] | NDArray[np.float64] | np.float64,
) -> NDArray[np.complex128] | np.complex128:
    """Generate entries of diagonal operator used in LCUStatePreparationBox.

    Args:
    ----
        phi_basis_k (np.float64): Phase for diagonal entry.

    Returns:
    -------
        np.complex128: Entry of the diagonal operator used in LCUStatePreparationBox.

    """
    return np.exp(1j * np.array(phi_basis_k))


def extend_functions(
    dims_variables: list[int],
    m_basis: NDArray[np.float64],
    phi_basis: NDArray[np.float64],
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[int],
    list[int],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extend arrays such that there entries can be encoded in a qubit register.

    Args:
    ----
        dims_variables (list[int]): Dimensions of the variables.
        m_basis (NDArray): Magnitudes of the basis coefficients.
        phi_basis (NDArray): Phases of the basis coefficients.

    Returns:
    -------
        tuple[list, list, list, list, NDArray, NDArray]: Outputs.

    """
    dims_basis_variables = list(m_basis.shape)
    n_qubits_basis = [int(np.ceil(np.log2(d))) for d in dims_basis_variables]
    n_qubits = [int(np.ceil(np.log2(d))) for d in dims_variables]

    n_counter = 0

    def _get_regs(
        n_qubits_list: list[int], n_counter: int
    ) -> tuple[list[list[int]], int]:
        """Auxiliary function that computes the registers."""
        regs: list[list[int]] = []
        for idx in range(len(n_qubits_list)):
            regs.append(list(range(n_counter, n_counter + n_qubits_list[idx])))
            n_counter += n_qubits_list[idx]
        return regs, n_counter

    regs_basis, n_counter = _get_regs(n_qubits_basis, n_counter)
    regs, n_counter = _get_regs(n_qubits, n_counter)

    dims_basis_variables_extended: list[int] = [2**d for d in n_qubits_basis]

    def _get_pad(basis: NDArray[np.float64]) -> NDArray[np.float64]:
        """Auxiliary function that pads the basis elements."""
        return np.pad(
            basis,
            np.array(
                [
                    (0, dims_basis_variables_extended[i] - dims_basis_variables[i])
                    for i in range(len(dims_basis_variables))
                ]
            ),
            mode="constant",
            constant_values=0,
        )

    m_basis_extended = _get_pad(m_basis)
    phi_basis_extended = _get_pad(phi_basis)

    return (
        regs,
        regs_basis,
        n_qubits,
        n_qubits_basis,
        m_basis_extended,
        phi_basis_extended,
    )


def create_einsum_string(n_tensors: int) -> str:
    """Create a generic einsum string of length n_tensors.

    Args:
    ----
        n_tensors (int): Number of tensors.

    Returns:
    -------
        str: Einsum string.

    """
    alphabet = string.ascii_lowercase
    input_string = ",".join(alphabet[:n_tensors])
    output_string = alphabet[:n_tensors]

    einsum_string = input_string + "->" + output_string
    return einsum_string


def generate_test_functions(
    dims_basis_variables: tuple[int, ...],
) -> NDArray[np.complex128]:
    """Generate test function for testing the LCUStatePreparationBox.

    Args:
    ----
        dims_basis_variables (tuple[int]): Dimensions of the basis variables.

    Returns:
    -------
        tuple[NDArray]: Random magnitudes and phases of the basis coefficients.

    """
    f_basis = np.random.uniform(0, 1, tuple(dims_basis_variables))
    phi_basis = np.random.uniform(-np.pi, np.pi, tuple(dims_basis_variables))
    coeffs = f_basis * np.exp(1j * phi_basis)
    return np.array(coeffs, dtype=np.complex128)


def generate_test_functions_separable(
    dims_basis_variables: tuple[int, ...],
) -> list[NDArray[np.floating[Any]]]:
    """Generate test function for testing the LCUStatePreparationBox.

    Args:
    ----
        dims_basis_variables (tuple[int]): Dimensions of the basis variables.

    Returns:
    -------
        tuple[NDArray]: Random magnitudes and phases of the basis coefficients.

    """
    coeffs = [
        np.random.uniform(0, 1, dim)
        * np.exp(1j * np.random.uniform(-np.pi, np.pi, dim))
        for dim in dims_basis_variables
    ]
    return coeffs
