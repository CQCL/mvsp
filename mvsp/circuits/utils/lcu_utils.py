"""Ultils functions for LCU circuits."""

from __future__ import annotations
from scipy.sparse import csc_matrix
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mvsp.circuits.lcu import LCUBox


def block_encoded_sparse_matrix(lcu_box: LCUBox) -> csc_matrix:
    """Return the block encoded sparse matrix of the operator divided by the l1 norm.

    To be used for benchmarking and testing.

    Returns
    -------
        csc_matrix: The block encoded sparse matrix of the operator divided by
        the l1 norm

    """
    return lcu_box.operator.to_sparse_matrix(lcu_box.n_state_qubits) / lcu_box.l1_norm


def int_to_bits(integer: int, length: int) -> list[bool]:
    """Convert an integer to a bit string of inputlength."""
    return [bool(int(x)) for x in bin(integer)[2:].zfill(length)]


def is_hermitian(lcu_box: LCUBox) -> bool:
    """Check if the block encoding unitary is hermitian.

    Args:
    ----
        lcu_box (LCUBox): The LCUBox to check.

    Returns:
    -------
        bool: Whether the block encoding unitary is hermitian.

    """
    identity = np.eye(2 ** (lcu_box.n_prepare_qubits + lcu_box.n_state_qubits))
    unitary = lcu_box.reg_circuit.get_unitary()

    u_squared = unitary @ unitary

    return np.allclose(identity, u_squared, rtol=1e-10, atol=1e-10)
