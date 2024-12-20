"""Linear algebra utilities for quantum circuits."""

from numpy.typing import NDArray
import numpy as np


def kron_list(vecs: list[NDArray[np.complex128]]):
    """Kronecker product of a list of vectors.

    Args:
    ----
        vecs (list[NDArray[np.complex128]]): List of vectors.

    Returns:
    -------
        NDArray[np.complex128]: The Kronecker product of the vectors.

    """
    vec_kron = vecs[0]
    for vec in vecs[1:]:
        vec_kron = np.kron(vec_kron, vec)
    return vec_kron
