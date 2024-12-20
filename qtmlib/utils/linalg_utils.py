"""Basic linear algebra operations."""

from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def get_projector_csr_matrix(
    projection_qubits: list[int],
    identity_qubits: list[int],
    proj_bit_string: str | None = None,
) -> csr_matrix:
    """Sparse projector onto subspace of qubits.

    Generate a projection matrix that projects a statevector onto a specific bitstring
    of the projecetion_qubits. The identity acts on the identity_qubits.
    The proj_bit_string is ordered according to the index in projection_qubit and not
    the actual qubit.

    Assume certain qubit ordering: AAA'BB
    Here AAA is the projection bitstring (here 3 qubits) and BB are qubits to
    keep (here 2 qubits).

    In contrast, the dense function get_projector_matrix allows for arbitrary ordering.

    Args:
    ----
        projection_qubits (list[int]): List of qubit indices which are projected out.
        identity_qubits (list[int]): List of qubit indices that remain after projection.
        proj_bit_string (str | None, optional): String specifying a specific state onto
            which it should be projected. Defaults to None, in which case the
            proj_bit_string is set to "0...0".

    Returns:
    -------
        np.ndarray: Matrix representation of projector.

    """
    all_qubits = list(set(projection_qubits + identity_qubits))
    assert min(all_qubits) == 0
    assert max(all_qubits) == len(projection_qubits) + len(identity_qubits) - 1
    assert len(all_qubits) == len(projection_qubits) + len(identity_qubits)
    if proj_bit_string is None:
        proj_bit_string = len(projection_qubits) * "0"
    else:
        assert len(proj_bit_string) == len(projection_qubits)

    n_all_qubits = len(all_qubits)
    n_qubits_to_keep = len(identity_qubits)
    idx_to_keep: list[int] = []
    # TODO Allow for arbitrary ordering of projection and identity qubits.
    # The following line bit shifts to the left the projection
    # bitstring by the number of qubits to keep. The subsequent loop
    # applies binary or to set all combinations of lower-order bitstings for the
    # qubits to keep, i.e. AAA is fixed, BB loops through all possible length-2
    # bitstrings.
    # This could be generalised to nonconsecutive projection qubits by
    # fixing bits at the position of each projected qubit.
    proj_bit_mask = int(proj_bit_string, 2) << n_qubits_to_keep
    for i in range(2**n_qubits_to_keep):
        idx: int = i | proj_bit_mask
        idx_to_keep.append(idx)

    return csr_matrix(
        (
            [1.0] * len(idx_to_keep),
            (tuple(idx_to_keep), tuple(range(len(idx_to_keep)))),
        ),
        shape=(2**n_all_qubits, 2**n_qubits_to_keep),
    )


def get_projector_matrix(
    projection_qubits: list[int],
    identity_qubits: list[int],
    proj_bit_string: str | None = None,
) -> NDArray[np.float64]:
    """Dense projector onto subspace of qubits.

    Generate a projection matrix that projects a statevector onto a specific bitstring
    of the projecetion_qubits. The identity acts on the identity_qubits.
    The proj_bit_string is ordered according to the index in projection_qubit and not
    the actual qubit. The list projections_qubits and identity_qubits can be arbitrary,
    they don't have to be ordered specifically. For the sparse projection function
    get_projector_csr_matrix, they have to be ordered in a specific way.

    Args:
    ----
        projection_qubits (list[int]): List of qubit indices which are projected out.
        identity_qubits (list[int]): List of qubit indices that remain after projection.
        proj_bit_string (str | None, optional): String specifying a specific state onto
            which it should be projected. Defaults to None, in which case the
            proj_bit_string is set to "0...0".

    Returns:
    -------
        NDArray[np.float64]: Matrix representation of projector.

    """
    all_qubits = list(set(projection_qubits + identity_qubits))
    assert min(all_qubits) == 0
    assert max(all_qubits) == len(projection_qubits) + len(identity_qubits) - 1
    assert len(all_qubits) == len(projection_qubits) + len(identity_qubits)
    if proj_bit_string is None:
        proj_bit_string = len(projection_qubits) * "0"
    else:
        assert len(proj_bit_string) == len(projection_qubits)

    if len(projection_qubits) == 0:
        return np.identity(2 ** len(identity_qubits))
    n_all_qubits = len(all_qubits)
    n_qubits_to_keep = len(identity_qubits)
    projector = np.zeros((2**n_all_qubits, 2**n_qubits_to_keep))
    idx_to_keep: list[int] = []
    # Assume certain qubit ordering:
    # AAA'BB
    # where AAA is the projection bitstring (here 3 qubits) and BB are qubits to
    # keep (here 2 qubits). The following line bit shifts to the left the projection
    # bitstring by the number of qubits to keep. The subsequent loop
    # applies binary or to set all combinations of lower-order bitstings for the
    # qubits to keep, i.e. AAA is fixed, BB loops through all possible length-2
    # bitstrings.
    # This could be generalised to nonconsecutive projection qubits by
    # fixing bits at the position of each projected qubit.
    proj_bit_mask = int(proj_bit_string, 2) << n_qubits_to_keep
    for i in range(2**n_qubits_to_keep):
        idx = i | proj_bit_mask
        idx_to_keep.append(idx)

    # The projector is 1 in the rows specified by the fixed bitstring AAA.
    for i, j in zip(idx_to_keep, range(len(idx_to_keep)), strict=True):
        projector[i, j] = 1.0

    return projector


def partial_trace(
    dmat: NDArray[np.complex128],
    projection_qubits: list[int],
    identity_qubits: list[int],
) -> NDArray[np.complex128]:
    """Partial trace of density matrix.

    This function calculates the partial trace over the projection qubits
    by summing up the projections of dmat on all bitstrings in the projection
    register. There might be better ways to implement the partial trace...

    Args:
    ----
        dmat (NDArray[np.complex128]): Matrix representation of density matrix.
        projection_qubits (list[int]): List of qubit indices which are traced out.
        identity_qubits (list[int]): List of qubit indices that remain after projection.

    Returns:
    -------
        NDArray[np.complex128]: Matrix representation of reduced state.

    """
    all_qubits = list(set(projection_qubits + identity_qubits))
    assert min(all_qubits) == 0
    assert max(all_qubits) == len(projection_qubits) + len(identity_qubits) - 1
    assert len(all_qubits) == len(projection_qubits) + len(identity_qubits)

    if len(projection_qubits) == 0:
        return dmat
    else:
        bit_tuples = list(product(["0", "1"], repeat=len(projection_qubits)))
        projected_dmats = np.zeros(
            [len(bit_tuples), 2 ** len(identity_qubits), 2 ** len(identity_qubits)],
            dtype=complex,
        )
        for idx in range(len(bit_tuples)):
            bit_tuple = bit_tuples[idx]
            bit_string = "".join(bit_tuple)
            projector = get_projector_matrix(
                projection_qubits, identity_qubits, proj_bit_string=bit_string
            )
            projected_dmats[idx] = (
                projector.conjugate().transpose() @ dmat
            ) @ projector
        return np.sum(projected_dmats, axis=0)
