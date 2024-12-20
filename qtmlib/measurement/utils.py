"""Estimates the expectation value of a circuit with respect to an operator."""

from pytket.circuit import Qubit
from pytket.backends.backendresult import BackendResult
from numpy.typing import NDArray
from pytket._tket.circuit import Circuit
import numpy as np


def _reorder_qlist(
    post_select_dict: dict[Qubit, int] | list[Qubit], qlist: list[Qubit]
) -> tuple[list[Qubit], Qubit]:
    """Reorder qlist so that post_select_qubit is first in the list.

    Args:
    ----
        post_select_dict (dict): Dictionary of post selection qubit and value
        qlist (list): List of qubits

    Returns:
    -------
        Tuple containing a list of qubits reordered so that `post_select_qubit` is first
        in the list, and the post select qubit.

    """
    if isinstance(post_select_dict, list):
        post_select_q = post_select_dict[0]
    else:
        post_select_q = next(iter(post_select_dict.keys()))

    pop_i = qlist.index(post_select_q)

    q = qlist.pop(pop_i)

    q_list_reordered = [q]
    q_list_reordered.extend(qlist)

    return q_list_reordered, q


def recursive_statevector_postselect(
    qlist: list[Qubit], sv: NDArray[np.complex128], post_select_dict: dict[Qubit, int]
) -> NDArray[np.complex128]:
    """Post selects a statevector.

    Recursively calls itself if there are multiple post select qubits. Uses backend
    result to get statevector and permutes so the the post select qubit for each
    iteration is first in the list.

    Args:
    ----
        qlist (list): List of qubits
        sv (NDArray): Statevector
        post_select_dict (dict): Dictionary of post selection qubit and value

    Returns:
    -------
        Post selected statevector.

    """
    n = len(qlist)
    n_p = len(post_select_dict)

    b_res = BackendResult(state=sv, q_bits=qlist)

    q_list_reordered, q = _reorder_qlist(post_select_dict, qlist)

    sv = b_res.get_state(qbits=q_list_reordered)

    if post_select_dict[q] == 0:
        new_sv = sv[: 2 ** (n - 1)]
    elif post_select_dict[q] == 1:
        new_sv = sv[2 ** (n - 1) :]
    else:
        raise ValueError("post_select_dict[q] must be 0 or 1")

    if n_p == 1:
        return new_sv

    post_select_dict.pop(q)
    q_list_reordered.pop(0)

    return recursive_statevector_postselect(q_list_reordered, new_sv, post_select_dict)


def statevector_postselect(
    qubits: list[Qubit],
    statevector: NDArray[np.complex128],
    post_select_dict: dict[Qubit, int],
    renorm: bool = False,
) -> NDArray[np.complex128]:
    """Post selects a circuit statevector.

    Recursively calls itself if there are multiple post select qubits.
    Should only be used for testing small circuits as it uses the
    circuit.get_statevector() method. Does not account for global phase.
    Does not normalise the output statevector.

    Args:
    ----
        qubits (list): List of qubits
        statevector (NDArray): Statevector
        post_select_dict (dict): Dictionary of post selection qubit and value
        renorm (bool): Whether to renormalise the statevector

    Returns:
    -------
        Post selected statevector function (not normalised).

    """
    post_select_dict_copy = post_select_dict.copy()
    ps_sv = recursive_statevector_postselect(qubits, statevector, post_select_dict_copy)
    if renorm:
        norm = np.linalg.norm(ps_sv)
        if norm == 0:
            raise ValueError("Post selected statevector is vanishingly small")
        ps_sv /= np.linalg.norm(ps_sv)
    return ps_sv


def circuit_statevector_postselect(
    circ: Circuit, post_select_dict: dict[Qubit, int], renorm: bool = False
) -> NDArray[np.complex128]:
    """Post selects a circuit statevector.

    Recursively calls itself if there are multiple post select qubits.
    Should only be used for testing small circuits as it uses the
    circuit.get_statevector() method. Does not account for global phase.

    Args:
    ----
        circ (Circuit): Circuit
        post_select_dict (dict): Dictionary of post selection qubit and value
        renorm (bool): Whether to renormalise the statevector

    Returns:
    -------
        post selected statevector

    """
    return statevector_postselect(
        circ.qubits, circ.get_statevector(), post_select_dict.copy(), renorm
    )


def unitary_postselect(
    qlist: list[Qubit],
    unitary: NDArray[np.complex128],
    post_select_dict: dict[Qubit, int],
    pre_select_dict: dict[Qubit, int] | None = None,
) -> NDArray[np.complex128]:
    """Post/Pre - selects a unitary matrix.

    Iterates over the dictionary if there are multiple post select qubits.
    Uses backend result to get unitary matrix and permutes so
    the the post select qubit for each iteration is first in the list.
    Then, the unitary is sliced to keep the section corresponding to the
    pre, post select qubit values.

    Unless pre_select_dict is passed, assumes that the postselected qubits begin
    in the 0 state.

    Preselected and Postselected qubits must be the same, and in the same order in
    their respective dictionaries.

    Args:
    ----
        qlist (list): List of qubits
        unitary (npt.NDArray): Unitary matrix
        post_select_dict (dict): Dictionary of post selection qubit and value
        pre_select_dict (dict): Dictionary of pre selection qubit and value (optional)

    Returns:
    -------
        unitary (npt.NDArray): Post selected unitary matrix

    """
    if pre_select_dict is not None:
        if len(pre_select_dict) != len(post_select_dict):
            raise ValueError(
                "pre_select_dict and post_select_dict must have the same length"
            )
        if list(pre_select_dict.keys()) != list(post_select_dict.keys()):
            raise ValueError(
                "both dictionaries must have the same keys, in the same order"
            )
    for _j in range(len(post_select_dict)):
        n = len(qlist)

        b_res = BackendResult(unitary=unitary, q_bits=qlist)

        dict_first_entry = next(iter(post_select_dict.keys()))
        dict_first_bit = post_select_dict[dict_first_entry]
        one_item_dict = {dict_first_entry: dict_first_bit}

        q_list_reordered, q = _reorder_qlist(one_item_dict, qlist)

        u = b_res.get_unitary(qbits=q_list_reordered)

        if pre_select_dict is not None:
            pre_select_bit = pre_select_dict[dict_first_entry]
            if dict_first_bit == 0 and pre_select_bit == 0:
                new_u: NDArray[np.complex128] = u[: 2 ** (n - 1), : 2 ** (n - 1)]
            elif dict_first_bit == 1 and pre_select_bit == 0:
                new_u: NDArray[np.complex128] = u[2 ** (n - 1) :, : 2 ** (n - 1)]
            elif dict_first_bit == 0 and pre_select_bit == 1:
                new_u: NDArray[np.complex128] = u[: 2 ** (n - 1), 2 ** (n - 1) :]
            elif dict_first_bit == 1 and pre_select_bit == 1:
                new_u: NDArray[np.complex128] = u[2 ** (n - 1) :, 2 ** (n - 1) :]
            else:
                raise ValueError("post_select_dict[q] must be 0 or 1")
            del pre_select_dict[q]
        else:
            if dict_first_bit == 0:
                new_u: NDArray[np.complex128] = u[: 2 ** (n - 1), : 2 ** (n - 1)]
            elif dict_first_bit == 1:
                new_u: NDArray[np.complex128] = u[2 ** (n - 1) :, : 2 ** (n - 1)]
            else:
                raise ValueError("post_select_dict[q] must be 0 or 1")

        del post_select_dict[q]
        del q_list_reordered[0]

        qlist = q_list_reordered
        unitary = new_u

    return unitary


def circuit_unitary_postselect(
    circ: Circuit,
    post_select_dict: dict[Qubit, int],
    pre_select_dict: dict[Qubit, int] | None = None,
) -> NDArray[np.complex128]:
    """Post selects a circuit unitary.

    Recursively calls itself if there are multiple post select qubits.
    Should only be used for testing small circuits as it uses the
    circuit.get_unitary() method.

    Args:
    ----
        circ (Circuit): Circuit
        post_select_dict (dict): Dictionary of post selection qubit and value
        pre_select_dict (dict): Dictionary of pre selection qubit and value (optional)

    Returns:
    -------
        Post selected unitary.

    """
    return unitary_postselect(
        circ.qubits,
        circ.get_unitary(),
        post_select_dict.copy(),
        pre_select_dict.copy() if pre_select_dict is not None else None,
    )  # TODO this does not account for global phase if just taking circuit


def bit_fixed_point(bits: tuple[int, ...]):
    """Convert a tuple of bits to a fixed point decimeal.

    Each bit string is incremented from 0 -> 1 by 1/(2**len(bits)))

    Args:
    ----
        bits (tuple[int, ...]): tuple of bits

    Returns:
    -------
        float: decimal value of bits

    """
    bit_str = "".join([str(b) for b in bits])
    return int(bit_str, 2) / 2 ** (len(bits))


def dist_to_fixed_point(dist: dict[tuple[int, ...], float]):
    """Convert a distribution of bit strings to a distribution of fixed point decimals.

    Args:
    ----
        dist (dict[tuple[int, ...], float]): dictionary of bit strings and probabilities

    Returns:
    -------
        dict[float, float]: dictionary of fixed point decimals and probabilities

    """
    return {bit_fixed_point(bits): prob for bits, prob in dist.items()}
