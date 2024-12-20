"""Utils functions for quantum arithmetic circuits."""

from pytket.circuit import CircBox
from pytket._tket.circuit import Circuit
from qtmlib.circuits.utils.lcu_utils import int_to_bits


def prep_bitstring_box(
    x: int,
    n_bits: int,
    reverse: bool = False,
    append_zero: bool = False,
) -> CircBox:
    r"""CircBox to prepare initial bitstring state.

    Args:
    ----
        x (int): The integer value of the bitstring.
        n_bits (int): The number of qubits in the state register.
        reverse (bool): If True, use big-endian notation.
        append_zero (bool): If True, append 0 as the most significant qubit.

    """
    str = "{0:{fill}" + f"{n_bits}b" + "}"
    x_bin = str.format(x, fill="0")

    n_qubits = n_bits if not append_zero else n_bits + 1
    circ = Circuit(n_qubits)
    circ.name = "Init"

    for i in range(n_bits):
        if reverse:
            flip_ind = n_bits - 1 - i
        else:
            flip_ind = i
            if append_zero:
                flip_ind += 1
        if x_bin[i] == "1":
            circ.X(flip_ind)

    return CircBox(circ)


def find_first_diff(a_bits: list[bool], b_bits: list[bool]) -> int:
    """Find the first differing bit index between two bitstrings."""
    assert len(a_bits) == len(b_bits)
    first_diff = -1
    for i in range(len(a_bits)):
        if a_bits[i] != b_bits[i] and first_diff == -1:
            first_diff = i
    return first_diff


def prep_two_bitstrings_box(
    a_int: int,
    b_int: int,
    n_bits: int,
) -> CircBox:
    r"""CircBox to prepare an equal superposition of two bitstring states.

    Args:
    ----
        a_int (int): The integer value of bitstring 'a'.
        b_int (int): The integer value of bitstring 'b'.
        n_bits (int): The number of qubits in the state register.

    """
    a_bits, b_bits = int_to_bits(a_int, n_bits), int_to_bits(b_int, n_bits)
    first_diff = find_first_diff(a_bits, b_bits)

    circ = Circuit(n_bits)
    circ.H(first_diff)

    for i in range(len(a_bits)):
        if a_bits[i] == b_bits[i]:
            if a_bits[i]:
                circ.X(i)
        else:
            if i > first_diff:
                circ.CX(first_diff, i)
                if a_bits[i] != a_bits[first_diff]:
                    circ.X(i)

    return CircBox(circ)
