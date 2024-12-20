"""Useful pytket CircBoxes."""

from pytket._tket.circuit import Circuit
from pytket.circuit import ConjugationBox, CircBox
from pytket.pauli import Pauli


def phased_paulig_box(paulis: list[Pauli], phase: float) -> ConjugationBox:
    """Return a ConjugationBox for a phased Pauli operator.

    This is to be used in the LCU classes. the operator is compiled
    using the Pauli Gadget Strategy.

    Args:
    ----
        paulis (list[Pauli]): The list of Pauli operators.
        phase (float): The phase of the operator.

    Returns:
    -------
        ConjugationBox: The compiled operator.

    """
    compute_circ = Circuit(len(paulis))

    for i, pauli in enumerate(paulis):
        if pauli == Pauli.X:
            compute_circ.H(i)
        elif pauli == Pauli.Y:
            compute_circ.V(i)
        else:
            continue

    for q0, q1 in zip(compute_circ.qubits[:-1], compute_circ.qubits[1:], strict=True):
        compute_circ.CX(q0, q1)

    action_circ = Circuit(len(paulis))
    action_circ.Z(len(paulis) - 1)
    action_circ.Phase(phase)

    return ConjugationBox(CircBox(compute_circ), CircBox(action_circ))
