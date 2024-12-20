"""Statevector measurement functions."""

from pytket.utils.operators import QubitPauliOperator
from qtmlib.measurement.utils import statevector_postselect
from pytket._tket.circuit import Circuit
from pytket.circuit import Qubit
from pytket.backends.backend import Backend
from pytket.backends.backendresult import BackendResult
from typing import cast
import numpy as np


def operator_expectation_statevector(
    state_circuit: Circuit,
    operator: QubitPauliOperator,
    backend: Backend,
    post_select: dict[Qubit, int] | None = None,
) -> complex:
    """Stavector expectation value of a circuit with respect to an operator.

    If post select is not None, the statevector is post selected before
    expaction value is calculated on the state. It is compabile with any
    pytket statevector backends.

    Args:
    ----
        state_circuit (Circuit): The circuit to be run.
        operator (QubitPauliOperator): The operator to be measured.
        backend (Backend): The backend to run the circuit on.
        post_select (dict): Dictionary of post selection qubit and value

    Returns:
    -------
        The expectation value of the operator act on the state circuit.

    """
    qubits = state_circuit.qubits

    if not backend.valid_circuit(state_circuit):
        state_circuit = backend.get_compiled_circuit(state_circuit)
    try:
        # TODO: Make QPO have terms list dataclass
        coeffs: list[complex] = [complex(v) for v in operator._dict.values()]
    except TypeError:
        raise ValueError("QubitPauliOperator contains unevaluated symbols.") from None
    if (
        backend.supports_expectation
        and (
            backend.expectation_allows_nonhermitian or all(z.imag == 0 for z in coeffs)
        )
        and (post_select is None)
    ):
        return backend.get_operator_expectation_value(state_circuit, operator)

    result = cast(BackendResult, backend.run_circuit(state_circuit))  # HACK fix pytket
    state = result.get_state()
    if post_select is not None:
        state = statevector_postselect(qubits, state, post_select)
        state = state / np.linalg.norm(state)
        return operator.state_expectation(state)
    else:
        return operator.state_expectation(state)


def get_statevector_distribution(
    backend: Backend, state_circuit: Circuit
) -> dict[tuple[int, ...], float]:
    """Get statevector distribution.

    Does not use measuremnt gates. Gives back full distribution.

    Args:
    ----
        backend (Backend): backend to run the circuit
        state_circuit (Circuit): circuit to be measured

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of shots distribution

    """
    if not backend.valid_circuit(state_circuit):
        state_circuit = backend.get_compiled_circuit(state_circuit)
    handle = backend.process_circuit(state_circuit)
    dist = backend.get_result(handle).get_probability_distribution().as_dict()
    return dist
