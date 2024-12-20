"""Test operator expectation value with post selection."""

from mvsp.measurement.statevector import (
    operator_expectation_statevector,
    statevector_postselect,
)
from pytket._tket.circuit import Circuit
from pytket.utils import QubitPauliOperator
from mvsp.operators import ising_model
from pytket.extensions.qiskit.backends.aer import AerStateBackend
from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend
from numpy.random import rand
import numpy as np
from pytest_lazyfixture import lazy_fixture
import pytest
from pytket.backends.backend import Backend


@pytest.mark.parametrize(
    "op",
    [
        lazy_fixture("op_fixture"),
    ],
)
@pytest.mark.parametrize("backend", [AerStateBackend(), QulacsBackend()])
def test_operator_expectation_statevector_postselect(
    op: QubitPauliOperator, backend: Backend
):
    """Test operator expectation value for statevector backend with post selection."""
    op = ising_model(3, 1, 1)

    n_state_qubits = (
        max([p.index[0] for p_list in list(op._dict.keys()) for p in p_list.map.keys()])
        + 1
    )

    circ = Circuit(n_state_qubits)
    for i in range(n_state_qubits):
        circ.Ry(float(rand(1)), i)

    a = circ.add_q_register("a", 1)

    circ.Ry(float(rand(1)), a[0])
    post_select = {a[0]: 0}
    op_dense = op.to_sparse_matrix(n_state_qubits).todense()
    sv = circ.get_statevector()
    sv_postselect = np.array(
        statevector_postselect(circ.qubits, sv, post_select, renorm=True)
    ).T

    expectation_value_np = (sv_postselect.conj().T @ op_dense @ sv_postselect)[0, 0]
    expectation_value_circ = operator_expectation_statevector(
        circ, op, backend, post_select
    )
    np.testing.assert_allclose(expectation_value_np, expectation_value_circ)
