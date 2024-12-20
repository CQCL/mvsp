"""Test phase estimation measurement functions."""

from pytket._tket.circuit import Circuit
from pytket.extensions.qiskit.backends.aer import AerBackend
from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend
import numpy as np
import pytest
from pytket.backends.backend import Backend
from qtmlib.measurement.phase_estimation import measure_phase_estimation
import itertools


# @pytest.mark.parametrize("backend", [AerStateBackend(), QulacsBackend()])
# @pytest.mark.parametrize("n_state_qubits", [1, 2, 3])
# def test_measure_phase_estimation_statevector(backend: Backend, n_state_qubits: int):
#     """Test measure_phase_estimation_statevector function."""
#     circ = Circuit()

#     q = circ.add_q_register("q", n_state_qubits)
#     a = circ.add_q_register("a", 2)
#     p = circ.add_q_register("p", 2)

#     for qubit in q:
#         circ.Ry(0.1, qubit)

#     for qubit in a:
#         circ.Ry(0.3, qubit)

#     for qubit in p:
#         circ.Ry(0.4, qubit)

#     post_select = {p: 0 for p in p}

#     n_shots = None
#     dist = measure_phase_estimation(
#         circ.copy(), backend, a, n_shots, post_select.copy()
#     )

#     dist_sv = np.array(list(dist.values()))

#     shots_backend = AerBackend()
#     n_shots = 1000000
#     dist = measure_phase_estimation(circ, shots_backend, a, n_shots, post_select)

#     dist_sv = np.array(list(dist.values()))

#     np.testing.assert_allclose(dist_sv, dist_sv, atol=1e-2)


@pytest.mark.parametrize("backend", [AerBackend(), QulacsBackend()])
def test_measure_phase_estimation_shots(backend: Backend) -> None:
    """Test distribution is correctly measured when no post-selection is specified."""
    circ = Circuit()
    n_ancilla = 2
    n_state = 2
    q = circ.add_q_register("q", n_state)
    circ.H(q[0]).H(q[1])
    ancilla_qreg = circ.add_q_register("ancilla", n_ancilla)
    circ.H(ancilla_qreg[0]).H(ancilla_qreg[1])

    expected_dist: dict[tuple[int, ...], float] = {
        bits: 1 / (2**n_ancilla)
        for bits in list(itertools.product([0, 1], repeat=n_ancilla))
    }

    n_shots = 10000
    actual_dist = measure_phase_estimation(circ, backend, ancilla_qreg, n_shots)

    expected_dist_vec = np.array(list(expected_dist.values()))
    actual_dist_vec = np.array(list(actual_dist.values()))

    np.testing.assert_array_almost_equal(actual_dist_vec, expected_dist_vec, decimal=2)

    """Test the distribution is correctly measured when post-selection is specified."""

    circ = Circuit()
    n_ancilla = 2
    n_state = 2
    q = circ.add_q_register("q", n_state)
    circ.H(q[0]).H(q[1])
    ancilla_qreg = circ.add_q_register("ancilla", n_ancilla)
    circ.H(ancilla_qreg[0]).H(ancilla_qreg[1])

    expected_dist: dict[tuple[int, ...], float] = {
        bits: 1 / (2**n_ancilla)
        for bits in list(itertools.product([0, 1], repeat=n_ancilla))
    }

    n_shots = 100000
    post_select_dict = {q[0]: 0, q[1]: 1}
    actual_dist = measure_phase_estimation(
        circ, backend, ancilla_qreg, n_shots, post_select_dict
    )

    expected_dist_vec = np.array(list(expected_dist.values()))
    actual_dist_vec = np.array(list(actual_dist.values()))

    np.testing.assert_array_almost_equal(actual_dist_vec, expected_dist_vec, decimal=2)
