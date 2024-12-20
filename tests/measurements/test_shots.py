"""Test the measurement functions in mvsp.measurements.shots."""

import numpy as np
from pytket.circuit import Qubit
from pytket._tket.unit_id import Bit
from pytket._tket.circuit import Circuit
from mvsp.measurement.statevector import statevector_postselect
from mvsp.measurement.shots import (
    add_measure_post_select,
    measure_distribution,
    expectation_from_dist,
    pauli_expectation,
    get_shots_distribution,
    post_select_distribution,
    append_pauli_measurement_register,
    operator_expectation,
)
from pytket.extensions.qiskit.backends.aer import AerBackend
from pytket.pauli import QubitPauliString, Pauli
from numpy.typing import NDArray
from pytket.utils import QubitPauliOperator
import itertools


def dist_to_vec(dist: dict[tuple[int, ...], float]) -> NDArray[np.float64]:
    """Convert a distribution to a vector."""
    return np.array(list(dist.values()))


def test_add_measure_post_select() -> None:
    """Test post-select bits are added to the circuit and measured correctly."""
    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)

    # Test case 1: post-select on a single qubit
    post_select_dict = {Qubit(0): 0}
    expected_postselect_bit_ind = {0: 0}
    expected_circ = Circuit(3)
    expected_circ.Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    m = Bit("cq", 0)
    expected_circ.add_bit(m)
    expected_circ.Measure(Qubit(0), m)

    postselect_bit_ind, actual_circ = add_measure_post_select(
        circ.copy(), post_select_dict
    )

    assert postselect_bit_ind == expected_postselect_bit_ind
    assert actual_circ == expected_circ

    # Test case 2: post-select on multiple qubits
    post_select_dict = {Qubit(0): 0, Qubit(1): 1}
    expected_postselect_bit_ind = {0: 0, 1: 1}
    expected_circ = Circuit(3)
    expected_circ.Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    m0 = Bit("cq", 0)
    m1 = Bit("cq", 1)
    expected_circ.add_bit(m0)
    expected_circ.add_bit(m1)
    expected_circ.Measure(Qubit(0), m0)
    expected_circ.Measure(Qubit(1), m1)

    postselect_bit_ind, actual_circ = add_measure_post_select(
        circ.copy(), post_select_dict
    )

    assert postselect_bit_ind == expected_postselect_bit_ind
    assert actual_circ == expected_circ

    circ = Circuit(4).H(0).CX(0, 1).CX(1, 2).CX(2, 3)
    q = circ.get_q_register("q")

    expected_circ = circ.copy()

    post_select_dict = {q[0]: 0, q[1]: 1}

    for q in post_select_dict.keys():
        m = Bit(f"c{q.reg_name}", q.index[0])
        expected_circ.add_bit(m)
        expected_circ.Measure(q, m)

    expected_bit_ind = {0: 0, 1: 1}

    postselect_bit_ind, actual_circ = add_measure_post_select(
        circ.copy(), post_select_dict
    )

    assert postselect_bit_ind == expected_bit_ind
    assert actual_circ == expected_circ


def test_post_select_distribution() -> None:
    """Test that the distribution is correctly post-selected."""
    dist: dict[tuple[int, ...], float] = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (1, 0): 0.25,
        (1, 1): 0.25,
    }

    # Test case 1: post-select on a single qubit
    post_select_dict = {0: 0}
    expected_postselect_dist: dict[tuple[int, ...], float] = {(0,): 0.5, (1,): 0.5}
    actual_postselect_dist = post_select_distribution(dist, post_select_dict)

    actual_postselect_dist_vec = dist_to_vec(actual_postselect_dist)
    expected_postselect_dist_vec = dist_to_vec(expected_postselect_dist)

    np.testing.assert_array_equal(
        actual_postselect_dist_vec, expected_postselect_dist_vec
    )

    # Test case 2: post-select on multiple qubits
    n_bits = 3
    dist: dict[tuple[int, ...], float] = {
        bits: 1 / (2**n_bits) for bits in list(itertools.product([0, 1], repeat=n_bits))
    }

    post_select_dict = {0: 0, 1: 1}
    expected_postselect_dist = {(0, 0): 0.5, (1, 1): 0.5}
    actual_postselect_dist = post_select_distribution(dist, post_select_dict)

    actual_postselect_dist_vec = dist_to_vec(actual_postselect_dist)
    expected_postselect_dist_vec = dist_to_vec(expected_postselect_dist)

    np.testing.assert_array_equal(
        actual_postselect_dist_vec, expected_postselect_dist_vec
    )

    # Test case 3: post-select on all qubits
    n_bits = 3
    dist: dict[tuple[int, ...], float] = {
        bits: 1 / (2**n_bits) for bits in list(itertools.product([0, 1], repeat=n_bits))
    }

    post_select_dict = {0: 0, 1: 1, 2: 0}
    expected_postselect_dist = {(0, 1, 0): 1.0}
    actual_postselect_dist = post_select_distribution(dist, post_select_dict)

    actual_postselect_dist_vec = dist_to_vec(actual_postselect_dist)
    expected_postselect_dist_vec = dist_to_vec(expected_postselect_dist)

    np.testing.assert_array_equal(
        actual_postselect_dist_vec, expected_postselect_dist_vec
    )


def test_measure_distribution() -> None:
    """Test that the distribution is correctly measured."""
    backend = AerBackend()
    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)

    # Test case 1: no post-selection
    n_shots = 1000
    expected_dist = get_shots_distribution(backend, circ.copy(), n_shots)
    actual_dist = measure_distribution(backend, circ.copy(), n_shots)

    expected_dist_vec = np.array(list(expected_dist.values()))
    actual_dist_vec = np.array(list(actual_dist.values()))

    np.testing.assert_array_almost_equal(actual_dist_vec, expected_dist_vec, decimal=2)

    # Test case 2: post-select on a single qubit
    post_select_dict = {Qubit(0): 0}

    post_select_dict_ind, new_circ = add_measure_post_select(
        circ.copy(), post_select_dict
    )
    expected_postselect_dist = get_shots_distribution(backend, new_circ, n_shots)
    expected_postselect_dist = post_select_distribution(
        expected_postselect_dist, post_select_dict_ind
    )

    actual_postselect_dist = measure_distribution(
        backend, circ.copy(), n_shots, post_select_dict
    )

    expected_dist_vec = np.array(list(expected_postselect_dist.values()))
    actual_dist_vec = np.array(list(actual_postselect_dist.values()))

    np.testing.assert_array_almost_equal(actual_dist_vec, expected_dist_vec, decimal=2)

    # Test case 3: post-select multiple different named qubits
    circ = Circuit()
    a = circ.add_q_register("a", 3)
    circ.Ry(0.1, a[0]).Ry(0.2, a[1]).Ry(0.2, a[2])
    post_select_dict = {a[0]: 0, a[1]: 1}

    post_select_dict_ind, new_circ = add_measure_post_select(
        circ.copy(), post_select_dict
    )
    expected_postselect_dist = get_shots_distribution(backend, new_circ, n_shots)
    expected_postselect_dist = post_select_distribution(
        expected_postselect_dist, post_select_dict_ind
    )

    actual_postselect_dist = measure_distribution(
        backend, circ.copy(), n_shots, post_select_dict
    )

    expected_dist_vec = np.array(list(expected_postselect_dist.values()))
    actual_dist_vec = np.array(list(actual_postselect_dist.values()))

    np.testing.assert_array_almost_equal(actual_dist_vec, expected_dist_vec, decimal=2)


def test_expectation_from_dist() -> None:
    """Test that the expectation value is correctly calculated from a distribution."""
    dist: dict[tuple[int, ...], float] = {
        (0, 0): 0.25,
        (0, 1): 0.25,
        (1, 0): 0.25,
        (1, 1): 0.25,
    }
    expected_expectation = 0.0
    actual_expectation = expectation_from_dist(dist)
    np.testing.assert_equal(actual_expectation, expected_expectation)

    dist: dict[tuple[int, ...], float] = {(0, 0, 0): 1.0}
    expected_expectation = 1.0
    actual_expectation = expectation_from_dist(dist)
    np.testing.assert_equal(actual_expectation, expected_expectation)

    dist: dict[tuple[int, ...], float] = {(1, 1, 1): 1.0}
    expected_expectation = -1.0
    actual_expectation = expectation_from_dist(dist)
    np.testing.assert_equal(actual_expectation, expected_expectation)


def test_append_pauli_measurement_register() -> None:
    """Test that the measurement register for circuit for a given Pauli string."""
    circ = Circuit(3)
    pauli_string = QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Y])
    expected_circ = Circuit(3)
    expected_circ.H(Qubit(0))
    expected_circ.Rx(0.5, Qubit(1))
    m0 = Bit("cq", 0)
    m1 = Bit("cq", 1)
    expected_circ.add_bit(m0, True)
    expected_circ.add_bit(m1, True)
    expected_circ.Measure(Qubit(0), m0)
    expected_circ.Measure(Qubit(1), m1)

    append_pauli_measurement_register(pauli_string, circ)

    assert circ == expected_circ


def test_pauli_expectation() -> None:
    """Test that the expectation value of a Pauli operator is correctly calculated."""
    backend = AerBackend()
    n_qubits = 3
    circ = Circuit(n_qubits).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)

    # Test case 1: no post-selection
    n_shots = 200000
    pauli = QubitPauliString([Qubit(0)], [Pauli.X])
    expected_expectation = pauli_expectation(backend, circ.copy(), pauli, n_shots)

    qpo = QubitPauliOperator({pauli: 1.0})

    sv = np.array(circ.get_statevector()).T

    mat = qpo.to_sparse_matrix(n_qubits).todense()

    actual_expectation = (sv.conj().T @ mat @ sv)[0, 0]

    np.testing.assert_almost_equal(actual_expectation, expected_expectation, decimal=2)

    # Test case 2: post-select on a single qubit
    post_select_dict = {Qubit(0): 0}

    pauli = QubitPauliString([Qubit(1)], [Pauli.X])
    expected_expectation = pauli_expectation(
        backend, circ.copy(), pauli, n_shots, post_select_dict
    )

    sv = circ.get_statevector()

    sv_postselect = np.array(
        statevector_postselect(circ.qubits, sv, post_select_dict)
    ).T
    sv_postselect = sv_postselect / np.linalg.norm(sv_postselect)

    mat = qpo.to_sparse_matrix(n_qubits - 1).todense()
    actual_expectation = (sv_postselect.conj().T @ mat @ sv_postselect)[0, 0]

    np.testing.assert_almost_equal(actual_expectation, expected_expectation, decimal=2)

    # Test case 3: post-select on a multiple qubits different names
    n_qubits = 3
    circ = Circuit()
    x = circ.add_q_register("x", n_qubits)
    circ = circ.Ry(0.2, x[0]).Ry(0.3, x[1]).Ry(0.7, x[2])
    post_select_dict = {x[1]: 0, x[2]: 0}

    n_shots = 200000
    pauli = QubitPauliString([x[0]], [Pauli.X])
    expected_expectation = pauli_expectation(
        backend, circ.copy(), pauli, n_shots, post_select_dict
    )

    sv = circ.get_statevector()

    sv_postselect = np.array(
        statevector_postselect(circ.qubits, sv, post_select_dict)
    ).T
    sv_postselect = sv_postselect / np.linalg.norm(sv_postselect)

    mat = qpo.to_sparse_matrix().todense()  # type: ignore
    actual_expectation = (sv_postselect.conj().T @ mat @ sv_postselect)[0, 0]  # type: ignore

    np.testing.assert_almost_equal(
        expected_expectation,
        actual_expectation,  # type: ignore
        decimal=2,
    )


def test_operator_expectation() -> None:
    """Test that the expectation value of a Pauli operator is correctly calculated."""
    backend = AerBackend()
    n_qubits = 3
    circ = Circuit(n_qubits).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)

    # Test case 1: no post-selection
    n_shots = 100000

    qpo = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.3,
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.2,
        }
    )
    expected_expectation = operator_expectation(backend, circ.copy(), qpo, n_shots)

    sv = np.array(circ.get_statevector()).T

    mat = qpo.to_sparse_matrix(n_qubits).todense()

    actual_expectation = (sv.conj().T @ mat @ sv)[0, 0]

    np.testing.assert_almost_equal(actual_expectation, expected_expectation, decimal=2)

    # Test case 2: post-select on a single qubit
    post_select_dict = {Qubit(1): 0}

    expected_expectation = operator_expectation(
        backend, circ.copy(), qpo, n_shots, post_select_dict
    )

    sv = circ.get_statevector()

    sv_postselect = np.array(
        statevector_postselect(circ.qubits, sv, post_select_dict)
    ).T
    sv_postselect = sv_postselect / np.linalg.norm(sv_postselect)

    mat = qpo.to_sparse_matrix(n_qubits - 1).todense()
    actual_expectation = (sv_postselect.conj().T @ mat @ sv_postselect)[0, 0]

    np.testing.assert_almost_equal(actual_expectation, expected_expectation, decimal=2)

    # Test case 3: post-select on a multiple qubits different names
    n_qubits = 3
    circ = Circuit()
    x = circ.add_q_register("x", n_qubits)
    circ = circ.Ry(0.2, x[0]).Ry(0.3, x[1]).Ry(0.7, x[2])
    post_select_dict = {x[1]: 0, x[2]: 0}

    n_shots = 100000

    qpo = QubitPauliOperator(
        {
            QubitPauliString([x[0]], [Pauli.Y]): 0.3,
            QubitPauliString([x[0]], [Pauli.X]): 0.2,
        }
    )

    expected_expectation = operator_expectation(
        backend, circ.copy(), qpo, n_shots, post_select_dict
    )

    sv = circ.get_statevector()

    sv_postselect = np.array(
        statevector_postselect(circ.qubits, sv, post_select_dict)
    ).T
    sv_postselect = sv_postselect / np.linalg.norm(sv_postselect)

    mat = qpo.to_sparse_matrix().todense()  # type: ignore
    actual_expectation = (sv_postselect.conj().T @ mat @ sv_postselect)[0, 0]  # type: ignore

    np.testing.assert_almost_equal(
        expected_expectation,
        actual_expectation,  # type: ignore
        decimal=2,
    )
