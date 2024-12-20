"""Conftest file for test parameterisation with fixture."""

import pytest
from pytest_lazyfixture import lazy_fixture
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit
from typing import Any


@pytest.fixture()
def ham_1q_posreal_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.1,
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.4,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posreal_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.3,
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.2,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posreal_2() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Z]): 0.5,
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posreal_3() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.5,
            QubitPauliString([Qubit(0)], [Pauli.Z]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posreal_4() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.I]): 0.3,
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.2,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posreal_5() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(): 0.5,
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posreal_6() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.5,
            QubitPauliString([Qubit(0)], [Pauli.I]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_negreal_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.X]): -0.1,
            QubitPauliString([Qubit(0)], [Pauli.Z]): -0.4,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_posimaginary_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.X]): 0.1j,
            QubitPauliString([Qubit(0)], [Pauli.Y]): 0.4j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_1q_negimaginary_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0)], [Pauli.Y]): -0.3j,
            QubitPauliString([Qubit(0)], [Pauli.X]): -0.2j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posreal_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.X]): 0.5,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posreal_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Y, Pauli.Z]): 0.3,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.X]): 0.2,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posreal_2() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Y, Pauli.Y]): 0.5,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posreal_3() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(): 0.5,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posreal_4() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.I, Pauli.I]): 0.5,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): 0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_negreal_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.X]): -0.5,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): -0.1,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_negreal_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Y, Pauli.Z]): -0.3,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.X]): -0.2,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posimaginary_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.X]): 0.5j,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): 0.1j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_posimaginary_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Y, Pauli.Z]): 0.3j,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.X]): 0.2j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_negimaginary_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.X]): -0.5j,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): -0.1j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_negimaginary_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Y, Pauli.Z]): -0.3j,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.X]): -0.2j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_mixed_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.X]): 0.7,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.Z]): -0.9j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_2q_mixed_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Y, Pauli.Z]): 0.7,
            QubitPauliString([Qubit(0), Qubit(1)], [Pauli.X, Pauli.X]): -0.8j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_posreal_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.X, Pauli.Y]
            ): 0.5,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.Z]
            ): 0.1,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.X]
            ): 0.2,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Y]
            ): 0.3,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_posreal_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Z, Pauli.X]
            ): 0.5,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Z]
            ): 0.1,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.Y]
            ): 0.2,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.X]
            ): 0.3,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_negreal_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.X, Pauli.Y]
            ): -0.5,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.Z]
            ): -0.1,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.X]
            ): -0.2,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Y]
            ): -0.3,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_negreal_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Z, Pauli.X]
            ): -0.5,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Z]
            ): -0.1,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.Y]
            ): -0.2,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.X]
            ): -0.3,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_posimaginary_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.X, Pauli.Y]
            ): 0.5j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.Z]
            ): 0.1j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.X]
            ): 0.2j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Y]
            ): 0.3j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_posimaginary_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Z, Pauli.X]
            ): 0.5j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Z]
            ): 0.1j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.Y]
            ): 0.2j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.X]
            ): 0.3j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_negimaginary_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.X, Pauli.Y]
            ): -0.5j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.Z]
            ): -0.1j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.X]
            ): -0.2j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Y]
            ): -0.3j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_negimaginary_1() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Z, Pauli.X]
            ): -0.5j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Z]
            ): -0.1j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.Y]
            ): -0.2j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.X]
            ): -0.3j,
        }
    )
    return hamiltonian


@pytest.fixture()
def ham_3q_mixed_0() -> QubitPauliOperator:
    hamiltonian = QubitPauliOperator(
        {
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Z, Pauli.X, Pauli.Y]
            ): 0.5,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.Z, Pauli.Z]
            ): 0.1,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.Y, Pauli.Y, Pauli.X]
            ): 0.7j,
            QubitPauliString(
                [Qubit(0), Qubit(1), Qubit(2)], [Pauli.X, Pauli.X, Pauli.Y]
            ): -0.6j,
        }
    )
    return hamiltonian


@pytest.fixture(
    params=[
        lazy_fixture("ham_1q_posreal_0"),
        lazy_fixture("ham_1q_posreal_1"),
        lazy_fixture("ham_1q_posreal_2"),
        lazy_fixture("ham_1q_posreal_3"),
        lazy_fixture("ham_1q_posreal_4"),
        lazy_fixture("ham_1q_posreal_5"),
        lazy_fixture("ham_1q_posreal_6"),
        lazy_fixture("ham_1q_negreal_0"),
        lazy_fixture("ham_2q_posreal_0"),
        lazy_fixture("ham_2q_posreal_1"),
        lazy_fixture("ham_2q_posreal_2"),
        lazy_fixture("ham_2q_posreal_3"),
        lazy_fixture("ham_2q_posreal_4"),
        lazy_fixture("ham_2q_negreal_0"),
        lazy_fixture("ham_2q_negreal_1"),
        lazy_fixture("ham_3q_posreal_0"),
        lazy_fixture("ham_3q_posreal_1"),
        lazy_fixture("ham_3q_negreal_0"),
        lazy_fixture("ham_3q_negreal_1"),
    ]
)
def op_hermitian_fixture(request: Any) -> QubitPauliOperator:
    """Fixture for parameterising tests with different hermitian operators."""
    return request.param


@pytest.fixture(
    params=[
        lazy_fixture("ham_1q_posimaginary_0"),
        lazy_fixture("ham_1q_negimaginary_1"),
        lazy_fixture("ham_2q_posimaginary_0"),
        lazy_fixture("ham_2q_posimaginary_1"),
        lazy_fixture("ham_2q_negimaginary_0"),
        lazy_fixture("ham_2q_negimaginary_1"),
        lazy_fixture("ham_2q_mixed_0"),
        lazy_fixture("ham_2q_mixed_1"),
        lazy_fixture("ham_3q_posimaginary_0"),
        lazy_fixture("ham_3q_posimaginary_1"),
        lazy_fixture("ham_3q_negimaginary_0"),
        lazy_fixture("ham_3q_negimaginary_1"),
        lazy_fixture("ham_3q_mixed_0"),
    ]
)
def op_nonhermitian_fixture(request: Any) -> QubitPauliOperator:
    """Fixture for parameterising tests with different nonhermitian operators."""
    return request.param


@pytest.fixture(
    params=[
        lazy_fixture("op_hermitian_fixture"),
        lazy_fixture("op_nonhermitian_fixture"),
    ]
)
def op_fixture(request: Any) -> QubitPauliOperator:
    """Fixture for parameterising tests with different operators."""
    return request.param
