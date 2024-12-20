"""tests for the measurement utils module."""

import numpy as np
from pytket.circuit import Qubit, QubitRegister
from pytket._tket.circuit import Circuit, Unitary3qBox
from mvsp.measurement.utils import (
    unitary_postselect,
    circuit_statevector_postselect,
    circuit_unitary_postselect,
    # statevector_contract,
)

# Define a 3-qubit unitary (from scipy.stats.unitary_group.rvs(8))
u = np.asarray(
    [
        [
            -0.01087783 - 0.08740685j,
            -0.16434033 + 0.07972852j,
            0.05968107 + 0.1089671j,
            0.15061742 + 0.04792927j,
            -0.01816583 - 0.10716219j,
            0.20625185 - 0.33877756j,
            0.37318756 - 0.43472286j,
            -0.28723982 - 0.58187235j,
        ],
        [
            0.23938051 + 0.08175321j,
            0.06875966 + 0.02545971j,
            0.52634763 - 0.42134468j,
            0.37137253 - 0.1913533j,
            -0.37312204 - 0.13636908j,
            -0.16399068 + 0.03796418j,
            -0.0424525 + 0.01908661j,
            0.2728527 - 0.1968872j,
        ],
        [
            -0.29659442 + 0.51081139j,
            0.29194227 - 0.48782823j,
            -0.08763983 - 0.33187163j,
            0.06548898 - 0.04507687j,
            0.08164181 - 0.26864309j,
            0.23944147 - 0.02749234j,
            0.14558098 - 0.10400333j,
            -0.14806174 + 0.11358353j,
        ],
        [
            0.32200994 + 0.16414724j,
            0.09510586 + 0.41696043j,
            -0.3190567 - 0.2663581j,
            0.43721077 + 0.40831071j,
            0.23944126 + 0.01045236j,
            0.2092303 + 0.14725049j,
            -0.15121202 - 0.07665942j,
            -0.05686498 + 0.0311881j,
        ],
        [
            0.25521052 + 0.36825342j,
            0.51097114 + 0.20281874j,
            0.05705804 + 0.09902171j,
            -0.42228849 + 0.06195022j,
            -0.11042025 + 0.19363511j,
            0.0762851 - 0.39579801j,
            -0.10478475 + 0.15609j,
            0.03205629 - 0.23080195j,
        ],
        [
            0.18589958 + 0.0486647j,
            -0.258725 + 0.0832211j,
            0.19482307 - 0.02672288j,
            -0.10140543 - 0.06745429j,
            0.0168671 - 0.3493762j,
            0.22900581 + 0.08194494j,
            -0.02878317 + 0.5946357j,
            -0.54613111 - 0.03897807j,
        ],
        [
            0.30550931 + 0.13014332j,
            0.08729782 + 0.22094503j,
            -0.01990302 - 0.01477675j,
            -0.26562384 - 0.35354902j,
            0.43974443 - 0.33809975j,
            -0.34981781 + 0.2305434j,
            0.31585617 - 0.22725712j,
            0.04698062 + 0.01715475j,
        ],
        [
            0.17493009 - 0.26491546j,
            -0.11226858 - 0.10009274j,
            0.33767585 - 0.27029117j,
            -0.1719355 - 0.11907399j,
            0.45302483 + 0.095605j,
            0.44911039 - 0.30323364j,
            -0.19603031 - 0.17095465j,
            0.1589249 + 0.21175611j,
        ],
    ]
)

u_postselected = np.asarray(
    [
        [
            -0.29659442 + 0.51081139j,
            0.29194227 - 0.48782823j,
            0.08164181 - 0.26864309j,
            0.23944147 - 0.02749234j,
        ],
        [
            0.32200994 + 0.16414724j,
            0.09510586 + 0.41696043j,
            0.23944126 + 0.01045236j,
            0.2092303 + 0.14725049j,
        ],
        [
            0.30550931 + 0.13014332j,
            0.08729782 + 0.22094503j,
            0.43974443 - 0.33809975j,
            -0.34981781 + 0.2305434j,
        ],
        [
            0.17493009 - 0.26491546j,
            -0.11226858 - 0.10009274j,
            0.45302483 + 0.095605j,
            0.44911039 - 0.30323364j,
        ],
    ]
)


def test_circuit_statevector_postselect() -> None:
    """Test that the statevector is post selected correctly."""
    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    sv = circ.get_statevector()

    n_state_qubits = 2

    sv_post_select = sv[: 2**n_state_qubits]

    post_select_dict = {Qubit(0): 0}

    sv_postselect = circuit_statevector_postselect(circ, post_select_dict)

    np.testing.assert_array_equal(sv_postselect, sv_post_select)

    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    sv = circ.get_statevector()

    n_state_qubits = 1

    sv_post_select = sv[: 2**n_state_qubits]

    post_select_dict = {Qubit(0): 0, Qubit(1): 0}

    sv_postselect = circuit_statevector_postselect(circ, post_select_dict)

    np.testing.assert_array_equal(sv_postselect, sv_post_select)


def test_circuit_unitary_postselect():
    """Test that the unitary is post selected correctly."""
    circ = Circuit(3).Ry(0.1, 0).Ry(0.2, 1).Ry(0.2, 2)
    unitary = circ.get_unitary()

    n_state_qubits = 2

    unitary_post_select = unitary[: 2**n_state_qubits, : 2**n_state_qubits]

    post_select_dict = {Qubit(0): 0}

    unitary_postselect = circuit_unitary_postselect(circ, post_select_dict)

    np.testing.assert_array_equal(unitary_postselect, unitary_post_select)

    unitary = circ.get_unitary()

    n_state_qubits = 1

    unitary_post_select = unitary[: 2**n_state_qubits, : 2**n_state_qubits]

    post_select_dict = {Qubit(0): 0, Qubit(1): 0}
    pre_select_dict = {Qubit(0): 0, Qubit(1): 0}

    unitary_postselect = circuit_unitary_postselect(
        circ, post_select_dict, pre_select_dict
    )

    np.testing.assert_array_equal(unitary_postselect, unitary_post_select)

    circ = Circuit(2).ISWAP(0.25, 0, 1)
    unitary = circ.get_unitary()

    unitary_post_select = unitary[2:, 2:]

    post_select_dict = {Qubit(0): 1}
    pre_select_dict = {Qubit(0): 1}

    unitary_postselect = circuit_unitary_postselect(
        circ, post_select_dict, pre_select_dict
    )

    np.testing.assert_array_equal(unitary_postselect, unitary_post_select)


def test_statevector_postselect() -> None:
    """Alternative test for statevector postselection."""
    circ = Circuit(3).add_unitary3qbox(Unitary3qBox(u), 0, 1, 2)
    sv = circ.get_statevector()

    post_select_dict = {Qubit(0): 0, Qubit(2): 1}
    psv = circuit_statevector_postselect(circ, post_select_dict)
    np.testing.assert_allclose(sv[1:4:2], psv)

    post_select_dict = {Qubit(1): 1, Qubit(0): 1}
    psv = circuit_statevector_postselect(circ, post_select_dict)
    np.testing.assert_allclose(sv[6:8], psv)

    post_select_dict = {Qubit(1): 0}
    psv = circuit_statevector_postselect(circ, post_select_dict)
    np.testing.assert_allclose(np.concatenate((sv[0:2], sv[4:6])), psv)

    post_select_dict = {Qubit(2): 0, Qubit(1): 0}
    psv = circuit_statevector_postselect(circ, post_select_dict)
    np.testing.assert_allclose(np.concatenate((sv[0:1], sv[4:5])), psv)

    post_select_dict = {Qubit(0): 1}
    psv = circuit_statevector_postselect(circ, post_select_dict)
    np.testing.assert_allclose(sv[4:8], psv)


def test_unitary_postselect() -> None:
    """Alternative test for unitary postselection."""
    qreg = QubitRegister("q", 3)
    qlist = [qreg[i] for i in range(qreg.size)]

    post_select_dict = {qreg[0]: 0, qreg[1]: 0}
    pu = unitary_postselect(qlist.copy(), u, post_select_dict)
    np.testing.assert_array_equal(pu, u[0:2, 0:2])

    post_select_dict = {qreg[0]: 0, qreg[1]: 1}
    pre_select_dict = {qreg[0]: 0, qreg[1]: 0}
    pu = unitary_postselect(qlist.copy(), u, post_select_dict, pre_select_dict)
    np.testing.assert_array_equal(pu, u[2:4, 0:2])

    post_select_dict = {qreg[1]: 0, qreg[0]: 1}
    pu = unitary_postselect(qlist.copy(), u, post_select_dict)
    np.testing.assert_array_equal(pu, u[4:6, 0:2])

    post_select_dict = {qreg[1]: 1, qreg[0]: 1}
    pu = unitary_postselect(qlist.copy(), u, post_select_dict)
    np.testing.assert_array_equal(pu, u[6:8, 0:2])

    post_select_dict = {qreg[1]: 1}
    pre_select_dict = {qreg[1]: 0}
    pu = unitary_postselect(qlist.copy(), u, post_select_dict, pre_select_dict)
    np.testing.assert_array_equal(pu, u_postselected)

    post_select_dict = {qreg[2]: 1, qreg[1]: 1}
    pre_select_dict = {qreg[2]: 0, qreg[1]: 0}
    pu = unitary_postselect(qlist.copy(), u, post_select_dict, pre_select_dict)
    np.testing.assert_array_equal(pu, u[3:8:4, 0:5:4])

    qreg = QubitRegister("q", 2)
    qlist = [qreg[i] for i in range(qreg.size)]

    post_select_dict = {qreg[0]: 1}
    pre_select_dict = {qreg[0]: 1}
    pu = unitary_postselect(
        qlist.copy(), u_postselected, post_select_dict, pre_select_dict
    )
    np.testing.assert_array_equal(pu, u_postselected[2:, 2:])

    post_select_dict = {qreg[0]: 0}
    pre_select_dict = {qreg[0]: 1}
    pu = unitary_postselect(
        qlist.copy(), u_postselected, post_select_dict, pre_select_dict
    )
    np.testing.assert_array_equal(pu, u_postselected[:2, 2:])


# @pytest.mark.parametrize("n_ps_qubits", [1, 2, 3, 4, 5, 6])
# def test_recursive_statevector_trace(n_ps_qubits: int):
#     """Test recursive statevector trace function."""
#     # Define a 3-qubit statevector
#     n_qubits = 7
#     sv = np.ones(2**n_qubits, dtype=np.complex128)
#     sv = sv / np.linalg.norm(sv)

#     # Define a list of qubits
#     qlist = [Qubit(i) for i in range(n_qubits)]

#     # Define a list of qubits to trace out
#     trace_qubits = qlist[:n_ps_qubits]

#     # Compute the expected post-selected statevector
#     expected_sv = np.ones(2 ** (n_qubits - n_ps_qubits), dtype=np.complex128)
#     expected_sv = expected_sv / np.linalg.norm(expected_sv)

#     # Compute the actual post-selected statevector
#     actual_sv = statevector_contract(qlist, sv, trace_qubits)

#     # Check that the actual and expected statevectors are equal
#     np.testing.assert_allclose(actual_sv, expected_sv)
