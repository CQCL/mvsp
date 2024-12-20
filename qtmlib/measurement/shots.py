"""Functions for measuring circuits and operators by sampling from a backend."""

from pytket.circuit import Qubit
from pytket._tket.circuit import Circuit
from pytket._tket.unit_id import Bit
from pytket.backends.backend import Backend
from pytket.pauli import QubitPauliString, Pauli
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.distribution import ProbabilityDistribution


def add_measure_post_select(
    circ: Circuit, post_select_dict: dict[Qubit, int]
) -> tuple[dict[int, int], Circuit]:
    """Add measurement to the circuit on post select qubits.

    Returb the dictionary of post select bit index. As the keys in the shot distribution
    are numerically indexed.

    Args:
    ----
        circ (Circuit): circuit to be measured
        post_select_dict (dict[Qubit,int]): dictionary of qubit and post select value

    Returns:
    -------
        tuple[dict[int,int], Circuit]: dictionary of post select bit index and circuit

    """
    postselect_bits: dict[Bit, int] = {}
    for q, b in post_select_dict.items():
        m = Bit(f"c{q.reg_name}", q.index[0])
        postselect_bits[m] = b
        circ.add_bit(m, True)
        circ.Measure(q, m)
    postselect_bit_ind = {
        i: postselect_bits[cbit]
        for i, cbit in enumerate(circ.bits)
        if cbit in list(postselect_bits.keys())
    }
    return postselect_bit_ind, circ


def get_shots_distribution(
    backend: Backend, circ: Circuit, n_shots: int
) -> dict[tuple[int, ...], float]:
    """Get the distribution of shots from the backend.

    Args:
    ----
        backend (Backend): backend to run the circuit
        circ (Circuit): circuit to be measured
        n_shots (int): number of shots

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of shots distribution

    """
    if not backend.valid_circuit(circ):
        circ = backend.get_compiled_circuit(circ)
    handle = backend.process_circuit(circ, n_shots)
    emp_dict = backend.get_result(handle).get_empirical_distribution()
    dist: dict[tuple[int, ...], float] = (
        ProbabilityDistribution.from_empirical_distribution(emp_dict).as_dict()  # type: ignore
    )
    return dist


def post_select_distribution(
    dist: dict[tuple[int, ...], float], postselect_bit_ind: dict[int, int]
) -> dict[tuple[int, ...], float]:
    """Post select the distribution.

    Args:
    ----
        dist (dict[tuple[int, ...], float]): dictionary of shots distribution
        postselect_bit_ind (dict[int, int]): dictionary of post select bit index

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of post select shots distribution

    """
    postselect_ind = list(postselect_bit_ind.keys())

    list_all_ind = list(range(len(next(iter(dist.keys())))))

    measure_ind = [i for i in list_all_ind if i not in postselect_ind]

    post_select_on = tuple(postselect_bit_ind.values())

    post_select_dist = dist.copy()
    for k in dist.keys():
        post_select_result = tuple([k[i] for i in postselect_ind])

        if post_select_result != post_select_on:
            post_select_dist.pop(k)

    renorm_factor = sum(list(post_select_dist.values()))

    post_select_dist_renorm = {
        tuple([k[i] for i in measure_ind]): v / renorm_factor
        for k, v in post_select_dist.items()
    }
    return post_select_dist_renorm


def measure_distribution(
    backend: Backend,
    circ: Circuit,
    n_shots: int,
    post_select_dict: dict[Qubit, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Measure the circuit and get distribution.

    Post select on the can be done by passing a dictionary of qubit and post select
    value.

    Args:
    ----
        backend (Backend): backend to run the circuit
        circ (Circuit): circuit to be measured
        n_shots (int): number of shots
        post_select_dict (dict[Qubit,int] |None): dictionary of qubit and post select
            value

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of shots distribution

    """
    if post_select_dict is not None:
        post_select_dict_ind, circ = add_measure_post_select(circ, post_select_dict)
        dist = get_shots_distribution(backend, circ, n_shots)
        dist = post_select_distribution(dist, post_select_dict_ind)
    else:
        dist = get_shots_distribution(backend, circ, n_shots)
    return dist


def measure_pauli_distribution(
    backend: Backend,
    circ: Circuit,
    pauli: QubitPauliString,
    n_shots: int,
    post_select_dict: dict[Qubit, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Measure the circuit appending paulis and get distribution.

    Post select on the can be done by passing a dictionary of qubit and post select
    value.

    Args:
    ----
        backend (Backend): backend to run the circuit
        circ (Circuit): circuit to be measured
        pauli (QubitPauliString): pauli to be measured
        n_shots (int): number of shots
        post_select_dict (dict[Qubit,int] |None): dictionary of qubit and post select
            value

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of shots distribution

    """
    measured_circ = circ.copy()
    append_pauli_measurement_register(pauli, measured_circ)
    return measure_distribution(backend, measured_circ, n_shots, post_select_dict)


def append_pauli_measurement_register(
    pauli_string: QubitPauliString, circ: Circuit
) -> None:
    """Append pauli measurements to the circuit inplace.

    Qubits rotationed in to the Z basis and measured.

    Args:
    ----
        pauli_string (QubitPauliString): pauli to be measured
        circ (Circuit): circuit to be measured

    Returns:
    -------
        None: in place circuit addition

    """
    measured_qbs: list[Qubit] = []
    for qb, p in pauli_string.map.items():
        if p == Pauli.I:
            continue
        measured_qbs.append(qb)
        if p == Pauli.X:
            circ.H(qb)
        elif p == Pauli.Y:
            circ.Rx(0.5, qb)
    for q in measured_qbs:
        m = Bit(f"c{q.reg_name}", q.index[0])
        circ.add_bit(m, True)
        circ.Measure(q, m)


def expectation_from_dist(dist: dict[tuple[int, ...], float]) -> float:
    """Get expectation value from the distribution.

    Args:
    ----
        dist (dict[tuple[int, ...], float]): dictionary of shots distribution

    Returns:
    -------
        float: expectation value

    """
    aritysum = 0.0
    for row, prob in dist.items():
        aritysum += prob * (sum(row) % 2)
    return -2 * aritysum + 1


def pauli_expectation(
    backend: Backend,
    circ: Circuit,
    pauli: QubitPauliString,
    n_shots: int,
    post_select_dict: dict[Qubit, int] | None = None,
) -> float:
    """Get pauli expectation value from the distribution.

    Post select on the can be done by passing a dictionary of qubit and post select
    value.

    Args:
    ----
        backend (Backend): backend to run the circuit
        circ (Circuit): circuit to be measured
        pauli (QubitPauliString): pauli to be measured
        n_shots (int): number of shots
        post_select_dict (dict[Qubit,int] |None): dictionary of qubit and post select
            value

    Returns:
    -------
        float: pauli expectation value

    """
    dist = measure_pauli_distribution(backend, circ, pauli, n_shots, post_select_dict)
    return expectation_from_dist(dist)


def operator_expectation(
    backend: Backend,
    circ: Circuit,
    qpo: QubitPauliOperator,
    n_shots: int,
    post_select_dict: dict[Qubit, int] | None = None,
):
    """Get operator expectation value from the distribution.

    Post select on the can be done by passing a dictionary of qubit and post select
    value.

    Args:
    ----
        backend (Backend): backend to run the circuit
        circ (Circuit): circuit to be measured
        qpo (QubitPauliOperator): operator to be measured
        n_shots (int): number of shots
        post_select_dict (dict[Qubit,int] |None): dictionary of qubit and post select
            value

    Returns:
    -------
        float: operator expectation value

    """
    expectation_value: complex
    id_string = QubitPauliString()
    if id_string in qpo._dict:
        expectation_value = complex(qpo[id_string])  # type: ignore
    else:
        expectation_value = 0
    operator_without_id = QubitPauliOperator(
        {p: c for p, c in qpo._dict.items() if (p != id_string)}
    )
    for pauli, coeff in operator_without_id._dict.items():
        expectation_value += complex(coeff) * pauli_expectation(
            backend, circ, pauli, n_shots, post_select_dict
        )
    return expectation_value
