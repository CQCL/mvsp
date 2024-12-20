"""Phase estimation module."""

from pytket.circuit import Qubit, QubitRegister
from pytket._tket.circuit import Circuit
from pytket.backends.backend import Backend
from qtmlib.measurement.shots import measure_distribution
from qtmlib.measurement.utils import dist_to_fixed_point
import numpy as np


def measure_phase_estimation(
    circ: Circuit,
    backend: Backend,
    ancilla_qreg: QubitRegister,
    n_shots: int | None,
    post_select: dict[Qubit, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Measure the circuit for phase estimation.

    Post selection can be done by passing a dictionary of qubit
    and post select outcome. Measurement gates are appended to the circuit and
    then filtered by post selection. The ancilla register which contains
    the bit strings is passed and measurement gates are appended.
    These are returned as a dictionary of bit strings and probabilities.

    If n_shots is not None, then shots are used to calculate the distribution.

    Args:
    ----
        circ (Circuit): circuit to be measured
        backend (Backend): backend to run the circuit
        ancilla_qreg (QubitRegister): ancilla qubits
        n_shots (int | None): number of shots
        post_select (dict[Qubit, int] |None): dictionary of qubit and post select

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of shots distribution

    """
    if n_shots is None:
        raise NotImplementedError("Statevector phase estimation not implemented")
    else:
        return _measure_phase_estimation_shots(
            circ, backend, ancilla_qreg, n_shots, post_select
        )


def _measure_phase_estimation_shots(
    circ: Circuit,
    backend: Backend,
    ancilla_qreg: QubitRegister,
    n_shots: int,
    post_select_dict: dict[Qubit, int] | None = None,
) -> dict[tuple[int, ...], float]:
    """Measure the circuit with phase estimation using shots.

    Post select can be done by passing a dictionary of qubit and post select
    value.

    Args:
    ----
        backend (Backend): backend to run the circuit
        circ (Circuit): circuit to be measured
        ancilla_qreg (QubitRegister): ancilla qubits
        n_shots (int): number of shots
        post_select_dict (dict[Qubit,int] |None): dictionary of qubit and post select

    Returns:
    -------
        dict[tuple[int, ...], float]: dictionary of shots distribution

    """
    circ.measure_register(ancilla_qreg, f"c{ancilla_qreg.name}")
    return measure_distribution(backend, circ, n_shots, post_select_dict)


def phase_estimation_results(dist: dict[tuple[int, ...], float], positive: bool = True):
    """Convert distribution to fixed point binary decimal number.

    Args:
    ----
        dist (dict[tuple[int, ...], float]): distribution
        positive (bool): whether the phase is positive or negative

    Returns:
    -------
        dict[float, float]: fixed point distribution

    """
    sign = 1 if positive else -1
    return {decimal * sign: prob for decimal, prob in dist_to_fixed_point(dist).items()}


def energy_timevo_qpe(phase: float, total_time: float):
    """Convert phase to energy for time evolution phase estimation.

    The readout of pytket phase estimation is in the range 2pi * [0,1].
    Hence there is a multiple of 2pi. The energy is then calculated by
    E = 2pi * phase / (pi * 0.5 * total_time).

    Args:
    ----
        phase (float): phase
        total_time (float): total time

    Returns:
    -------
        float: energy

    """
    return (2 * np.pi * phase) / (np.pi * 0.5 * total_time)


def energy_qubitised_qpe(phase: float, l1_norm: float):
    """Convert phase to energy from qubitised phase estimation.

    The readout of the pytket phase estimation is in the range 2pi * [0,1].
    Hence there is a multiple of 2pi. The energy is then calculated by
    E = l1_norm * cos(phase*2pi).

    Args:
    ----
        phase (float): phase
        l1_norm (float): l1 norm

    Returns:
    -------
        float: energy

    """
    return l1_norm * np.cos(phase * 2 * np.pi)


def largest_values_dict(d: dict[float, float], n: int) -> dict[float, float]:
    """Return the n largest values from a dictionary.

    Args:
    ----
        d (dict): dictionary
        n (int): number of largest values

    Returns:
    -------
        dict: dictionary of n largest values

    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])


def process_timeevo_qpe_results(
    dist: dict[tuple[int, ...], float], total_time: float, n: int, positive: bool = True
):
    """Process the results of time evolution phase estimation.

    Returns the n largest values of the energy distribution.
    The readout off the pytket phase estimation is in the range 2pi * [0,1].
    Hence there is a multiple of 2pi. The energy is then calculated by
    E = 2pi * phase / (pi * 0.5 * total_time).

    Args:
    ----
        dist (dict[float, float]): distribution
        total_time (float): total time
        n (int): number of largest values
        positive (bool): whether the phase is positive or negative

    Returns:
    -------
        dict[float, float]: dictionary of n largest energy values
            with their probabilities

    """
    fixedpoint_dist = phase_estimation_results(dist, positive)
    energy_dist = {
        energy_timevo_qpe(phase, total_time): prob
        for phase, prob in fixedpoint_dist.items()
    }
    return largest_values_dict(energy_dist, n)


def process_qubitised_qpe_results(
    dist: dict[tuple[int, ...], float], l1_norm: float, n: int, positive: bool = True
):
    """Process the results of qubitised phase estimation.

    Returns the n largest values of the energy distribution.
    The readout of the pytket phase estimation is in the range 2pi * [0,1].
    Hence there is a multiple of 2pi. The energy is then calculated by
    E = l1_norm * cos(phase*2pi).

    Args:
    ----
        dist (dict[float, float]): distribution
        l1_norm (float): l1 norm
        n (int): number of largest values
        positive (bool): whether the phase is positive or negative

    Returns:
    -------
        dict[float, float]: dictionary of n largest energy values
            with their probabilities

    """
    fixedpoint_dist = phase_estimation_results(dist, positive)
    energy_dist = {
        energy_qubitised_qpe(phase, l1_norm): prob
        for phase, prob in fixedpoint_dist.items()
    }
    return largest_values_dict(energy_dist, n)
