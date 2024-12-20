"""utils functions for QSVT."""

import numpy as np
from scipy.linalg import svd
from numpy.polynomial.polynomial import Polynomial
from sympy import Symbol, acos
from sympy.core.expr import Expr
from pytket._tket.circuit import Circuit
from numpy.typing import NDArray
from collections.abc import Sequence
from pandas.core.frame import DataFrame


def scipy_qsvt(
    operator: NDArray[np.complex128], polynomial: Polynomial
) -> NDArray[np.complex128]:
    """Scipy implementation of QSVT.

    Polynomial transform of the singular values, where the SVD
    is performed using scipy.linalg.svd. For even polynomial,
    the right singular vectors are used, and for odd polynomial,
    the left singular vectors are used.

    Args:
    ----
        operator (npt.ndarray): matrix operator
        polynomial (Polynomial): polynomial to be applied to
        the singular values of the matrix

    """
    U, s, Vh = svd(operator, full_matrices=True)

    if (len(polynomial) - 1) % 2 == 0:
        # even polynomial
        # ∑_{k} Poly(s_k)|vk> <vk|
        # ONLY USES RIGHT SINGULAR VECS |vk>!
        qsvt = Vh.conj().T @ np.diag(polynomial(s)) @ Vh  # type:ignore

    else:
        # odd polynomial
        # ∑_{k} Poly(s_k)|uk> <vk|
        qsvt = U @ np.diag(polynomial(s)) @ Vh  # type: ignore

    return qsvt


def qsp_phase_reflection(phi_list: Sequence[float | Expr]) -> Sequence[float | Expr]:
    """QSP reflection phase convention conversion from pyqsp.

    Converts a list of qsp phases to radians for use in pytket
    in the convention used in Appendix A2 of
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040203

    Args:
    ----
        phi_list (list[float]): The list of phases to be converted

    Returns:
    -------
        np.array: The converted phases

    """
    phi_array = np.array(phi_list)
    d = len(phi_list) - 1  # poly is a d+1 approximation!!!
    phi_1 = phi_array[0] + phi_array[-1] + (d - 1) * (np.pi / 2)
    phi_2_d = phi_array[1:-1] - np.pi / 2
    new_phi = [phi_1, *phi_2_d]
    phi_list_rev = new_phi[::-1]

    return (-2 * np.array(phi_list_rev) / np.pi).tolist()


def single_qubit_qsp_circuit(
    phi_list: list[float], apply_QSP_hadamards: bool = True
) -> Circuit:
    r"""Build symbolic pytket circuit QSP circuit.

    From input phi list Used for debugging qsp conventions an
    fitting qsp phases to a polynomial.

    𝑈𝜙⃗ = 𝑒^{𝑖𝜙_{0}𝑍} \Prod_{k=1}^{d} 𝑊(𝑎)⋅𝑒^{+𝑖𝜙_{𝑘}𝑍}

    0:  ────Rz(𝜙_k)────W(a)────Rz(𝜙_k-1)────W(a)────Rz(𝜙_k-2)──── ... ────Rz(𝜙_0)────

    Args:
    ----
        phi_list (list): list of phi angles that approximate the desired polynomial
        apply_QSP_hadamards (bool, optional): Apply hadamards to the circuit. Defaults
        to True.

    """
    reverse_phi = phi_list[::-1]

    circuit = Circuit(1, "U_phi")

    if apply_QSP_hadamards is True:
        circuit.H(0)

    theta = -2 * acos(Symbol("s"))  # type: ignore
    for phi in reverse_phi[:-1]:
        # e^{Z}
        circuit.Rz(-2 * phi / np.pi, 0)
        # W(a)
        circuit.Rx((-1.0 * theta) / np.pi, 0)  # type: ignore

    circuit.Rz(-2 * phi_list[0] / np.pi, 0)

    if apply_QSP_hadamards is True:
        circuit.H(0)

    return circuit


def measure_single_qubit_qsp(circ: Circuit) -> DataFrame:
    """Measure single qubit QSP circuit for a between -1 and 1.

    Args:
    ----
        circ (Circuit): Single qubit QSP circuit

    Returns:
    -------
        DataFrame: Dataframe of measurement probabilities

    """
    qsp_dict: dict[np.float64, dict[int, np.complex128]] = {}
    for te in np.linspace(-1, 1, 1000, dtype=np.float64):
        new_circ = circ.copy()
        new_circ.symbol_substitution({Symbol("s"): te})
        u = new_circ.get_unitary()
        qsp_dict[te] = {0: u[0, 0], 1: u[1, 1]}
    return DataFrame.from_dict(data=qsp_dict, orient="index")
