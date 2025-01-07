"""Plane wave Hamiltonian for a ND system."""

import multiprocessing

import numpy as np
from numpy.typing import NDArray


def kinetic(
    k_points: NDArray[np.int32],
    hbar: float = 1.0,
    m: float = 1.0,
) -> NDArray[np.complex128]:
    """Return the diagonal kinetic energy matrix for a ND plane wave system.

    Args:
    ----
        k_points: The k points of the system.
        hbar: The reduced Planck constant.
        m: The mass of the particle.

    Returns:
    -------
        The digonal kinetic energy matrix.

    """
    return np.diag(
        [(((hbar * np.pi) ** 2) * np.linalg.norm(k) ** 2) / (2 * m) for k in k_points]
    )


def elec_nuc_potential_element(
    args: tuple[
        int,
        int,
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.float64],
        float,
    ],
) -> tuple[int, int, complex]:
    """Return the matrix element of the electron-nucleus potential.

    This functions is parallelized and should be used with the multiprocessing
    module. Where the each matrix element will be calculated in parallel.

    Args:
    ----
        args: A tuple containing the following elements:
            i: The row index of the matrix element.
            j: The column index of the matrix element.
            k_bra: The bra k point.
            k_ket: The ket k point.
            r_pos: The position of the nucleus.
            cell_area: The area of the cell.

    Returns:
    -------
        The matrix element of the electron-nucleus potential.

    """
    i, j, k_bra, k_ket, r_pos, cell_area = args
    mat_element = 0
    for r in r_pos:
        if np.array_equal(k_bra, k_ket):
            mat_element = 0
        else:
            mat_element += (
                4 / (np.pi * cell_area * np.linalg.norm(k_bra - k_ket) ** 2)
            ) * np.exp(1j * np.pi * np.dot(k_ket - k_bra, r))
    return i, j, mat_element


def elec_nuc_potential(
    k_points: NDArray[np.int32],
    cell_area: float,
    r_pos: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return the electron-nucleus potential matrix for a ND plane wave system.

    This function is parallelized and should be used with the multiprocessing.

    Args:
    ----
        k_points: The k points of the system.
        cell_area: The area of the cell.
        r_pos: The position of the nucleus.

    Returns:
    -------
        The electron-nucleus potential matrix.

    """
    mat = np.zeros((len(k_points), len(k_points)), dtype=complex)
    args_list: list[
        tuple[
            tuple[
                int,
                int,
                NDArray[np.int32],
                NDArray[np.int32],
                NDArray[np.float64],
                float,
            ]
        ]
    ] = []
    for i, k_bra in enumerate(k_points):
        for j, k_ket in enumerate(k_points):
            args_list.append((i, j, k_bra, k_ket, r_pos, cell_area))

    with multiprocessing.Pool() as pool:
        results = pool.map(elec_nuc_potential_element, args_list)

    for i, j, mat_element in results:
        mat[i, j] = mat_element

    return mat


def plane_wave_hamiltonian(
    k_points: NDArray[np.int32],
    cell_area: float,
    r_pos: NDArray[np.float64],
    hbar: float = 1.0,
    m: float = 1.0,
) -> NDArray[np.complex128]:
    """Return the Hamiltonian matrix for a ND plane wave system.

    This function is parallelized and should be used with the multiprocessing.

    Args:
    ----
        k_points: The k points of the system.
        cell_area: The area of the cell.
        r_pos: The position of the nucleus.
        hbar: The reduced Planck constant.
        m: The mass of the particle.

    Returns:
    -------
    The Hamiltonian matrix.

    """
    return -kinetic(k_points, hbar, m) - elec_nuc_potential(k_points, cell_area, r_pos)
