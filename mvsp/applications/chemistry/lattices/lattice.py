"""Lattice module for Chemical Systems."""

import numpy as np
from numpy.typing import NDArray


def center(L: float) -> list[NDArray[np.float64]]:
    """Generate the position for a simple cubic (SC) lattice center.

    Parameters
    ----------
    L (float): The lattice constant.

    Returns
    -------
    list: A list containing the position of the center.

    """
    return [np.array([L / 2, L / 2, L / 2])]


def sc(L: float) -> list[NDArray[np.float64]]:
    """Generate the positions for a simple cubic (SC) lattice.

    Parameters
    ----------
    L (float): The lattice constant.

    Returns
    -------
    list: A list of numpy arrays representing the positions of the SC lattice points.

    """
    return [
        np.array([0, 0, 0]),
        np.array([0, 0, L]),
        np.array([0, L, 0]),
        np.array([0, L, L]),
        np.array([L, 0, 0]),
        np.array([L, 0, L]),
        np.array([L, L, 0]),
        np.array([L, L, L]),
    ]


def bcc(L: float) -> list[NDArray[np.float64]]:
    """Generate the positions for a body-centered cubic (BCC) lattice.

    Parameters
    ----------
    L (float): The lattice constant.

    Returns
    -------
    list: A list of numpy arrays representing the positions of the BCC lattice points.

    """
    return [
        np.array([0, 0, 0]),
        np.array([0, 0, L]),
        np.array([0, L, 0]),
        np.array([0, L, L]),
        np.array([L, 0, 0]),
        np.array([L, 0, L]),
        np.array([L, L, 0]),
        np.array([L, L, L]),
        np.array([L / 2, L / 2, L / 2]),
    ]


def fcc(L: float) -> list[NDArray[np.float64]]:
    """Generate the positions for a face-centered cubic (FCC) lattice.

    Parameters
    ----------
    L (float): The lattice constant.

    Returns
    -------
    list: A list of numpy arrays representing the positions of the FCC lattice points.

    """
    return [
        np.array([0, 0, 0]),
        np.array([0, 0, L]),
        np.array([0, L, 0]),
        np.array([0, L, L]),
        np.array([L, 0, 0]),
        np.array([L, 0, L]),
        np.array([L, L, 0]),
        np.array([L, L, L]),
        np.array([L / 2, L / 2, 0]),
        np.array([L / 2, 0, L / 2]),
        np.array([0, L / 2, L / 2]),
        np.array([L / 2, L / 2, L]),
        np.array([L / 2, L, L / 2]),
        np.array([L, L / 2, L / 2]),
    ]
