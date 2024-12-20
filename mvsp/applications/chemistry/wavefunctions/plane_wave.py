"""This module provides functions to represent plane waves in ND space."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def plane_wave(
    coeffs_dict: dict[tuple[int, ...], np.complex128],
) -> Callable[[NDArray[np.float64]], NDArray[np.complex128]]:
    """Return a function that represents a plane wave in ND space.

    The plane wave is represented as a linear combination of complex exponentials.
    The tuple keys are used to determine the dimension of the k space.

    Args:
    ----
        coeffs_dict: A dictionary where the keys are tuples of integers representing the
        wave vector and the values are the coefficients of the complex exponentials.

    Returns:
    -------
        A function that takes a vector r in ND space and returns the value of the plane
        wave at r. Where the x,y ... are stacked in a vector r = [x,y,...]


    Example:
    -------
        plane_wave({(1,0):
        0.4343, (0,1): 0.343434}) returns a function that represents a plane wave in 2D
        space with wave vector (1,0) and (0,1), with expansion coefficients 0.4343
        and 0.343434 respectively.

    """
    dim = len(next(iter(coeffs_dict.keys())))

    def space_function(
        r: NDArray[np.float64],
    ) -> NDArray[np.complex128]:  # Vectorised function
        """Return the value of the plane wave at r.

        It is not normalised and you must use plane_wave_renorm to normalise it.
        Where the x,y ... are stacked in a vector r = [x,y,...].

        Args:
        ----
            r: A vector in ND space.

        Returns:
        -------
            The value of the plane wave at r.

        """
        return sum(
            [
                coeffs_dict[k]
                * np.exp(np.pi * 1j * np.tensordot(r, k, axes=([dim], [0])))
                for k in coeffs_dict
            ],
        )  # type: ignore

    return space_function


def plane_wave_renorm(plane_wave_r: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the normalised plane wave.

    Args:
    ----
        plane_wave_r: The plane wave to normalise.

    Returns:
    -------
        The normalised plane wave.

    """
    pw_flat = plane_wave_r.flatten()
    pw_flat_norm = pw_flat / np.linalg.norm(pw_flat)
    return pw_flat_norm.reshape(plane_wave_r.shape)
