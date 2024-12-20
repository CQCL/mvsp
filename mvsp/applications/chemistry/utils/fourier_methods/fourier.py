"""Fourier methods for 1D and 2D signals."""

import numpy as np
from numpy.typing import NDArray


def dft(
    signal: NDArray[np.complex128],
    k_min: int,
    k_max: int,
) -> NDArray[np.complex128]:
    """Compute the Discrete Fourier Transform of a signal.

    Args:
    ----
        signal: The input signal.
        k_min: The minimum index of the DFT.
        k_max: The maximum index of the DFT.

    Returns:
    -------
        The DFT of the signal.

    """
    truncated_dft_result = np.zeros(k_max - k_min, dtype=complex)

    for k in range(k_min, k_max):
        sum_result = 0
        for n in range(len(signal)):
            sum_result += signal[n] * np.exp(-1j * 2 * np.pi * k * n / len(signal))
        truncated_dft_result[k - k_min] = sum_result / len(signal)

    return truncated_dft_result


def idft(
    truncated_dft_result: NDArray[np.complex128],
    sig_len: int,
    k_min: int,
    k_max: int,
) -> NDArray[np.complex128]:
    """Compute the Inverse Discrete Fourier Transform of a signal.

    Args:
    ----
        truncated_dft_result: The truncated DFT of the signal.
        sig_len: The length of the original signal.
        k_min: The minimum index of the DFT.
        k_max: The maximum index of the DFT.

    Returns:
    -------
        The IDFT of the signal.

    """
    reconstructed_signal = np.zeros(sig_len, dtype=complex)

    for n in range(sig_len):
        sum_result = 0
        for k in range(k_min, k_max):
            sum_result += truncated_dft_result[k - k_min] * np.exp(
                1j * 2 * np.pi * k * n / sig_len,
            )
        reconstructed_signal[n] = sum_result

    return reconstructed_signal


def dft2(
    signal_2d: NDArray[np.complex128],
    k_min: int,
    k_max: int,
) -> NDArray[np.complex128]:
    """Compute the 2D Discrete Fourier Transform of a signal.

    Args:
    ----
        signal_2d: The input 2D signal.
        k_min: The minimum index of the DFT.
        k_max: The maximum index of the DFT.

    Returns:
    -------
        The 2D DFT of the signal.

    """
    # Apply DFT to each row
    dft_rows = np.array([dft(row, k_min, k_max) for row in signal_2d])

    # Apply DFT to each column of the result
    dft_columns = np.array(
        [dft(dft_rows[:, col], k_min, k_max) for col in range(dft_rows.shape[1])],
    )

    # Transpose the result back to the original orientation
    return dft_columns.T


def idft2(
    truncated_dft_2d: NDArray[np.complex128],
    original_shape: tuple[int, int],
    k_min: int,
    k_max: int,
) -> NDArray[np.complex128]:
    """Compute the 2D Inverse Discrete Fourier Transform of a signal.

    Args:
    ----
        truncated_dft_2d: The truncated 2D DFT of the signal.
        original_shape: The shape of the original signal.
        k_min: The minimum index of the DFT.
        k_max: The maximum index of the DFT.

    Returns:
    -------
        The 2D IDFT of the signal.

    """
    # Apply IDFT to each column
    idft_cols = np.array(
        [
            idft(truncated_dft_2d[:, col], original_shape[0], k_min, k_max)
            for col in range(truncated_dft_2d.shape[1])
        ],
    ).T

    # Apply IDFT to each row of the result
    return np.array(
        [
            idft(idft_cols[row], original_shape[1], k_min, k_max)
            for row in range(idft_cols.shape[0])
        ],
    )
