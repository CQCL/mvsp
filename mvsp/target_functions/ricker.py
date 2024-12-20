import numpy as np


def ricker2d(x, y, sigma=None):
    if sigma is None:
        sigma = 0.5
    r2 = x**2 + y**2
    return np.exp(-r2 / (2 * sigma**2)) * (1 - 0.5 * r2 / sigma**2) / (np.pi * sigma**4)


def ricker1d(x, sigma=None):
    if sigma is None:
        sigma = 0.5
    r2 = x**2
    return (
        2
        * np.exp(-r2 / (2 * sigma**2))
        * (1 - r2 / sigma**2)
        / (np.sqrt(3 * sigma * np.sqrt(np.pi)))
    )
