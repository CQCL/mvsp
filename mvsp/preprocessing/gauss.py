import numpy as np
from scipy.stats import multivariate_normal


def gaussian_function(mean=None, cov=None):
    rv = multivariate_normal(mean=mean, cov=cov)
    return rv.pdf


def gaussian_function_old(
    vec, mean, cov_inv
):  # note, takes inverse covariance in input
    factor = np.sqrt(np.linalg.det(cov_inv)) / (2 * np.pi)
    value = factor * np.exp(
        -0.5 * (vec - mean).transpose().conjugate() @ cov_inv @ (vec - mean)
    )
    return value[0, 0]


def coefficient_function_old(vec, mean, cov):  # note, takes covariance in input
    factor = 2 ** (-2)
    value = factor * np.exp(
        -np.pi * 1j * mean.transpose().conjugate() @ vec
        - 0.5 * np.pi**2 * vec.transpose().conjugate() @ cov @ vec
    )
    return value[0, 0]


def gauss_coefficient_function(mean, cov):  # note, takes covariance in input
    factor = 2 ** (-2)

    def fun(vec):
        return factor * np.exp(
            -np.pi * 1j * mean.conjugate() @ vec
            - 0.5 * np.pi**2 * vec.conjugate() @ cov @ vec
        )

    fun = np.vectorize(fun, signature="(a)->()")
    return fun
