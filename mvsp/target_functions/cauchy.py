import numpy as np
from scipy.stats import multivariate_t


def cauchy2d(x, y, loc=None, shape=None):
    if loc is None:
        loc = np.array([0.5, 0.5])
    if shape is None:
        shape = np.array([[0.05, 0.0], [0.0, 0.05]])
    return multivariate_t(loc=loc, shape=shape, df=1).pdf(np.dstack([x, y]))


def cauchy2d_sqrt(x, y, loc=None, shape=None):
    if loc is None:
        loc = np.array([0.5, 0.5])
    if shape is None:
        shape = np.array([[0.05, 0.0], [0.0, 0.05]])
    return np.sqrt(multivariate_t(loc=loc, shape=shape, df=1).pdf(np.dstack([x, y])))
