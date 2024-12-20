import numpy as np
from scipy.stats import multivariate_normal


def gaussian2d(x, y, mean=None, cov=None):
    if mean is None:
        mean = np.array([0.5, 0.5])
    if cov is None:
        cov = np.eye(2)
    return multivariate_normal(mean=mean, cov=cov).pdf(np.dstack([x, y]))
