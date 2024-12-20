from typing import Callable

import numpy as np
import scipy

from mvsp.utils.paper_utils import NDArrayFloat


class Chebyshev:
    def __init__(self, fun: Callable[[NDArrayFloat], NDArrayFloat], degree: int):
        self.fun = fun
        self.degree = degree
        self.coeffs = self._compute_coeffs()
        pass

    @staticmethod
    def chebyshev_roots(degree: int) -> NDArrayFloat:
        return np.cos((np.arange(degree + 1) + 0.5) * np.pi / (degree + 1))

    @staticmethod
    def chebyshev_extrema(degree: int) -> NDArrayFloat:
        return np.cos(np.arange(degree + 1) * np.pi / (degree + 1))

    def _compute_coeffs(self) -> NDArrayFloat:
        roots = self.chebyshev_roots(self.degree)
        y = self.fun(roots)

        dct = scipy.fft.dct(y)
        coeffs = dct / (self.degree + 1)
        coeffs[0] = coeffs[0] / 2
        return coeffs

    def __call__(self, x: NDArrayFloat):
        return np.polynomial.chebyshev.chebval(x, self.coeffs)


class Chebyshev2D:
    def __init__(
        self,
        fun: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        degree: list[int, int] | int,
        options: dict | None = None,
        run_ge: bool | None = False,
        method: str | None = None,
    ):
        if method is None:
            method = "dct"
        self.fun = fun
        if isinstance(degree, int):
            degree = [degree, degree]
        elif len(degree) == 1:
            degree = [degree[0], degree[0]]
        elif len(degree) != 2:
            raise ValueError("Length of `degree` should be 2")
        self.degree = degree
        self._coeffs_lr = None
        self._coeffs_dct = None
        self._coeffs = self._initialize_coeffs(method)
        # TODO: add iterative Gaussian elimination to perform a low-rank approximation
        # The following variables are currently not used
        self.ge = None
        self.cs = []
        self.rs = []
        self._A = []
        self._B = []
        self._D = []

    pass

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def D(self):
        return self._D

    @property
    def coeffs(self):
        return self._coeffs

    def _initialize_coeffs(self, method: str = "dct"):
        r"""Returns the Chebyshev coefficient matrix.

        The coefficient matrix is the :math:`n \times m` matrix determined by ..
        math::
            f(x, y) = \sum_{i=0}^n \sum_{j=0}^m X_{ij} T_i(y) T_j(x)
        FIXME in the Fourier implementation the first index i is associated with direction x

        If `method=="full"`, then the matrix is returned as a full :math:`n
        \times m` matrix. For any other value of `method` it is returned as a
        list of its rank-:math:`k` decomposition :math:`[A, D, B]` with
        :math:`X=ADB^T`.

        Args:
            method (str, optional): _description_. Defaults to "dct".

        Returns:
            _type_: _description_
        """
        method = method.lower()
        if method == "full":
            if self._coeffs_lr is None:
                self._initialize_coeffs_lr()

            return self._coeffs_lr
        elif method == "dct" or method == "fft":
            if self._coeffs_dct is None:
                self._initialize_coeffs_dct()

            return self._coeffs_dct
        else:
            return (self.A, self.D, self.B)

    def _initialize_coeffs_lr(self):
        self._coeffs_lr = self.A @ np.diag(self.D) @ self.B.T

    def _initialize_coeffs_dct(self):
        xs = Chebyshev.chebyshev_roots(self.degree[0])
        ys = Chebyshev.chebyshev_roots(self.degree[1])
        y = np.array([[self.fun(x, y) for x in xs] for y in ys])
        dct = scipy.fft.dctn(y)
        coeffs = dct / np.prod(np.array(self.degree) + 1)
        coeffs[0, :] /= 2
        coeffs[:, 0] /= 2
        self._coeffs_dct = coeffs

    def __call__(self, x: NDArrayFloat, y: NDArrayFloat):
        # f = np.sum([c(y) * d * r(x) for c, r, d in zip(self.cs, self.rs, self.D)])

        return self.eval_cheb(x, y)

    def eval_cheb(self, x, y, coeffs=None):
        r"""Evaluate Chebyshev polynomials using a coefficient matrix.

        If no coefficients are passed, it uses the coefficients generated from
        the low-rank approximation.

        Direct implementation of the formula
        .. math::
            f(x, y) = \sum_{i=0}^n \sum_{j=0}^n \alpha_{ij} T_i(y) T_j(x)\\
            \text{with } alpha = ADR^T

        """
        if coeffs is None:
            coeffs = self.coeffs

        def unit_vec(size, index):
            e = np.zeros(size)
            e[index] = 1.0
            return e

        f = np.sum(
            [
                coeffs[i, j]
                * np.polynomial.chebyshev.chebval(y, unit_vec(self.degree[0] + 1, i))
                * np.polynomial.chebyshev.chebval(x, unit_vec(self.degree[1] + 1, j))
                for i in range(self.degree[0] + 1)
                for j in range(self.degree[1] + 1)
            ]
        )

        return f


def p(x: NDArrayFloat, s: int, d: int) -> NDArrayFloat:
    """Evaluates a degree `d` polynomial approximating a monomial of degree :math:`s`
    on the interval :math:`[-1, 1]`.

    See Section 3 of
    ```
    S. Sachdeva and N. K. Vishnoi, Faster Algorithms via Approximation Theory,
    TCS 9, 125 (2014).
    ```

    Args:
        x (NDArrayFloat): Evaluation points
        s (int): Degree of the monomoial
        d (int): Degree of the approximating polynomial

    Returns:
        NDArrayFloat: Values of the approximating polynomial
    """
    lower = 0
    upper = d
    # Scipy's binom.pmf is 0 for values not in the support of the binomial. This ensures
    # that even and odd parts are correctly zeroed out for even and odd s
    coeffs = np.array(
        [
            2 * scipy.stats.binom.pmf((j + s) / 2, s, 0.5)
            for j in np.arange(lower, upper + 1)
        ]
    )
    coeffs[0] /= 2

    return np.polynomial.chebyshev.chebval(x, coeffs)


def approx_degree(s: int, delta: float) -> int:
    r"""Computes the minimal degree needed to approximate a monomial :math:`x^s` to
    error :math:`\delta`.

    See Section 3 of
    ```
    S. Sachdeva and N. K. Vishnoi, Faster Algorithms via Approximation Theory,
    TCS 9, 125 (2014).
    ```

    Args:
        s (int): _description_
        delta (float): _description_

    Returns:
        int: _description_
    """
    return int(min(np.ceil((np.sqrt(2 * s * np.log(2 / delta)))), s))
