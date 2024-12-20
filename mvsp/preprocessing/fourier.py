from itertools import chain, product
from typing import Callable

import numpy as np
import scipy
from scipy.fft import fftfreq, fftshift

from mvsp.utils.paper_utils import NDArrayFloat, NDArrayInt


class FourierExpansion:
    def __init__(
        self,
        degrees: list[int] | NDArrayFloat,
        func: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat] | None = None,
        coefficient_function: Callable | None = None,
        options: dict | None = None,
        run_ge: bool | None = True,
    ):
        self.func = func
        self.coefficient_function = coefficient_function
        self.degrees = degrees
        self.n_coeffs = np.array(degrees) + 1.0
        self._coeffs_matrix = None
        self._coeffs_dct = None
        self._evaluation_fun = None

    def coeffs(self, method: str = "dct"):
        r"""Returns the Chebyshev coefficient matrix.

        The coefficient matrix is the :math:`n \times m` matrix determined by ..
        math::
            f(x, y) = \sum_{i=0}^n \sum_{j=0}^m X_{ij} T_i(y) T_j(x)

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
            if self._coeffs_matrix is None:
                self._initialize_coeffs_matrix()

            return self._coeffs_matrix
        elif method == "dct" or method == "fft":
            if self._coeffs_dct is None:
                self._initialize_coeffs_dct()

            return self._coeffs_dct

    def _initialize_coeffs_matrix(self):
        if self.coefficient_function is not None:
            coeffs_indices = product(
                [
                    range(int(-n_c / 2), int(-n_c / 2) + int(n_c))
                    for n_c in self.n_coeffs
                ]
            )
            self._coeffs_matrix = np.zeros(shape=self.n_coeffs)
            for idx in coeffs_indices:  # TODO: not efficient implementation
                self._coeffs_matrix[idx] = self.coefficient_function(np.array(idx))
        else:
            raise NotImplementedError

    def _initialize_coeffs_dct(self):
        if self.coefficient_function is not None:
            coeffs_indices = product(
                *[
                    range(int(-n_c / 2), int(-n_c / 2) + int(n_c))
                    for n_c in self.n_coeffs
                ]
            )
            self._coeffs_dct = {
                tuple(idx): self.coefficient_function(np.array(idx))
                for idx in coeffs_indices
            }
        else:
            raise NotImplementedError

    def _initialize_eval(self):
        if self._coeffs_dct is not None:
            eval_fun = np.vectorize(
                lambda x: np.sum(
                    [
                        self._coeffs_dct[idx] * np.exp(np.pi * 1j * np.dot(idx, x))
                        for idx in self._coeffs_dct.keys()
                    ]
                ),
                signature="(m)->()",
            )
            self._evaluation_fun = eval_fun
        elif self._coeffs_matrix is not None:
            raise NotImplementedError
        else:
            print(
                "Before initializing evaluation function, you need to initialize coefficients."
            )

    def __call__(self, x):
        if self._evaluation_fun is None:
            self._initialize_eval()
        return self._evaluation_fun(x)

    def get_max_error(
        self, interval_min=[0, 0], interval_max=[1, 1], n_discretization=[64, 64]
    ):
        # only works for 2D, test for higher dimension
        x = [
            np.linspace(interval_min[i], interval_max[i], n_discretization[i])
            for i in range(len(interval_min))
        ]
        xx = np.meshgrid(*x)
        pos = np.stack(xx, axis=-1)

        exact_eval = self.func(pos)
        fourier_eval = self.__call__(pos)
        self.max_error = np.max(np.abs(exact_eval - fourier_eval))
        return self.max_error

    def get_max_error_normalized(
        self, interval_min=[0, 0], interval_max=[1, 1], n_discretization=[64, 64]
    ):
        # only works for 2D, test for higher dimension
        x = [
            np.linspace(interval_min[i], interval_max[i], n_discretization[i])
            for i in range(len(interval_min))
        ]
        xx = np.meshgrid(*x)
        pos = np.stack(xx, axis=-1)

        exact_eval = self.func(pos)
        exact_eval = exact_eval / np.linalg.norm(exact_eval)
        fourier_eval = self.__call__(pos)
        fourier_eval = fourier_eval / np.linalg.norm(fourier_eval)
        self.max_error_normalized = np.max(np.abs(exact_eval - fourier_eval))
        return self.max_error_normalized

    def get_norm_error_normalized(
        self,
        interval_min=[0, 0],
        interval_max=[1, 1],
        n_discretization=[64, 64],
        ord=None,
    ):
        # only works for 2D, test for higher dimension
        x = [
            np.linspace(interval_min[i], interval_max[i], n_discretization[i])
            for i in range(len(interval_min))
        ]
        xx = np.meshgrid(*x)
        pos = np.stack(xx, axis=-1)

        exact_eval = self.func(pos)
        exact_eval = exact_eval / np.linalg.norm(exact_eval)
        fourier_eval = self.__call__(pos)
        fourier_eval = fourier_eval / np.linalg.norm(fourier_eval)
        self.norm_error_normalized = np.linalg.norm(exact_eval - fourier_eval, ord=ord)
        return self.norm_error_normalized


class Fourier:
    def __init__(self, fun: Callable[[NDArrayFloat], NDArrayFloat], degree: int):
        def _fun(x):
            r"""Construct even symmetric, periodic extension of the input
            function according to Appendix A.

            Given an input function :math:`f(x)` this implements
            :math:`\tilde{f}(x) = f(|x|)`.

            Additionally it wraps the function around the interval :math:`[-1,
            1]`, i.e. it provides a periodic extension with period 2.

            Args:
                x (NDArrayFloat): x values

            Returns:
                NDArrayFloat: Array of values of the periodically extended
                function
            """
            x = np.array(x)
            assert np.allclose(np.imag(x), 0)
            # Periodic extension around [a, b)
            # Note: this formula maps 1 -> -1
            a, b = -1, 1
            x_ = (x - a) % (b - a) + a
            f = fun(np.where(x_ >= 0, x_, -x_))
            return f

        self.fun = np.vectorize(_fun)
        self.degree = degree
        self._orig_fun = fun
        self._coeffs_s = self._compute_coeffs()
        self.coeffs = fftshift(self._coeffs_s)

    @staticmethod
    def _collocation_points_s(degree: int) -> NDArrayFloat:
        """Interpolation points for the Fourier expansion.

        We use `scipy.fft.fftfreq` so that the order of the points matches
        `scipy`'s convention for the FFT.

        Args:
            degree (int): Degree of the Fourier expansion

        Returns:
            NDArrayFloat: Array of :math:`2N+1` interpolation points.
        """
        N = 2 * degree + 1
        return fftfreq(N, d=0.5)

    @staticmethod
    def collocoation_points(degree: int) -> NDArrayFloat:
        return fftshift(Fourier._collocation_points_s(degree))

    @staticmethod
    def _wave_numbers_s(N: int) -> NDArrayInt:
        return fftfreq(N, d=1 / N).astype(int)

    @staticmethod
    def wave_numbers(N: int) -> NDArrayInt:
        return fftshift(Fourier._wave_numbers_s(N))

    def _compute_coeffs(self) -> NDArrayFloat:
        r"""Compute coefficients of the Fourier expansion of order `self.degree`.

        A Fourier expansion :math:`f_N(x)` of order `N` has :math:`2N+1` coefficients:

        .. math::
            f_N(x) = \sum_{n=-N}^N c_n e^{inx}.

        See Sec. 7.7.2 in
        http://sites.science.oregonstate.edu/~restrepo/475A/Notes/sourcea.pdf
        for details of the computation.

        Returns:
            NDArrayFloat: :math:`2N+1` Fourier coefficients
        """
        x = self._collocation_points_s(self.degree)
        y = self.fun(x)

        yfft = scipy.fft.fft(y)
        coeffs = yfft / (2 * self.degree + 1)
        return coeffs

    def __call__(self, x: NDArrayFloat):
        def _eval(_x: NDArrayFloat):
            N = int(2 * self.degree + 1)
            p = np.dot(
                self._coeffs_s,
                np.exp(1j * np.pi * _x * self._wave_numbers_s(N)),
            )
            return p

        return np.vectorize(_eval)(x)


class Fourier2D:
    def __init__(
        self,
        fun: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
        degree: list[int, int] | int,
    ):
        def _fun(x: NDArrayFloat, y: NDArrayFloat) -> NDArrayFloat:
            r"""Construct even symmetric, periodic extension of the input
            function according to Appendix A.

            Given an input function :math:`f(x, y)` this implements
            :math:`\tilde(f)(x, y) = f(|x|, |y|)`.

            Additionally it wraps the function around the interval :math:`[-1,
            1]^2`, i.e. it provides a periodic extension with period 2.

            Args:
                x (NDArrayFloat): x values y (NDArrayFloat): y values

            Returns:
                NDArrayFloat: Array of values of the periodically extended
                function
            """
            x = np.array(x)
            y = np.array(y)
            assert np.allclose(np.imag(x), 0)
            assert np.allclose(np.imag(y), 0)
            # Periodic extension around [a, b)
            # Note: This formula maps 1 -> -1
            a, b = -1, 1
            x_ = (x - a) % (b - a) + a
            y_ = (y - a) % (b - a) + a

            xhat = np.where(x_ >= 0, x_, -x_)
            yhat = np.where(y_ >= 0, y_, -y_)
            f = fun(xhat, yhat)

            return f

        self.fun = np.vectorize(_fun)
        if isinstance(degree, int):
            degree = [degree, degree]
        elif len(degree) == 1:
            degree = [degree[0], degree[0]]
        elif len(degree) != 2:
            raise ValueError("Length of `degrees` should be 2")
        self.degree = degree
        self._orig_fun = fun
        self._coeffs_s = self._compute_coeffs()
        self.coeffs = fftshift(self._coeffs_s)

    def _wave_number_array_s(self) -> list[list[tuple]]:
        """Returns the array of 2D wave numbers with order corresponding to the
        order of the coefficients.

        Say :math:`A` is the matrix returned by this function, then `A[0,0]` is
        the wave number `(k_0, k_0)` corresponding to the coefficient `C_{00}`.

        Returns:
            list[list[tuple]]: Matrix of tuples with integer wave numbers
        """
        N = int(2 * self.degree[0] + 1)
        M = int(2 * self.degree[1] + 1)
        return [
            [(n, m) for n in Fourier._wave_numbers_s(N)]
            for m in Fourier._wave_numbers_s(M)
        ]

    def wave_number_array(self) -> list[list[tuple]]:
        return fftshift(self._wave_number_array_s())

    def _compute_coeffs(self) -> NDArrayFloat:
        r"""Compute coefficients of the Fourier expansion of order
        `self.degree`.

        A Fourier expansion :math:`f_{NM}(x, y)` of order `N\times M` has
        :math:`(2N+1)(2M+1)` coefficients:

        .. math::
            f_{NM}(x, y) = \sum_{n=-N}^N \sum{m=-M}^M c_{nm} e^{inx+imy}.

        See Sec. 7.7.2 in
        http://sites.science.oregonstate.edu/~restrepo/475A/Notes/sourcea.pdf
        for details of the computation.

        Returns:
            NDArrayFloat: Array of size :math:`2N+1 \times 2M+1` of Fourier coefficients
        """
        x = Fourier._collocation_points_s(self.degree[0])
        y = Fourier._collocation_points_s(self.degree[1])
        xx, yy = np.meshgrid(x, y)
        f = self.fun(xx, yy)

        yfft = scipy.fft.fftn(f)
        coeffs = yfft / np.prod(2 * np.array(self.degree) + 1)
        return coeffs

    def __call__(self, x: NDArrayFloat, y: NDArrayFloat):
        def _eval(_x: NDArrayFloat, _y: NDArrayFloat):
            def _basis(k):
                xx = np.array([_x, _y])
                k = np.array(k)
                return np.exp(1j * np.pi * np.dot(xx, k))

            wave_indices = self._wave_number_array_s()
            shape = np.array(wave_indices).shape

            # Iterate over all wave number combinations and reshape so it
            # matches the shape of `self.wave_number_array()`
            C = np.array(
                [_basis(k) for k in chain.from_iterable(wave_indices)]
            ).reshape(shape[:-1])

            # Elementwise multiplication of coefficients and matrix of basis
            # functions
            return np.sum(self._coeffs_s * C)

        return np.vectorize(_eval)(x, y)
