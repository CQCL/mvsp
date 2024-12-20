"""angles functions for QSVT."""

from collections.abc import Callable
from abc import ABC, abstractmethod

import numpy as np
from numpy import complexfloating
from numpy.typing import NDArray
from numba import njit  # type: ignore

from scipy.fft import dct
from scipy.optimize import (
    OptimizeResult,
    minimize,
)
from typing import Any
from collections.abc import Sequence


class ChebyshevPolynomial:
    """Chebyshev class.

    Args:
    ----
        fun (Callable[[NDArray], NDArray]): function to be approximated
        degree (int): degree of the polynomial

    """

    def __init__(
        self, fun: Callable[[NDArray[np.float64]], NDArray[np.float64]], degree: int
    ):
        """Initialise Chebyshev object."""
        self._fun = fun
        self._degree = degree
        self._roots = np.cos(
            (np.arange(self._degree + 1) + 0.5) * np.pi / (self._degree + 1)
        )
        self._extrema = np.cos(np.arange(self._degree + 1) * np.pi / (self._degree + 1))
        self._coeffs = self._compute_coeffs()

    @property
    def coeffs(self) -> NDArray[np.float64]:
        """Return the coefficients."""
        return self._coeffs

    @property
    def degree(self) -> int:
        """Return the degree."""
        return self._degree

    @property
    def roots(self) -> NDArray[np.float64]:
        """Return the roots."""
        return self._roots

    @property
    def extrema(self) -> NDArray[np.float64]:
        """Return the extrema."""
        return self._extrema

    @property
    def fun(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the original function."""
        return self._fun

    def _compute_coeffs(self) -> NDArray[np.float64]:
        """Compute coefficients of Chebyshev polynomial approximation.

        Returns
        -------
            NDArray[np.float64]: coefficients

        """
        y = self._fun(self._roots)
        coeffs = dct(y) / (self._degree + 1)
        coeffs[0] = coeffs[0] / 2
        return coeffs

    def __call__(
        self, x: NDArray[np.float64] | NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Evaluate Chebyshev polynomial.

        Args:
        ----
            x (NDArray[np.complex128]): x values where to evaluate the polynomial

        """
        return np.polynomial.chebyshev.chebval(x, self._coeffs)


class BaseCompilePhases(ABC):
    """Abstract class for Phase Compilers for QSPAngleOptimiser."""

    @abstractmethod
    def _construct_loss_function(
        self,
        x: NDArray[np.float64],
        fun_vals: NDArray[np.complex128] | NDArray[np.float64],
        d_phi: int,
    ) -> Callable[[NDArray[np.float64]], np.float64]:
        """Construct loss function for the QSP angle optimisation accelerated with njit.

        Args:
        ----
            x (NDArray[np.float64]): x positions where the function is evaluated.
            fun_vals (NDArray[np.complex128] | NDArray[np.float64]): Values of the
                target function.
            d_phi (int): Degree of the QSP.

        Returns:
        -------
            Callable[[NDArray[np.float64]], np.float64]: Loss function.

        """
        pass

    @staticmethod
    def _f_phi(
        phi: Sequence[float] | NDArray[np.float64], x: NDArray[np.float64]
    ) -> NDArray[Any] | NDArray[complexfloating[Any, Any]]:
        """Compute the QSP unitary given x and the phases.

        Args:
        ----
            phi (list[np.float64] | NDArray[np.float64]): Standard phases.
            x (NDArray[np.float64]): x positions where to evaluate the function.

        Returns:
        -------
            NDArray[Any] | NDArray[complexfloating[Any, Any]]: Values of the function.

        """
        # Construct U_phi
        W_x = np.array([[x, 1j * np.sqrt(1 - x**2)], [1j * np.sqrt(1 - x**2), x]])
        U_phi = np.array([[np.exp(1j * phi[0]), 0], [0, np.exp(-1j * phi[0])]])
        for p in phi[1:]:
            U_phi = U_phi @ W_x @ np.array([[np.exp(1j * p), 0], [0, np.exp(-1j * p)]])
        return np.real(U_phi[0, 0])


class CompilerPhasesNumpy(BaseCompilePhases):
    """Compiler based on numpy operations."""

    def __init__(self):
        """Initialise CompilerPhasesNumpy."""
        pass

    def _construct_loss_function(
        self,
        x: NDArray[np.float64],
        fun_vals: NDArray[np.complex128] | NDArray[np.float64],
        d_phi: int,
    ) -> Callable[[NDArray[np.float64]], np.float64]:
        def loss_function(
            phi_hat: list[np.float64] | NDArray[np.float64],
        ) -> np.float64:
            if d_phi % 2 == 0:
                phi = np.concatenate([phi_hat, phi_hat[-2::-1]])
            else:
                phi = np.concatenate([phi_hat, phi_hat[::-1]])
            return np.sum(
                [
                    np.abs(self._f_phi(phi, x[i]) - fun_vals[i]) ** 2
                    for i in range(len(x))
                ]
            )

        return loss_function


class CompilerPhasesNumba(BaseCompilePhases):
    """Compiler based on numba operations."""

    def __init__(self):
        """Initialise CompilerPhasesNumba."""
        self._f_phi = njit(self._f_phi)  # type: ignore

    def _construct_loss_function(
        self,
        x: NDArray[np.float64],
        fun_vals: NDArray[np.complex128] | NDArray[np.float64],
        d_phi: int,
    ) -> Callable[[NDArray[np.float64]], np.float64]:
        f_phi_nb = self._f_phi

        def loss_function(
            phi_hat: list[np.float64] | NDArray[np.float64],
        ) -> np.float64:
            phi = np.zeros(d_phi + 1, dtype=np.float64)
            if (d_phi + 1) % 2 == 0:
                phi[0 : len(phi_hat)] = phi_hat
                phi[len(phi_hat) :] = phi_hat[::-1]
            else:
                phi[0 : len(phi_hat)] = phi_hat
                phi[len(phi_hat) :] = phi_hat[-2::-1]

            total = np.abs(f_phi_nb(phi, x[0]) - fun_vals[0]) ** 2  # type: ignore
            for i in range(1, len(x)):
                total += np.abs(f_phi_nb(phi, x[i]) - fun_vals[i]) ** 2  # type: ignore
            return total

        return njit(loss_function)  # type: ignore


class QSPAngleOptimiser:
    """Find QSP angles via optimisation.

    Based on https://arxiv.org/abs/2002.11649. Given a target polynomial f(x), this
    protocol minimises the loss function L(phi)=dist(Re(bra(0)U_phi(x)ket(0)), f(x)).
    This protocol makes use of the symmetry of the phases to reduce the complexity of
    the problem. Then, given the degree d, we have d+1 phases (named standard phases),
    but under the symmetric conditions we only need to find d_hat = ceil((d+1)/2) phases
    (named phi_hat and symmetric phases). The way to map one convention from the other
    is that if phi_hat = (phi_hat_0, ..., phi_hat_{d_hat-1}):
        -d odd: phi = (phi_hat_0, ..., phi_hat_{d_hat-1}, phi_hat_{d_hat-1}, ...,
                        phi_hat_0)
        -d even: phi = (phi_hat_0, ..., phi_hat_{d_hat-2}, phi_hat_{d_hat-1},
                        phi_hat_{d_hat-2},..., phi_hat_0)

    There is also the option to use numba to accelerate the evaluation of the loss
    function.

    Args:
    ----
        d_phi (int): Degree of the QSP.
        target_polynomial (Callable[[NDArray[np.float64]], NDArray[np.float64]]):
            Target Chebyshev polynomial.
        phi_0 (NDArray[np.float64]):  Set of initial phases.
        compiler (BaseCompilePhases): Compiler used for generating the loss function.
            Availables: 'numpy' and 'numba'.
        verbose (bool). If True, prints the optimization error. Defaults to False.

    """

    def __init__(
        self,
        d_phi: int,
        target_polynomial: (
            Callable[[NDArray[np.float64]], NDArray[np.float64]] | ChebyshevPolynomial
        ),
        phi_0: NDArray[np.float64] | None = None,
        compiler: BaseCompilePhases | None = None,
        verbose: bool = False,
    ):
        """Initialise QSPAngleOptimiser."""
        self._d_phi = d_phi
        self._target_polynomial = target_polynomial

        if compiler is None:
            self._compiler = CompilerPhasesNumpy()
        else:
            self._compiler = compiler

        self._phi_0 = phi_0 if phi_0 is not None else self._get_phi_0()
        self._phi_hat_0 = self._convert_phi_to_phi_hat(self._phi_0)

        self._d_tilde = int(np.ceil((self._d_phi + 1) / 2))
        self._x_Chebyshev_roots = self._get_x_Chebyshev_roots()
        self._fun_vals = self._target_polynomial(self._x_Chebyshev_roots)

        self._loss = self._compiler._construct_loss_function(
            self._x_Chebyshev_roots, self._fun_vals, self._d_phi
        )
        self._res = self._minimize()

        self._phi_hat = self._res.x
        self._phi = self._convert_phi_hat_to_phi(self._phi_hat)

        if verbose:
            print("Optimization error:", self._res.fun)

    @property
    def phi(self) -> Sequence[float]:
        """Return the phases."""
        return self._phi

    @property
    def target_polynomial(
        self,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]] | ChebyshevPolynomial:
        """Return the target polynomial."""
        return self._target_polynomial

    @property
    def x_Chebyshev_roots(self) -> NDArray[np.float64]:
        """Return the roots."""
        return self._x_Chebyshev_roots

    @property
    def fun_vals(self) -> NDArray[np.complex128] | NDArray[np.float64]:
        """Return the value of the target function."""
        return self._fun_vals

    @property
    def compiler(self) -> BaseCompilePhases:
        """Return the compiler."""
        return self._compiler

    def _get_phi_0(self) -> NDArray[np.float64]:
        """Compute initial value for phi."""
        phi_0 = np.zeros(self._d_phi + 1)
        phi_0[0] = np.pi / 4
        phi_0[-1] = np.pi / 4
        return phi_0

    def _get_x_Chebyshev_roots(self) -> NDArray[np.float64]:
        """Compute the Chebyshev roots."""
        return np.array(
            [
                np.cos((2 * i - 1) * np.pi / 4 / self._d_tilde)
                for i in range(1, self._d_tilde + 1)
            ]
        )

    def _minimize(self) -> OptimizeResult:
        """Minimisation process using scipy.

        Returns
        -------
            OptimizeResult: result of scipy minimize

        """
        bounds = [(-np.pi, np.pi) for _ in range(self._d_tilde)]

        return minimize(
            self._loss,
            self._phi_hat_0,
            # tol = 1E-12,
            # ftol = 1E-12,
            bounds=bounds,
            method="L-BFGS-B",
            # options={"ftol": 1E-12, "gtol": 1E-12}
        )

    def __call__(self, x: NDArray[np.float64]) -> NDArray[Any]:
        """Call optimiser and evaluate the approximation function.

        Args:
        ----
            x (NDArray[np.float64]): x positions where to evaluate the function.

        Returns:
        -------
            NDArray[np.float64]: Values of the approximation function.

        """
        return np.array([self._compiler._f_phi(self.phi, x_i) for x_i in x])

    def _convert_phi_hat_to_phi(
        self, phi_hat: list[np.float64] | NDArray[np.float64]
    ) -> Sequence[float]:
        """Convert phi_hat format to standard phi format.

        Args:
        ----
            phi_hat (list[np.float64] | NDArray[np.float64]): Symmetric phases.

        Returns:
        -------
            list[np.float64] | NDArray[np.float64]: Standard phases.

        """
        if self._d_phi % 2 == 0:
            phi = np.concatenate([phi_hat, phi_hat[-2::-1]])
        else:
            phi = np.concatenate([phi_hat, phi_hat[::-1]])
        return list(phi)

    @staticmethod
    def _convert_phi_to_phi_hat(
        phi: list[np.float64] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert standard phi format to phi_hat format.

        Args:
        ----
            phi (list[np.float64] | NDArray[np.float64]): Standard phases.

        Returns:
        -------
            list[np.float64] | NDArray[np.float64]: Symmetric phases.

        """
        d_tilde = int(np.ceil((len(phi)) / 2))
        phi_hat = phi[:d_tilde]
        return np.array(phi_hat)
