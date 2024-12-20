"""Utils functions for GQSP."""

from numba import njit  # type: ignore
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfromroots
from scipy.linalg import expm


class FourierPolynomial:
    """Fourier class.

    Args:
    ----
        coeffs (dict): dictionary of coefficients.
        d_max (Optional[int]): maximum degree.
        d_min (int): minimum degree.

    """

    def __init__(
        self, coeffs: dict[int, np.complex128], d_max: None | int = None, d_min: int = 0
    ):
        """Initialise Fourier class."""
        self._coeffs = coeffs
        self._d_max = len(coeffs) - 1 if d_max is None else d_max
        self._d_min = d_min

    @property
    def coeffs(self) -> dict[int, np.complex128]:
        """Return dict of coefficients."""
        return self._coeffs

    @property
    def coeffs_list(self) -> NDArray[np.complex128]:
        """Return list of coefficients."""
        return np.array(list(self._coeffs.values()))

    @property
    def d_max(self) -> int:
        """Return maximum degree."""
        return self._d_max

    @property
    def d_min(self) -> int:
        """Return minimum degree."""
        return self._d_min

    def eval_mat(self, mat: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Given matrix mat, evaluates mat * pi."""
        powers = np.arange(self._d_min, self._d_max + 1)
        s = np.zeros_like(mat, dtype=np.complex128)
        for n in powers:
            s += self._coeffs[n] * expm(1j * n * np.pi * mat)
        return s

    def __call__(
        self, x: NDArray[np.float64], matrix: bool = False
    ) -> NDArray[np.complex128]:
        """Evaluate Fourier polynomial.

        Given a 1 or 2 dimensional array x, evaluates x * pi.

        Args:
        ----
            x (NDArray[np.complex128]): x values where to evaluate the polynomial.
            matrix (bool): If True, uses scipy expm to compute the exponential of the
                matrix.

        """
        exp_fun = np.exp
        if matrix:
            exp_fun = expm

        powers = np.arange(self._d_min, self._d_max + 1)
        s = np.zeros_like(x, dtype=np.complex128)
        for n in powers:
            s += self._coeffs[n] * exp_fun(1j * n * np.pi * x)
        return s


class GQSPAngleFinder:
    """Find GQSP angles via exact methods.

    Given a target Fourier series P, this class computes the phase factors to implement
    the GQSP circuit to apply such series to a given operator. It consists on two steps:
        - Step 1: Find the complementary series Q s.t. $|P(x)|^2+|Q(x)|^2 = 1$.
            Follows Proof of Lemma 4 in https://arxiv.org/pdf/2206.02826.
        - Step 2: Compute phase factors following the constructive method in
            https://arxiv.org/pdf/2308.01501.

    Although the procedure is exact, when considering large degrees numerical errors may
    arise.

    Args:
    ----
        target_polynomial (FourierPolynomial): Target Fourier series.
        verbose (bool): If True, prints details. Defaults to False.

    """

    def __init__(self, target_polynomial: FourierPolynomial, verbose: bool = False):
        """Initialise GQSPAngleFinder."""
        self._target_polynomial = target_polynomial
        self._verbose = verbose
        self._degree = len(target_polynomial.coeffs) - 1
        self._complementary_polynomial = self._get_complementary_polynomial()

        S = np.array(
            [
                self._target_polynomial.coeffs_list,
                self._complementary_polynomial.coeffs_list,
            ],
            order="F",
        )
        self._phase_factors = self._get_phase_factors(S, self._degree)

    @property
    def degree(self) -> int:
        """Return the degree."""
        return self._degree

    @property
    def target_polynomial(self) -> FourierPolynomial:
        """Return the target polynomial."""
        return self._target_polynomial

    @property
    def complementary_polynomial(self) -> FourierPolynomial:
        """Return the complementary polynomial."""
        return self._complementary_polynomial

    @property
    def phase_factors(self) -> NDArray[np.float64]:
        """Return the phases."""
        return self._phase_factors

    def _get_complementary_polynomial(self) -> FourierPolynomial:
        """Return the auxiliary polynomial Q satisfying unitary condition.

        Follows Proof of Lemma 4 in https://arxiv.org/pdf/2206.02826.
        """
        f_coeffs = self._target_polynomial.coeffs
        # Find coefficients of Laurent polynomial
        laurent_coeff: list[np.complex128] = []
        # k<0
        for k in range(-int(self._degree), 0):
            tmp = np.complex128(0)
            for i in range(-int(self._degree / 2), int(self._degree / 2) + k + 1):
                tmp += f_coeffs[i] * np.conj(f_coeffs[i - k])
            laurent_coeff.append(-1.0 * tmp)
        # k=0
        tmp = np.complex128(0)
        for i in range(-int(self._degree / 2), int(self._degree / 2) + 1):
            tmp += np.square(np.abs(f_coeffs[i]))
        laurent_coeff.append(1.0 - 1 * tmp)
        # k>0
        for k in range(1, int(self._degree) + 1):
            tmp = np.complex128(0)
            for i in range(-int(self._degree / 2) + k, int(self._degree / 2) + 1):
                tmp += f_coeffs[i] * np.conj(f_coeffs[i - k])
            laurent_coeff.append(-1 * tmp)

        # Find roots of Laurent polynomial and sort them by magnitude
        p = Polynomial(np.array(laurent_coeff))
        G_roots = p.roots()
        mod_G_roots = np.abs(G_roots)
        idxs = mod_G_roots.argsort()
        G_roots = G_roots[idxs]

        prefactor = np.sqrt(laurent_coeff[-1] * np.prod(G_roots[0 : self._degree]))

        # Vieta's formula
        roots = G_roots[: self._degree]
        h_poly_coeffs = polyfromroots(1 / np.conj(roots))
        h_coeff = {
            k - int(self._degree / 2): prefactor * complex(h_poly_coeffs[k])
            for k in range(len(h_poly_coeffs))
        }

        if self._verbose:
            print(f"Roots of the Laurent poly sorted by increasing magn:\n {G_roots}\n")
            print(f"The prefactor must be real: \n{prefactor}\n")

        complementary_polynomial = FourierPolynomial(
            h_coeff, self._target_polynomial.d_max, self._target_polynomial.d_min
        )
        return complementary_polynomial

    def _get_phase_factors(
        self, S: NDArray[np.complex128 | np.float64], d: int
    ) -> NDArray[np.float64]:
        """Return phase factors for GQSP given polynomials coeffs.

        From https://arxiv.org/pdf/2308.01501.
        """
        a_d, b_d = S[0][d], S[1][d]
        theta_d: np.float64 = np.arctan(np.abs(b_d) / np.abs(a_d))
        phi_d: np.float64 = np.angle(np.array([a_d / b_d]))[0]
        if d == 0:
            lam = np.angle(b_d)
            return np.array([[theta_d, phi_d, lam]])
        else:
            new_S = self._arbitrary_su2(theta_d, phi_d, 0.0).conj().T @ S
            new_S = np.array([new_S[0][1:], new_S[1][0:d]])
            return np.append(
                self._get_phase_factors(new_S, d - 1),
                np.array([(theta_d, phi_d, 0)]),
                axis=0,
            )

    @staticmethod
    @njit
    def _arbitrary_su2(
        theta: np.float64, phi: np.float64, lam: float
    ) -> NDArray[np.complex128]:
        """Return R(theta, phi, lambda)."""
        return np.array(
            [
                [
                    np.exp(1j * (lam + phi)) * np.cos(theta),
                    np.exp(1j * phi) * np.sin(theta),
                ],
                [np.exp(1j * lam) * np.sin(theta), -np.cos(theta)],
            ]
        )
