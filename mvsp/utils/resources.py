import logging
from collections.abc import Iterable

import numpy as np
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation_block_encoding import (
    ChebychevBlockEncoding,
    FourierBlockEncoding,
)
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation import (
    LCUStatePreparationBox,
)

from mvsp.preprocessing.chebyshev import Chebyshev2D
from mvsp.preprocessing.fourier import Fourier2D
from mvsp.utils.paper_utils import EvaluateLCU


def resource_scaling(
    func,
    d,
    log_grid_sizes,
    backend,
    series_type,
    compile_only=True,
    compiler_options=None,
    logger=None,
):
    """XXX: Note that this function assumes 2D approximations"""
    if compiler_options is None:
        compiler_options = {"optimisation_level": 0}

    if logger is None:
        logger = logging.getLogger("Resources")
        logger.setLevel(logging.DEBUG)

    if not issubclass(type(d), Iterable):
        d = [d, d]

    if series_type == "chebyshev":
        series = Chebyshev2D
        block_encoding = ChebychevBlockEncoding
        interval_min = [-1, -1]
        series_kwargs = {"run_ge": False}
        min_fourier_indices = [0 for _ in d]
    elif series_type == "fourier":
        series = Fourier2D
        block_encoding = FourierBlockEncoding
        interval_min = [0, 0]
        min_fourier_indices = [int(-d_) for d_ in d]
        series_kwargs = {}
    else:
        raise ValueError(
            f"series_type should be 'chebyshev' or 'fourier' (was: {series_type})"
        )

    func_approx = series(func, d, **series_kwargs)
    coeffs = func_approx.coeffs
    logger.debug("Creating LCU circuit")
    lcu = LCUStatePreparationBox(
        coeffs,
        [block_encoding(log_grid_size) for log_grid_size in log_grid_sizes],
        min_basis_indices=min_fourier_indices,
    )
    logger.debug("Compiling circuit")
    evaluated_lcu = EvaluateLCU(
        lcu,
        backend,
        compiler_options=compiler_options,
        compile_only=compile_only,
    )

    xx, yy = evaluated_lcu.grid_points(interval_min=interval_min)
    f_eval = np.array([[func(x, y) for x in xx[0]] for y in yy[:, 0]])
    f_eval /= np.sqrt(np.sum(np.abs(f_eval) ** 2))
    func_approx_eval = np.array([[func_approx(x, y) for x in xx[0]] for y in yy[:, 0]])
    if compile_only:
        eval_real = None
        eval_imag = None
    else:
        eval_real = evaluated_lcu().real
        eval_imag = evaluated_lcu().imag
    if compile_only:
        logger.debug("Computing analytical success probability")
        evaluated_lcu.success_probability_analytical(func_approx_eval, coeffs)
        logger.debug(f"   p_success={evaluated_lcu.success_probability}")
        max_error_f = None
        max_error_approx = None
    else:
        logger.debug("Using success probability from statevector simulation")
        logger.debug(f"   p_success={evaluated_lcu.success_probability}")
        func_approx_eval_norm = func_approx_eval / np.sqrt(
            np.sum(np.abs(func_approx_eval) ** 2)
        )
        max_error_f = np.max(np.abs(f_eval - (eval_real + 1j * eval_imag)))
        max_error_approx = np.max(
            np.abs(func_approx_eval_norm - (eval_real + 1j * eval_imag))
        )
    n_1qb = evaluated_lcu.n_1qb_gates
    n_2qb = evaluated_lcu.n_2qb_gates
    n_qubits = evaluated_lcu.n_qubits
    success_probability = evaluated_lcu.success_probability

    return {
        "n_1qb": n_1qb,
        "n_2qb": n_2qb,
        "n_qubits": n_qubits,
        "success_probability": success_probability,
        "max_error_f": max_error_f,
        "max_error_approx": max_error_approx,
        "eval_real": eval_real,
        "eval_imag": eval_imag,
    }
