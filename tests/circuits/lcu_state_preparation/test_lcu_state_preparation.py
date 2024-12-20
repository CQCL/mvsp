"""Tests for circuits/lcu_state_preparation/lcu_state_preparation.py."""

from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
from numpy.typing import NDArray

from qtmlib.circuits.lcu_state_preparation.lcu_state_preparation import (
    LCUStatePreparationBox,
    SeparableLCUStatePreparationBox,
)
from qtmlib.circuits.lcu_state_preparation.lcu_state_preparation_block_encoding import (
    ChebychevBlockEncoding,
    FourierBlockEncoding,
)
from qtmlib.circuits.utils.lcu_state_preparation_utils import (
    create_einsum_string,
)
from qtmlib.utils.linalg_utils import (
    get_projector_matrix,
)

coeffs_fourier = {
    (8,): np.array(
        [
            -0.17127856 - 0.97059215j,
            -0.11346184 + 0.22676301j,
            0.15654503 + 0.7389392j,
            0.40605496 - 0.00193564j,
            -0.08136622 + 0.30454794j,
            -0.6745198 + 0.37244842j,
            -0.10471255 + 0.43750408j,
            0.21022982 - 0.6782518j,
        ]
    ),
    (3, 7): np.array(
        [
            [
                0.62937788 + 0.34417013j,
                -0.56935792 + 0.23265085j,
                0.73292403 + 0.41309417j,
                0.17229402 + 0.36604647j,
                0.2665281 + 0.36674144j,
                -0.08832155 + 0.03148176j,
                -0.13585525 + 0.17131375j,
            ],
            [
                0.33999182 - 0.40572039j,
                -0.23169221 + 0.47541383j,
                0.05021443 - 0.71130622j,
                -0.28529085 + 0.02509597j,
                0.09381541 - 0.0683472j,
                0.1763995 + 0.06282024j,
                0.25980122 + 0.34729755j,
            ],
            [
                0.07261735 - 0.1707416j,
                -0.08462347 + 0.08990364j,
                -0.72353472 + 0.04646923j,
                0.82473724 + 0.38498549j,
                -0.01037039 - 0.06002271j,
                0.00932944 - 0.04077063j,
                -0.11859345 + 0.25993853j,
            ],
        ]
    ),
    (2, 3, 5): np.array(
        [
            [
                [
                    0.44027719 - 0.34771439j,
                    -0.43945611 + 0.31791185j,
                    -0.14579151 - 0.03336436j,
                    0.71058444 - 0.20744203j,
                    -0.15544054 + 0.01831235j,
                ],
                [
                    -0.18188354 - 0.04234521j,
                    0.27992475 + 0.51494864j,
                    0.05936555 + 0.61629472j,
                    -0.15004201 - 0.11621398j,
                    0.0658536 - 0.93282939j,
                ],
                [
                    0.69948629 - 0.11023551j,
                    -0.24249192 - 0.76416425j,
                    -0.7119519 + 0.32047359j,
                    0.11015826 + 0.51157957j,
                    -0.50855597 - 0.37484208j,
                ],
            ],
            [
                [
                    -0.0131551 + 0.01334635j,
                    0.13860979 - 0.5662966j,
                    0.5039112 - 0.34329343j,
                    0.48779661 + 0.45607645j,
                    0.20350914 - 0.00671929j,
                ],
                [
                    0.31281693 + 0.07109167j,
                    -0.90721497 + 0.35302847j,
                    0.20771337 - 0.1783998j,
                    0.69369284 + 0.09361666j,
                    -0.09266342 - 0.24859678j,
                ],
                [
                    -0.55993588 - 0.12022556j,
                    0.86475372 + 0.11328346j,
                    -0.66120448 - 0.48003407j,
                    0.01289186 + 0.0262336j,
                    -0.04635647 - 0.4218215j,
                ],
            ],
        ]
    ),
}

coeffs_fourier_sep = {
    (8,): [
        np.array(
            [
                0.05707272 - 0.03353304j,
                0.69115973 - 0.13566778j,
                0.48206485 - 0.200016j,
                0.21205383 - 0.11848204j,
                0.06997989 - 0.0681886j,
                0.24524896 - 0.28349393j,
                0.21079446 + 0.6268635j,
                -0.00210063 + 0.34059913j,
            ]
        )
    ],
    (3, 7): [
        np.array(
            [
                0.32383164 - 0.10151252j,
                -0.67959841 + 0.37570071j,
                0.46889263 + 0.16893796j,
            ]
        ),
        np.array(
            [
                0.19293248 + 0.62713477j,
                0.17389628 + 0.1470449j,
                0.34543781 - 0.03235117j,
                0.40648575 - 0.24030835j,
                -0.12119799 + 0.60096322j,
                0.72830501 - 0.08484854j,
                -0.39086511 - 0.7471735j,
            ]
        ),
    ],
    (2, 3, 5): [
        np.array([-0.10085174 + 0.23259693j, -0.24247727 + 0.00409608j]),
        np.array(
            [0.5939308 - 0.544755j, 0.20135081 + 0.68787833j, 0.58031434 + 0.1567956j]
        ),
        np.array(
            [
                0.0584636 + 0.33241982j,
                0.08158119 - 0.01134815j,
                0.17877227 + 0.10557574j,
                0.20676343 - 0.95809369j,
                0.2031021 + 0.2727776j,
            ]
        ),
    ],
}

coeffs_cheb = {
    (8,): np.array(
        [
            0.08402472 + 0.02696278j,
            0.25998584 + 0.58607263j,
            0.76330584 + 0.40953835j,
            -0.42386848 + 0.78891378j,
            0.29322543 + 0.52235372j,
            -0.13545858 + 0.04397038j,
            0.02460477 - 0.84542177j,
            0.29747009 + 0.30975595j,
        ]
    ),
    (3, 7): np.array(
        [
            [
                -0.48671806 + 0.36964815j,
                0.40552801 - 0.56557467j,
                -0.18042622 + 0.13257534j,
                -0.07328804 + 0.01708539j,
                -0.00282363 + 0.02104235j,
                0.44835075 - 0.12740526j,
                -0.54793573 + 0.19101932j,
            ],
            [
                -0.00652878 - 0.03324221j,
                -0.47614447 + 0.23998946j,
                -0.12360664 + 0.56714067j,
                0.05407834 + 0.04489617j,
                -0.54782733 + 0.70474794j,
                0.06118513 - 0.08772823j,
                -0.1286732 - 0.15330703j,
            ],
            [
                -0.35845053 - 0.02460834j,
                0.45047396 - 0.43442366j,
                0.31295037 - 0.13642886j,
                0.07526407 + 0.54259133j,
                -0.26376034 - 0.48288314j,
                0.25216276 - 0.15389337j,
                0.35810936 - 0.0725207j,
            ],
        ]
    ),
}

coeffs_cheb_sep = {
    (8,): [
        np.array(
            [
                -0.37535691 - 0.67382692j,
                -0.01763648 - 0.01093608j,
                0.25033668 + 0.58210105j,
                -0.71692611 + 0.21615781j,
                -0.49835362 - 0.01236554j,
                0.22413736 + 0.01720399j,
                -0.07593443 + 0.18292857j,
                0.57823119 + 0.4940199j,
            ]
        )
    ],
    (3, 7): [
        np.array(
            [
                0.00579675 + 7.71298861e-01j,
                0.02075104 - 1.94665337e-04j,
                -0.09992399 - 6.25719810e-01j,
            ]
        ),
        np.array(
            [
                0.19748198 + 0.01515802j,
                -0.29157644 + 0.70241736j,
                0.1285749 + 0.10984976j,
                0.0155952 + 0.08695236j,
                0.17825503 - 0.66177279j,
                -0.82895986 + 0.47093993j,
                0.00087155 + 0.00385087j,
            ]
        ),
    ],
}


def fouriervalnd(
    coeffs: NDArray[np.complex128] | NDArray[np.float64],
    dims_variables: list[int],
    fourier_indices: list[list[int]],
) -> NDArray[np.complex128]:
    """Get array of the Fourier transform, specified by Fourier coefficients.

    Args:
    ----
        coeffs (NDArray): Fourier coefficients.
        dims_variables (list[int]): Dimensions of the variables.
        fourier_indices (list | None = None): Fourier indices, if None fourier_indices
            is set to [list(range(dim)) for dim in f_fourier.shape]. Defaults to None.

    Returns:
    -------
        NDArray: Array representation of the Fourier transform specified by the given
            Fourier coefficients.

    """

    def _custom_FT(
        coeffs: NDArray[np.complex128] | NDArray[np.float64],
        dims_fourier_variables: tuple[int, ...],
        fourier_indices: list[list[int]],
    ) -> Callable[[list[np.complex128] | NDArray[np.complex128]], np.complex128]:
        def _generate_index_list(k: tuple[int, ...]) -> list[int]:
            return [fourier_indices[i][k[i]] for i in range(len(k))]

        def _fun(x: list[np.complex128] | NDArray[np.complex128]) -> np.complex128:
            return np.sum(
                [
                    coeffs[k] * np.exp(1j * np.dot(_generate_index_list(k), x))
                    for k in product(*[range(dims) for dims in dims_fourier_variables])
                ]
            )

        return _fun

    dims_fourier_variables = tuple(coeffs.shape)
    fun = _custom_FT(coeffs, dims_fourier_variables, fourier_indices)
    fun_vec = np.vectorize(fun, signature="(m)->()")

    meshes = np.meshgrid(*[np.linspace(0, np.pi, dims) for dims in dims_variables])
    meshes_reshape = [mesh.reshape([*list(mesh.shape), 1]) for mesh in meshes]
    flattened_mesh = np.concatenate(meshes_reshape, axis=-1)

    fun_eval = fun_vec(flattened_mesh)
    if len(fun_eval.shape) > 1:
        fun_eval = np.swapaxes(fun_eval, 0, 1)
    return fun_eval


@pytest.mark.parametrize(
    ("dims_fourier_variables", "dims_variables"),
    [
        ((8,), (16,)),
        ((3, 7), (8, 8)),
        ((2, 3, 5), (2, 2, 4)),
    ],
)
def test_LCUStatePreparationBox_Fourier(
    dims_fourier_variables: tuple[int, ...], dims_variables: tuple[int, ...]
):
    """Test LCUStatePreparationBox for Fourier.

    Args:
    ----
        dims_fourier_variables (tuple): Dimensions of the Fourier variables.
        dims_variables (tuple): Dimensions of the function that should be prepared,
            i.e. the discretizations into each direction.

    """
    coeffs = coeffs_fourier[dims_fourier_variables]  # type: ignore
    ### Generate circuit
    n_qubits = [int(np.ceil(np.log2(d))) for d in dims_variables]
    dims_fourier_variables = coeffs.shape
    min_fourier_indices = [int(-deg / 2) for deg in dims_fourier_variables]

    lcuspbox = LCUStatePreparationBox(
        coeffs,
        [FourierBlockEncoding(n_block_qubits=n) for n in n_qubits],
        min_basis_indices=min_fourier_indices,
    )

    fourier_indices_extended = [
        list(
            range(
                min_fourier_indices[i],
                min_fourier_indices[i] + lcuspbox.coeffs_abs.shape[i],
            )
        )
        for i in range(len(min_fourier_indices))
    ]

    f = fouriervalnd(
        coeffs,
        lcuspbox.dims_variables,
        fourier_indices=fourier_indices_extended,
    )

    vec = lcuspbox.get_statevector()
    n_qubits_fourier = sum([len(r) for r in lcuspbox.qreg.coeffs])
    n_qubits_state = sum([len(r) for r in lcuspbox.qreg.state])
    projector: NDArray[np.float64] = np.array(
        get_projector_matrix(
            list(range(n_qubits_fourier)),
            list(range(n_qubits_fourier, n_qubits_fourier + n_qubits_state)),
        )
    )
    vec_projected = np.array(projector.conjugate().transpose() @ vec)

    success_probability = np.power(np.linalg.norm(vec_projected), 2)
    vec_projected_normed = vec_projected / np.sqrt(success_probability)

    f_vec = f / np.linalg.norm(f)
    np.testing.assert_allclose(
        f_vec, vec_projected_normed.reshape(dims_variables), atol=1e-10
    )


@pytest.mark.parametrize(
    ("dims_fourier_variables", "dims_variables"),
    [
        ((8,), (16,)),
        ((3, 7), (8, 8)),
        ((2, 3, 5), (2, 2, 4)),
    ],
)
def test_SeparableLCUStatePreparationBox_Fourier(
    dims_fourier_variables: tuple[int, ...], dims_variables: tuple[int, ...]
):
    """Test LCUStatePreparationBox for Fourier.

    Args:
    ----
        dims_fourier_variables (tuple): Dimensions of the Fourier variables.
        dims_variables (tuple): Dimensions of the function that should be prepared,
            i.e. the discretizations into each direction.

    """
    coeffs = coeffs_fourier_sep[dims_fourier_variables]  # type: ignore
    coeffs_array = np.einsum(create_einsum_string(len(coeffs)), *coeffs)  # type: ignore

    ### Generate circuit
    n_qubits = [int(np.ceil(np.log2(d))) for d in dims_variables]
    dims_fourier_variables = coeffs_array.shape
    min_fourier_indices = [int(-deg / 2) for deg in dims_fourier_variables]

    lcuspbox = SeparableLCUStatePreparationBox(
        coeffs,
        [FourierBlockEncoding(n_block_qubits=n) for n in n_qubits],
        min_basis_indices=min_fourier_indices,
    )

    fourier_indices_extended = [
        list(
            range(
                min_fourier_indices[i],
                min_fourier_indices[i] + lcuspbox._coeffs_abs[i].shape[0],
            )
        )
        for i in range(len(min_fourier_indices))
    ]

    f = fouriervalnd(
        coeffs_array,
        [x for y in lcuspbox._dims_variables for x in y],
        fourier_indices=fourier_indices_extended,
    )

    vec = lcuspbox.get_statevector()
    n_qubits_fourier = sum([len(r) for r in lcuspbox.qreg.coeffs])
    n_qubits_state = sum([len(r) for r in lcuspbox.qreg.state])
    projector: NDArray[np.float64] = np.array(
        get_projector_matrix(
            list(range(n_qubits_fourier)),
            list(range(n_qubits_fourier, n_qubits_fourier + n_qubits_state)),
        )
    )
    vec_projected = np.array(projector.conjugate().transpose() @ vec)

    success_probability = np.power(np.linalg.norm(vec_projected), 2)
    vec_projected_normed = vec_projected / np.sqrt(success_probability)

    f_vec = f / np.linalg.norm(f)
    np.testing.assert_allclose(
        f_vec, vec_projected_normed.reshape(dims_variables), atol=1e-10
    )


def chebvalnd(
    meshgrid: NDArray[np.float64], coeffs: NDArray[np.complex128] | NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Evaluate a n-dimensional Chebyshev polynomial on a meshgrid of points.

    Args:
    ----
        meshgrid (NDArray): Points where to evaluate the polynomial.
        coeffs (NDArray): Coefficients of polynomial.

    """
    result = np.zeros_like(meshgrid[0], dtype=np.complex128)
    for i, c in np.ndenumerate(coeffs):
        aux = np.ones_like(meshgrid[0], dtype=np.complex128)
        for j, d in enumerate(meshgrid):
            ind = np.zeros(coeffs.shape[j], dtype=np.complex128)
            ind[i[j]] = 1
            aux *= np.polynomial.chebyshev.Chebyshev(ind)(d)  # type: ignore
        result += c * aux  # type: ignore
    return result  # type: ignore


@pytest.mark.parametrize(
    ("dims_cheb_variables", "dims_variables"),
    [
        ((8,), (16,)),
        ((3, 7), (4, 4)),
    ],
)
def test_LCUStatePreparationBox_Chebyshev(
    dims_cheb_variables: tuple[int, ...], dims_variables: tuple[int, ...]
):
    """Test LCUStatePreparationBox for Chebyshev.

    Args:
    ----
        dims_cheb_variables (tuple): Dimensions of the Chebyshev variables.
        dims_variables (tuple): Dimensions of the function that should be prepared,
            i.e. the discretizations into each direction.

    """
    coeffs = coeffs_cheb[dims_cheb_variables]  # type: ignore
    ### Generate circuit
    n_qubits = [int(np.ceil(np.log2(d))) for d in dims_variables]

    lcuspbox = LCUStatePreparationBox(
        coeffs,
        [ChebychevBlockEncoding(n_block_qubits=n) for n in n_qubits],
    )
    meshes = np.meshgrid(*[np.linspace(-1, 1, dims) for dims in dims_variables])

    f = chebvalnd(np.array(meshes), coeffs)

    vec = lcuspbox.get_statevector()
    n_qubits_cheb = sum([len(r) for r in lcuspbox.qreg.coeffs])
    n_qubits_state = sum([len(r) for r in lcuspbox.qreg.state])
    n_qubits_block_encoding = sum([len(r) for r in lcuspbox.qreg.block])

    projector = np.array(
        get_projector_matrix(
            list(range(n_qubits_cheb + n_qubits_block_encoding)),
            list(
                range(
                    n_qubits_cheb + n_qubits_block_encoding,
                    n_qubits_cheb + n_qubits_block_encoding + n_qubits_state,
                )
            ),
        )
    )
    vec_projected = projector.transpose() @ vec

    vec_projected = np.array(projector.conjugate().transpose() @ vec)

    success_probability = np.power(np.linalg.norm(vec_projected), 2)
    vec_projected_normed = vec_projected / np.sqrt(success_probability)

    f_vec = f / np.linalg.norm(f)

    np.testing.assert_allclose(
        f_vec.transpose(), vec_projected_normed.reshape(dims_variables), atol=1e-10
    )


@pytest.mark.parametrize(
    ("dims_cheb_variables", "dims_variables"),
    [
        ((8,), (16,)),
        ((3, 7), (4, 4)),
    ],
)
def test_SeparableLCUStatePreparationBox_Chebyshev(
    dims_cheb_variables: tuple[int, ...], dims_variables: tuple[int, ...]
):
    """Test LCUStatePreparationBox for Chebyshev.

    Args:
    ----
        dims_cheb_variables (tuple): Dimensions of the Chebyshev variables.
        dims_variables (tuple): Dimensions of the function that should be prepared,
            i.e. the discretizations into each direction.

    """
    coeffs = coeffs_cheb_sep[dims_cheb_variables]  # type: ignore
    coeffs_array = np.einsum(create_einsum_string(len(coeffs)), *coeffs)  # type: ignore

    ### Generate circuit
    n_qubits = [int(np.ceil(np.log2(d))) for d in dims_variables]

    lcuspbox = SeparableLCUStatePreparationBox(
        coeffs,
        [ChebychevBlockEncoding(n_block_qubits=n) for n in n_qubits],
    )

    meshes = np.meshgrid(*[np.linspace(-1, 1, dims) for dims in dims_variables])
    f = chebvalnd(np.array(meshes), coeffs_array)

    vec = lcuspbox.get_statevector()
    n_qubits_cheb = sum([len(r) for r in lcuspbox.qreg.coeffs])
    n_qubits_state = sum([len(r) for r in lcuspbox.qreg.state])
    n_qubits_block_encoding = sum([len(r) for r in lcuspbox.qreg.block])

    projector = np.array(
        get_projector_matrix(
            list(range(n_qubits_cheb + n_qubits_block_encoding)),
            list(
                range(
                    n_qubits_cheb + n_qubits_block_encoding,
                    n_qubits_cheb + n_qubits_block_encoding + n_qubits_state,
                )
            ),
        )
    )
    vec_projected = projector.transpose() @ vec

    vec_projected = np.array(projector.conjugate().transpose() @ vec)

    success_probability = np.power(np.linalg.norm(vec_projected), 2)
    vec_projected_normed = vec_projected / np.sqrt(success_probability)

    f_vec = f / np.linalg.norm(f)
    np.testing.assert_allclose(
        f_vec.transpose(), vec_projected_normed.reshape(dims_variables), atol=1e-10
    )
