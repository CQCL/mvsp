import numpy as np
from numpy.typing import NDArray
from typing import Any

def svd(
    a: NDArray[np.complex128],
    full_matrices: bool = True,
    compute_uv: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]: ...
def expm(a: NDArray[np.complex128]) -> NDArray[np.complex128]: ...
def norm(
    x: NDArray[np.complex128],
    ord: int | str | None = None,
    axis: int | tuple[int, int] | None = None,
    keepdims: bool = False,
    check_finite: bool = True,
) -> NDArray[np.float64] | np.float64: ...

class LinearOperator:
    def __init__(self, dtype: np.dtype[Any], shape: tuple[int]) -> None: ...
