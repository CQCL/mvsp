from typing import Any
import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence

def chebval(
    x: NDArray[np.complex128] | NDArray[np.float64],
    c: Sequence[int] | NDArray[np.float64] | NDArray[np.complex128],
) -> NDArray[np.complex128]: ...

class Chebyshev:
    def __init__(
        self,
        coef: NDArray[np.complex128],
        domain: NDArray[np.float64] | None = None,
        window: NDArray[np.complex128] | None = None,
        symbol: str = "x",
    ) -> None: ...
    def __call__(self, arg: Any) -> Any: ...

def cheb2poly(c: NDArray[np.complex128] | NDArray[np.float64]) -> NDArray[np.complex128] | NDArray[np.float64]: ...