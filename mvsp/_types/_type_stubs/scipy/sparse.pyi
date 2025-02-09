import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import Self

class csc_matrix:
    def __init__(
        self,
        arg1: tuple[int, int] | NDArray[np.complex128],
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
    ) -> None: ...
    def get_shape(self) -> tuple[int, int]: ...
    def __mul__(self, other: float) -> Self: ...
    def __rmul__(self, other: float) -> Self: ...
    def __truediv__(self, other: float) -> Self: ...
    def todense(self) -> NDArray[np.complex128]: ...

class csr_matrix:
    def __init__(
        self,
        arg1: (
            tuple[int, int]
            | NDArray[np.complex128]
            | tuple[list[float], tuple[tuple[int, ...], tuple[int, ...]]]
        ),
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
    ) -> None: ...
    def get_shape(self) -> tuple[int, int]: ...
    def __mul__(self, other: float) -> Self: ...
    def __rmul__(self, other: float) -> Self: ...
    def __truediv__(self, other: float) -> Self: ...
    def todense(self) -> NDArray[np.complex128]: ...
