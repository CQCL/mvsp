import numpy as np
from typing import Any, Optional, Literal
from numpy.typing import NDArray

def angle(z:NDArray[np.float64 | np.complex128], deg: bool = False) -> NDArray[np.float64]: ...

def append(arr: NDArray[Any], values: NDArray[Any], axis: Optional[int]=None)  -> NDArray[Any]:...

def meshgrid(*xi: NDArray[Any], copy: bool = True, sparse: bool = False, indexing: Literal['xy', 'ij'] = 'xy') -> tuple[NDArray[Any]]:...