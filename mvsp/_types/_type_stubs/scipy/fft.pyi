import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional

def dct(
    x: NDArray[np.float64],
    type: Literal[1, 2, 3, 4] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["backward", "ortho", "forward"]] = None,
    overwrite_x: Optional[bool] = False,
    workers: Optional[int] = None,
    orthogonalize: Optional[bool] = None,
) -> NDArray[np.float64]: ...
