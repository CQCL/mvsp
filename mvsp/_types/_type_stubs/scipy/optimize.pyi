from scipy.linalg import LinearOperator
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Optional, Callable, Any, Sequence, Dict

class OptimizeResult:
    @property
    def x(self) -> NDArray[np.float64]: ...
    @property
    def success(self) -> bool: ...
    @property
    def status(self) -> int: ...
    @property
    def message(self) -> str: ...
    @property
    def fun(self) -> NDArray[np.float64]: ...
    @property
    def jac(self) -> NDArray[np.float64]: ...
    @property
    def hess(self) -> NDArray[np.float64]: ...
    @property
    def hess_inv(self) -> LinearOperator | NDArray[np.float64]: ...
    @property
    def nfev(self) -> int: ...
    @property
    def njev(self) -> int: ...
    @property
    def nhev(self) -> int: ...
    @property
    def nit(self) -> int: ...
    @property
    def maxcv(self) -> float: ...

class HessianUpdateStrategy: ...

class Bounds:
    def __init__(
        self,
        lb: Optional[float] = -np.inf,
        ub: Optional[float] = np.inf,
        keep_feasible: Optional[bool] = False,
    ) -> None: ...

def minimize(
    fun: Callable[[NDArray[np.float64]], np.float64],
    x0: NDArray[np.float64],
    args: Optional[tuple[Any]] = tuple(),
    method: Optional[str | Callable[[Any], OptimizeResult]] = None,
    jac: Optional[
        Callable[[NDArray[np.float64] | tuple[Any]], np.float64]
        | bool
        | Literal["2-point", "3-point", "cs"]
    ] = None,
    hess: Optional[
        Callable[[NDArray[np.float64], tuple[Any]], LinearOperator]
        | HessianUpdateStrategy
        | Literal["2-point", "3-point", "cs"]
    ] = None,
    hessp: Optional[
        Callable[
            [NDArray[np.float64], NDArray[np.float64], tuple[Any]], NDArray[np.float64]
        ]
    ] = None,
    bounds: Optional[Sequence[Optional[tuple[float, float]]] | Bounds] = None,
    constraints: Optional[tuple[Any]] = tuple(),
    tol: Optional[float] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> OptimizeResult: ...
