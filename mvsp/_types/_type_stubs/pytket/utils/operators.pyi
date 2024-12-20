from pytket.pauli import QubitPauliString
from numpy.typing import NDArray
import numpy as np
from typing import Self
from mvsp._types._core_types import CoeffType

from scipy.sparse import csc_matrix

class QubitPauliOperator:
    def __init__(
        self,
        dictionary: dict[QubitPauliString, CoeffType],
    ) -> None:
        self._dict = dictionary

    def state_expectation(self, state: NDArray[np.complex128]) -> complex: ...
    def to_sparse_matrix(self, n_qubits: int) -> csc_matrix: ...
    def __add__(self, other: QubitPauliOperator) -> Self: ...
