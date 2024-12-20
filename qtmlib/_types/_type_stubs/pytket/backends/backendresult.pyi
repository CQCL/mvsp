import numpy as np
from numpy.typing import NDArray
from typing import Any
from collections.abc import Sequence
from collections import Counter
from pytket.utils.outcomearray import OutcomeArray
from pytket._tket.circuit import Circuit
from pytket._tket.unit_id import Bit
from pytket.circuit import Qubit, BasisOrder

class BackendResult:
    def __init__(
        self,
        *,
        q_bits: Sequence[Qubit] | None = None,
        c_bits: Sequence[Bit] | None = None,
        counts: Counter[OutcomeArray] | None = None,
        shots: OutcomeArray | None = None,
        state: Any = None,
        unitary: Any = None,
        density_matrix: Any = None,
        ppcirc: Circuit | None = None,
    ) -> None: ...
    def get_state(
        self,
        qbits: Sequence[Qubit] | None = None,
        basis: BasisOrder = BasisOrder.ilo,
    ) -> NDArray[np.complex128]: ...
    def get_unitary(
        self,
        qbits: Sequence[Qubit] | None = None,
        basis: BasisOrder = BasisOrder.ilo,
    ) -> NDArray[np.complex128]: ...
