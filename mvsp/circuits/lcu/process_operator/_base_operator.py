"""Base class for LCU operators."""

from abc import ABC, abstractmethod
from pytket.utils.operators import QubitPauliOperator
from math import ceil, log2
from typing import Any


class BaseLCUOperator(ABC):
    def __init__(self, qubit_operator: QubitPauliOperator) -> None:
        self._qubit_operator = qubit_operator

    @property
    @abstractmethod
    def terms(self) -> Any:
        pass

    @property
    def n_terms(self) -> int:
        if self.terms is None:
            raise ValueError("Terms not set")
        return len(self.terms)

    @property
    def qubit_operator(self) -> QubitPauliOperator:
        return self._qubit_operator

    @property
    def n_prep_qubits(self) -> int:
        if self.terms is None:
            raise ValueError("Terms not set")
        return int(ceil(log2(self.n_terms)))
