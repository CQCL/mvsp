"""Multiplexed operator class for LCUBoxes."""

from mvsp.circuits.lcu.process_operator import BaseLCUOperator
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Unitary1qBox, Op, OpType
import cmath
import numpy as np


class MulitplexedOperatorTerm:
    """A class to store a multiplexed operator term.

    The term is stored as a list of pytket Ops and a magnitude.
    The phase of the term is absorbed into the pauli Op making it
    a general SU(2) Op.
    """

    def __init__(
        self, qubit_term: QubitPauliString, coeff: float, n_system_qubits: int
    ):
        """Initialise the MulitplexedOperatorTerm.

        Args:
        ----
            qubit_term (QubitPauliString): The term to be applied.
            coeff (complex): The coefficient of the term.
            n_system_qubits (int): The number of qubits in the state register.

        """
        self._n_system_qubits = n_system_qubits
        self._is_hermitian = False
        self._magnitude, self._op_list = self._absorb_sign_phase(coeff, qubit_term)

    @property
    def magnitude(self) -> float:
        """Return the magnitude of the term."""
        return self._magnitude

    @property
    def is_hermitian(self) -> bool:
        """Return whether the operator is Hermitian or not."""
        return self._is_hermitian

    @property
    def op_list(self) -> list[Op]:
        """Return the list of pytket Ops."""
        return self._op_list

    # TODO the fact that his handles circuit logoc is not ideal
    def _optype_list(self, term: QubitPauliString) -> list[Op]:
        """Return a list of the optype Pauli operators.

        Forn a given Pauli return the equivalent pytket Op.

        Args:
        ----
            term (QubitPauliString): The term to be applied.

        """
        ops = {
            Pauli.I: Op.create(OpType.noop),
            Pauli.X: Op.create(OpType.X),
            Pauli.Y: Op.create(OpType.Y),
            Pauli.Z: Op.create(OpType.Z),
        }

        optype_list = [ops[Pauli.I] for _ in range(self._n_system_qubits)]
        for term_qubit, pauli in term.map.items():
            optype_list[term_qubit.index[0]] = ops[pauli]
        return optype_list

    def _absorb_sign_phase(
        self, coeff: float, term: QubitPauliString
    ) -> tuple[float, list[Op]]:
        """Return the new coefficient and Pauli operators.

        Absorb the sign and phase of the coefficient into
        the first Pauli pytket Ops making them general SU(2)

        Args:
        ----
            coeff (complex): The coefficient of the term.
            term (QubitPauliString): The term to be applied.

        """
        op_list = self._optype_list(term)
        mag, self._phase = cmath.polar(coeff)
        exp = cmath.exp(self._phase * 1j)
        if cmath.isclose(exp, 1) or cmath.isclose(exp, -1):
            self._is_hermitian = True
        op_list[0] = Unitary1qBox(op_list[0].get_unitary() * exp)
        return mag, op_list

    @property
    def phase(self) -> float:
        """Return the phase of the term."""
        return self._phase / np.pi


class MulitplexedOperator(BaseLCUOperator):
    """A class to store the multiplexed operator terms.

    The terms are stored as a list of MulitplexedOperatorTerm objects.
    """

    def __init__(self, qubit_operator: QubitPauliOperator, n_state_qubits: int) -> None:
        """Initialise the MulitplexedOperator.

        Args:
        ----
            qubit_operator (QubitPauliOperator): The operator to be applied.
            n_state_qubits (int): The number of qubits in the state register.

        """
        super().__init__(qubit_operator)

        # TODO: Make QPO have terms list dataclass
        self._terms = [
            MulitplexedOperatorTerm(pauli_term, coeff, n_state_qubits)  # type: ignore
            for pauli_term, coeff in qubit_operator._dict.items()  # type: ignore
        ]
        self._is_hermitian = all(term.is_hermitian for term in self._terms)

    @property
    def magnitudes(self):
        """Return a list of the magnitudes of the terms."""
        return [term.magnitude for term in self._terms]

    @property
    def terms(self) -> list[MulitplexedOperatorTerm]:
        """Return the list of terms."""
        return self._terms

    @property
    def is_hermitian(self) -> bool:
        """Return whether the operator is Hermitian or not."""
        return self._is_hermitian
