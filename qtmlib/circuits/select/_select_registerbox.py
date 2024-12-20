from abc import abstractmethod
from pytket.circuit import Op, QubitRegister
from qtmlib.circuits.core import RegisterCircuit
from qtmlib.circuits.core import RegisterBox
from dataclasses import dataclass
from typing import Any


@dataclass
class SelectQRegs:
    """SelectBox qubit registers.

    Attributes
    ----------
        prepare (QubitRegister): The prepare register (default - p)
        state (QubitRegister): The state register (default - q)

    """

    prepare: QubitRegister
    state: QubitRegister


class SelectBox(RegisterBox):
    """Abstract SelectRegisterBox base class."""

    """This class inherits from RegisterBox. To be used in the LCUBox
    class. The circuit is constructed by adding the select box to the prepare
    register. The circuit construction logic is done in this class. The child
    classes set the box to be used in the select box.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).

    Args:
        select_box (CircBox): The select box to be used.
        n_state_qubits (int): The number of state qubits.
        prepare_qreg_str (str): The prepare register string. Defaults to "p".
        state_qreg_str (str): The state register string. Defaults to "q".
        is_hermitian (bool): Whether if the operator is Hermitian. Defaults to False.
    """

    def __init__(
        self,
        select_box: Op,
        n_state_qubits: int,
        prepare_qreg_str: str = "p",
        state_qreg_str: str = "q",
        is_hermitian: bool = False,
    ):
        """Initialise the SelectRegisterBox."""
        self._select_box = select_box

        circ = RegisterCircuit(f"{self.__repr__()}")
        state_qreg = circ.add_q_register(state_qreg_str, n_state_qubits)

        self._n_state_qubits = n_state_qubits
        self._is_hermitian = is_hermitian

        prepare_qreg = circ.add_q_register(prepare_qreg_str, self.n_prep_qubits)

        qreg = SelectQRegs(prepare_qreg, state_qreg)

        circ.add_gate(self.select_box, circ.qubits)

        super().__init__(qreg, circ)

    @property
    def select_box(self) -> Op:
        """Return the select box."""
        return self._select_box

    @property
    @abstractmethod
    def operator(self) -> Any:
        """Return the operator."""
        pass

    @property
    @abstractmethod
    def n_prep_qubits(self) -> Any:
        """Return the number of prepare qubits."""
        pass

    @property
    def n_state_qubits(self) -> int:
        """Return the number of state qubits."""
        return self._n_state_qubits

    @property
    def is_hermitian(self) -> bool:
        """Return whether the operator is Hermitian or not."""
        return self._is_hermitian
