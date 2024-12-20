"""Class for the LCU oracle."""

from pytket.circuit import QubitRegister
from qtmlib.circuits.core import QControlRegisterBox, RegisterBox
from qtmlib.circuits.select import SelectBox
from qtmlib.circuits.prepare import PrepareBox
from qtmlib.circuits.core import RegisterCircuit
from pytket.utils.operators import QubitPauliOperator
from dataclasses import dataclass


@dataclass
class LCUQRegs:
    """LCUBox qubit registers.

    Attributes
    ----------
        prepare (QubitRegister): The prepare register (default - p)
        state (QubitRegister): The state register (default - q)

    """

    prepare: QubitRegister
    state: QubitRegister


class LCUBox(RegisterBox):
    """Abstract LCUBox base class.

    This class inherits from RegisterBox. The circuit is constructed
    by adding the prepare box to the prepare register. And the select box to
    the prepare and select registers. The circuit construction logic is done
    in the class. The child classes set the boxes to be used in the prepare
    and select boxes in the init of this class.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).

    Args:
    ----
        prepare_box (PrepareBox): The prepare box to be used in the
            circuit.
        select_box (SelectBox): The select box to be used in the circuit.

    Raises:
    ------
        ValueError: If the number of prepare qubits in the prepare box and the
            select box are not the same.

    """

    def __init__(self, prepare_box: PrepareBox, select_box: SelectBox):
        """Initialise the LCUBox."""
        self._prepare_box = prepare_box
        self._select_box = select_box

        self._is_hermitian = select_box.is_hermitian

        if self.prepare_box.qreg.prepare != self.select_box.qreg.prepare:
            raise ValueError(
                "The number of prepare qubits in"
                "the prepare box and the select box must be the same."
            )

        circ = RegisterCircuit(self.__class__.__name__)
        prepare_qreg = circ.add_q_register(prepare_box.qreg.prepare)
        state_qreg = circ.add_q_register(select_box.qreg.state)

        qregs = LCUQRegs(prepare_qreg, state_qreg)

        circ.add_registerbox(self.prepare_box)

        circ.add_registerbox(self.select_box)

        circ.add_registerbox(self.prepare_box.dagger)

        self._postselect = {p: 0 for p in self.prepare_box.qreg.prepare}

        super().__init__(qregs, circ)

    @property
    def prepare_box(self) -> PrepareBox:
        """Return the prepare box."""
        return self._prepare_box

    @property
    def select_box(self) -> SelectBox:
        """Return the select box."""
        return self._select_box

    @property
    def l1_norm(self) -> float:
        """Return the l1 norm of the operator."""
        return self.prepare_box.l1_norm

    @property
    def operator(self) -> QubitPauliOperator:
        """Return the operator."""
        return self.select_box.operator

    @property
    def n_state_qubits(self) -> int:
        """Return the number of state qubits."""
        return self.select_box.n_state_qubits

    @property
    def n_prepare_qubits(self) -> int:
        """Return the number of prepare qubits."""
        return len(self.prepare_box.qreg.prepare)

    @property
    def is_hermitian(self) -> bool:
        """Return whether the LCUBox is Hermitian or not."""
        return self._is_hermitian

    def qcontrol(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ) -> QControlRegisterBox:
        """Return a QControlLCUBox.

        Generates a QControlLCUBox from the LCUBox. The QControlLCUBox
        is the controlled version of the LCUBox. The control qubits are
        added to the prepare register.

        Args:
        ----
            n_control (int): The number of control qubits to be used in the
                QControlLCUBox.
            control_qreg_str (str): The string of the control qreg. default - 'a'.
            control_index (int): The binary control index to be used in the control

        Returns:
        -------
            QControlLCUBox: The QControlLCUBox of the LCUBox.

        """
        return QControlLCUBox(self, n_control, control_qreg_str, control_index)


@dataclass
class QControlLCUQRegs(LCUQRegs):
    """QControlQubitiseBox registers.

    Attributes
    ----------
        prepare (QubitRegister): The prepare register (default - p)
        state (QubitRegister): The state register (default - q)
        control (QubitRegister): The control register (default - a)

    """

    control: QubitRegister


class QControlLCUBox(QControlRegisterBox):
    """Constructs a Controlled LCUBox.

    This box structure is the fact that the LCUBox is constructed by
    Prepare x Select x Prepare^dagger. Therefore because of this congujation
    structure only the select box needs to be controlled.

    Args:
    ----
        lcu_box (LCUBox): The LCUBox.
        n_control (int): The number of control qubits.
    x
    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).
        control_qreg (QubitRegister): The control register (default - a).
        control_index (int): The binary control index to be used in the control.

    """

    def __init__(
        self,
        lcu_box: LCUBox,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ):
        """Construct a controlled LCUBox."""
        circ = lcu_box.initialise_circuit()
        circ.name = f"Q{n_control}{lcu_box.__class__.__name__}"
        control_qreg = circ.add_q_register(control_qreg_str, n_control)

        qregs = QControlLCUQRegs(
            lcu_box.qreg.prepare, lcu_box.qreg.state, control=control_qreg
        )

        circ.add_registerbox(lcu_box.prepare_box)

        circ.add_registerbox(lcu_box.select_box.qcontrol(n_control, control_qreg_str))

        circ.add_registerbox(lcu_box.prepare_box.dagger)

        super().__init__(lcu_box, qregs, circ, n_control, control_index)
