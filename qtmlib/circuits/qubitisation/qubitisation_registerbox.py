"""Contains the QubitiseBox and QControlQubitiseBox classes."""

from __future__ import annotations

from dataclasses import dataclass

from pytket.circuit import Qubit, QubitRegister

from qtmlib.circuits.core import (
    PowerBox,
    QControlRegisterBox,
    RegisterBox,
    QRegMap,
    extend_new_qreg_dataclass,
)
from qtmlib.circuits.lcu import LCUBox
from qtmlib.circuits.reflection import ReflectionBox


@dataclass
class QubitiseQRegs:
    """QubitiseBox qubit registers.

    Attributes
    ----------
        prepare (QubitRegister): The prepare register (default - p)
        state (QubitRegister): The state register (default - q)

    """

    prepare: QubitRegister
    state: QubitRegister


class QubitiseBox(RegisterBox):
    """Constructs a QSVTOracleBox from an LCUBox.

    Takes an LCUBox box and it reflects it about the |00...0> state on the
    prepare qubits by applying a CnZ gate on the prepare register.

    Args:
    ----
        lcu_box (LCUBox): The LCUBox to be converted to a QSVTOracleBox.

    Raises:
    ------
        ValueError: If the Select Box from the LCU is not Hermitian.

    """

    def __init__(self, lcu_box: LCUBox):
        """Initialise the QSVTOracleBox."""
        if not lcu_box.is_hermitian:
            raise ValueError("QubitiseBox only available for Hermitian LCUs.")
        self._lcu_box = lcu_box
        self._reflection_box = ReflectionBox(lcu_box.qreg.prepare.size)
        self._control_reflection = True

        circ = lcu_box.initialise_circuit()

        qregs = QubitiseQRegs(prepare=lcu_box.qreg.prepare, state=lcu_box.qreg.state)

        circ.add_registerbox(lcu_box)

        qreg_map = QRegMap(
            [self._reflection_box.qreg.reflection], [lcu_box.qreg.prepare]
        )
        circ.add_registerbox(self._reflection_box, qreg_map)

        super().__init__(qregs, circ)

    @property
    def lcu_box(self) -> LCUBox:
        """Return the LCU box."""
        return self._lcu_box

    @property
    def reflection_box(self) -> ReflectionBox:
        """Return the Reflection box."""
        return self._reflection_box

    @property
    def postselect(self) -> dict[Qubit, int]:
        """Return the postselect dictionary."""
        return self.lcu_box.postselect

    @property
    def control_reflection(self) -> bool:
        """Return the option to control either the reflection or the LCU."""
        return self._control_reflection

    @control_reflection.setter
    def control_reflection(self, value: bool):
        """Set value of control reflection property."""
        self._control_reflection = value

    def qcontrol(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ) -> QControlQubitiseBox:
        """Return a QControlQubitiseBox.

        This will use the .qcontrol() method of the register box to
        generate the QControlRegisterBox. But uses it power times.
        This is general is more efficient than qcontroling the PowerBox.
        This should be reviewed as pytket improves for larger circuits.

        Args:
        ----
            n_control (int): The number of n_control qubits to be used in the
                QControlRegisterBox.
            control_qreg_str (str): The string of the control qreg. default - 'a'.
            control_index (int): The binary control index to be used in the control

        Returns:
        -------
            QControlRegisterBox: The QControlRegisterBox of the PowerBox.

        """
        return QControlQubitiseBox(self, n_control, control_qreg_str, control_index)

    def _qcontrol_squared(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ) -> QControlSquareQubitiseBox:
        """Return a QControlSquareQubitiseBox.

        This will use the .qcontrol() method of the register box to
        generate the QControlRegisterBox. But uses it power times.
        This is general is more efficient than qcontroling the PowerBox.
        This should be reviewed as pytket improves for larger circuits.
        Only when B_V^2=1 and C_V^2=1.

        Args:
        ----
            n_control (int): The number of n_control qubits to be used in the
                QControlRegisterBox.
            control_qreg_str (str): The string of the control qreg. default - 'a'.
            control_index (int): The binary control index to be used in the control

        Returns:
        -------
            QControlRegisterBox: The QControlRegisterBox of the PowerBox.

        """
        return QControlSquareQubitiseBox(
            self, n_control, control_qreg_str, self._control_reflection, control_index
        )

    def power(self, power: int) -> PowerBox:
        """Return a power of the Qubitise.

        Generates a PowerBox from the QubitiseBox. When the PowerBox is applied
        As 2**power repetitions of the QubitiseBox. The power must be even.
        This is being investigated.

        When the power is of the shape 2**k, we apply the circuit twice and select
        the qcontrol_squared option as qcontrol to implement the decomposition.

        Args:
        ----
            power (int): The power to raise the Qubitise to.

        Raises:
        ------
            ValueError: If power is odd.

        """
        if (power & (power - 1) == 0) and power != 0 and power != 1:
            power_box = PowerBox(self, 2)
            power_box.qcontrol = self._qcontrol_squared
            return PowerBox(power_box, power // 2)
        else:
            return PowerBox(self, power)


class QControlQubitiseBox(QControlRegisterBox):
    """Constructs a Controlled QubitiseBox.

    Controls the LCUBox and adds a CnZ to the prepare register
    and the ancilla qubit.

    Args:
    ----
        qubitise_box (QubitiseBox): The QubitiseBox to be controlled.
        n_control (int): The number of ancilla qubits to be used in the
            QControlQubitiseBox.
        control_qreg_str (str): The string of the control qreg. default - 'a'.
        control_index (int): The binary control index to be used in the control

    """

    def __init__(
        self,
        qubitise_box: QubitiseBox,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ):
        """Construct a Controlled from an LCUBox."""
        if n_control != 1:
            raise ValueError("n_ancilla must be 1 for QControlQubitiseBox")
        self._qubitise_box = qubitise_box
        self._n_control = n_control
        self._control_qreg_str = control_qreg_str

        circ = qubitise_box.initialise_circuit()
        circ.name = f"Q{n_control}C{qubitise_box.__class__.__name__}"
        control_qreg = circ.add_q_register(control_qreg_str, 1)

        qregs = extend_new_qreg_dataclass(
            "QControlQubitiseQRegs", qubitise_box.qreg, {"control": control_qreg}
        )

        qc_lcu_box = qubitise_box.lcu_box.qcontrol(1, control_qreg_str)
        circ.add_registerbox(qc_lcu_box)

        qc_reflection_box = qubitise_box.reflection_box.qcontrol(1, control_qreg_str)
        qreg_map = QRegMap(
            [qc_reflection_box.qreg.reflection, qc_reflection_box.qreg.control],
            [qregs.prepare, qregs.control],
        )
        circ.add_registerbox(qc_reflection_box, qreg_map)

        super().__init__(qubitise_box, qregs, circ, n_control, control_index)

    def power(self, power: int) -> PowerBox:
        """Return a power of the QControlQubitiseBox.

        Generates a PowerBox from the QControlQubitiseBox. When the PowerBox is applied
        As 2**power repetitions of the QControlQubitiseBox. The power must be even.
        This is being investigated.

        When the power is of the shape 2**k, we apply the circuit twice and select
        the QControlSquareQubitiseBox option as qcontrol to implement the decomposition.

        Args:
        ----
            power (int): The power to raise the QControlQubitiseBox to.

        """
        if (power & (power - 1) == 0) and power != 0 and power != 1:
            qcontrol_square_box = QControlSquareQubitiseBox(
                self._qubitise_box,
                self._n_control,
                self._control_qreg_str,
                self._qubitise_box._control_reflection,
            )
            return PowerBox(qcontrol_square_box, power // 2)
        else:
            return PowerBox(self, power)


class QControlSquareQubitiseBox(QControlRegisterBox):
    """Constructs a Controlled SquaredQubitiseBox.

    Controls the LCUBox and adds a CnZ to the prepare register
    and the ancilla qubit two times. When B_V^2=1 and C_V^2=1,
    it is possible to avoid the controlled operations on the LCU
    or the reflection operators.

    Args:
    ----
        squared_qubitise_box (QubitiseBox): The squared QubitiseBox to be controlled.
        n_control (int): The number of ancilla qubits to be used in the
            QControlQubitiseBox.
        control_qreg_str (str): The string of the control qreg. default - 'a'.
        control_reflection (bool): Whether to control the reflection (True) or
                the LCUBox (False).
        control_index (int): The binary control index to be used in the control

    """

    def __init__(
        self,
        squared_qubitise_box: QubitiseBox,
        n_control: int,
        control_qreg_str: str = "a",
        control_reflection: bool = True,
        control_index: int | None = None,
    ):
        """Construct a Controlled from an LCUBox."""
        if n_control != 1:
            raise ValueError("n_ancilla must be 1 for QControlQubitiseBox")

        circ = squared_qubitise_box.initialise_circuit()
        circ.name = f"Q{n_control}C{squared_qubitise_box.__class__.__name__}"
        control_qreg = circ.add_q_register(control_qreg_str, 1)

        qregs = extend_new_qreg_dataclass(
            "QControlSquareQubitiseQRegs",
            squared_qubitise_box.qreg,
            {"control": control_qreg},
        )

        for _ in range(2):
            if control_reflection:
                circ.add_registerbox(squared_qubitise_box.lcu_box)

                qc_reflection_box = squared_qubitise_box.reflection_box.qcontrol(
                    1, control_qreg_str
                )
                qreg_map = QRegMap(
                    [qc_reflection_box.qreg.reflection, qc_reflection_box.qreg.control],
                    [qregs.prepare, qregs.control],
                )
                circ.add_registerbox(qc_reflection_box, qreg_map)
            else:
                qc_lcu_box = squared_qubitise_box.lcu_box.qcontrol(1, control_qreg_str)
                circ.add_registerbox(qc_lcu_box)

                qreg_map = QRegMap(
                    [squared_qubitise_box.reflection_box.qreg.reflection],
                    [qregs.prepare],
                )
                circ.add_registerbox(squared_qubitise_box.reflection_box, qreg_map)

        super().__init__(squared_qubitise_box, qregs, circ, n_control, control_index)
