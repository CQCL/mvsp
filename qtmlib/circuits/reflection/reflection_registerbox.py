"""Contains the ReflectionBox class."""

from dataclasses import dataclass
from pytket.circuit import OpType, QubitRegister
from qtmlib.circuits.core import (
    RegisterBox,
    RegisterCircuit,
    QControlRegisterBox,
    PytketQControlRegisterBox,
    extend_new_qreg_dataclass,
)


@dataclass
class ReflectionQRegs:
    """ReflectionBox qubit registers.

    Attributes
    ----------
        reflection (QubitRegister): The reflection register (default - r)

    """

    reflection: QubitRegister


class ReflectionBox(RegisterBox):
    """Constructs a ReflectionBox.

    Implements the reflection operator +-R=kron((2|000...><000...| - 1), Identity).

    Registers:
        reflection_qreg (QubitRegister): The reflection register (default - r)

    Args:
    ----
        n_qubits (int): The number of qubits.
        positive (bool): True if +R, -R otherwise. Defaults to True
        reflection_qreg_str (str): The prepare register string. Defaults to "r".

    """

    def __init__(
        self, n_qubits: int, positive: bool = True, reflection_qreg_str: str = "r"
    ):
        """Initialise the ReflectionBox."""
        self._positive = positive

        circ = RegisterCircuit(f"{self.__repr__()}")

        reflection_qreg = circ.add_q_register(reflection_qreg_str, n_qubits)
        qreg = ReflectionQRegs(reflection_qreg)

        for p in reflection_qreg.to_list():
            circ.X(p)

        if n_qubits == 1:
            circ.Z(reflection_qreg[0])
        else:
            circ.add_gate(
                OpType.CnZ, [*reflection_qreg.to_list()[1:], reflection_qreg[0]]
            )
        for p in reflection_qreg.to_list():
            circ.X(p)

        if self._positive:
            circ.add_phase(1.0)
        super().__init__(qreg, circ)

    @property
    def positive(self) -> bool:
        """Return the sign of the reflection."""
        return self._positive

    def qcontrol(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ) -> QControlRegisterBox:
        """Return a controlled QControlReflectionBox.

        If the number of ancilla qubits is 1 then the QControlReflectionBox is returned.
        Else the default QControlRegisterBox is returned.

        Args:
        ----
            n_control (int): The number of ancilla qubits.
            control_qreg_str (str): The string of the control qreg.
                default - 'a'.
            control_index (int): The binary control index to be used in the control

        Returns:
        -------
            QControlRegisterBox: The controlled QControlReflectionBox.

        """
        if n_control != 1:
            return PytketQControlRegisterBox(
                self, n_control, control_qreg_str, control_index
            )
        else:
            return QControlReflectionBox(self, control_qreg_str, control_index)


class QControlReflectionBox(QControlRegisterBox):
    """Constructs a Controlled QControlReflectionBox.

    Only available for one control qubit. The controlled operation on a ReflectionBox
    can be simplified via applying a controlled Z operation on the ancilla as target
    and the rest of qubits as controls.

    Registers:
        reflection_qreg (QubitRegister): The reflection register (default - r).
        control_qreg (QubitRegister): The control register (default - a).

    Args:
    ----
        reflection_box (ReflectionBox): The ReflectionBox.
        control_qreg_str (str): The string of the control qreg.
            default - 'a'.
        control_index (int): The binary control index to be used in the control

    """

    def __init__(
        self,
        reflection_box: ReflectionBox,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ):
        """Initialise the QControlReflectionBox."""
        circ = reflection_box.initialise_circuit()
        circ.name = f"Q{1}C{self.__class__.__name__}"
        control_qreg = circ.add_q_register(control_qreg_str, 1)

        qregs = extend_new_qreg_dataclass(
            "QControlReflectionQRegs", reflection_box.qreg, {"control": control_qreg}
        )

        for p in reflection_box.qreg.reflection.to_list():
            circ.X(p)

        circ.add_gate(
            OpType.CnZ, [*reflection_box.qreg.reflection.to_list(), control_qreg[0]]
        )

        for p in reflection_box.qreg.reflection.to_list():
            circ.X(p)
        # The following Z gate ensures that we implement the reflection at 0 and not
        # minus the reflectation at zero.
        if reflection_box.positive:
            circ.add_gate(OpType.Z, [control_qreg[0]])

        n_control = 1
        super().__init__(reflection_box, qregs, circ, n_control, control_index)
