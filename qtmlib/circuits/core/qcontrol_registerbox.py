"""QControlOracleBox class."""

from typing import cast

from pytket.circuit import QControlBox, Qubit, UnitID

from qtmlib.circuits.core import RegisterBox, RegisterCircuit
from typing import Any


class QControlRegisterBox(RegisterBox):
    """Abstract QControlRegisterBox base class.

    This abstract class generates a circuit with a QControl of a register circuit,
    QControlling the register box. But also keeps the registers and qubit
    names of the register box. The registers are stored in the register box attribute.

    When the control_index is None, the control qubits are set to False. Otherwise
    the control qubits are set to the binary representation of the control_index by
    adding X gates around the control.

    This functionality is here rather than just using the pytket QControlBox
    because the are other controlled methods which need to indexed that do not
    use QControlBox.

    Registers:
        ancilla_qreg (QubitRegister): The ancilla register (default - a).
        register_box qreg clones (QubitRegisters): The clones of the input register
        box.

    Args:
    ----
        register_box (RegisterBox): The oracle box to be controlled.
        n_ancilla (int): The number of ancilla qubits to be used in the
        qreg (dataclass): The qubit registers of the register box.
        circ (RegisterCircuit): The circuit of the register box.
        n_control (int): The number of control qubits.
        control_index (int): The binary control index to be used in the control
            default - 1.

    """

    def __init__(
        self,
        register_box: RegisterBox,
        qreg: Any,
        circ: RegisterCircuit,
        n_control: int,
        control_index: int | None = None,
    ):
        """Initialise the QControlOracleBox."""
        self._register_box = register_box

        if not hasattr(qreg, "control"):
            raise AttributeError("qreg data does not have the 'control' attribute")

        if control_index is None:
            control_index = (2**n_control) - 1

        control_circ = RegisterCircuit(
            f"QC{n_control}({control_index}){register_box.get_circuit().name}"
        )
        for q_reg in circ.q_registers:
            control_circ.add_q_register(q_reg)

        from qtmlib.circuits.utils import int_to_bits

        control_bits = int_to_bits(cast(int, control_index), n_control)

        for i, control_bit in enumerate(control_bits):
            if control_bit is False:
                control_circ.X(qreg.control[i])

        control_circ.append(circ)

        for i, control_bit in enumerate(control_bits):
            if control_bit is False:
                control_circ.X(qreg.control[i])

        super().__init__(qreg, control_circ)

    @property
    def register_box(self) -> RegisterBox:
        """Return the Register box which is controlled.

        All the register information is stored in register_box.
        """
        return self._register_box

    @property
    def postselect(self) -> dict[Qubit, int]:
        """Return the postselect dictionary."""
        return self.register_box.postselect


class PytketQControlRegisterBox(QControlRegisterBox):
    """PytketQControlRegisterBox class.

    This class generates a circuit with a pytket QControlBox containing the register
    box. The circuit uses pytket QControlBox class to generate the controlled box.
    But also keeps the registers and qubit names of the register box. It is the
    default controlled register box used in qtmlib.

    Registers:
        ancilla_qreg (QubitRegister): The ancilla register (default - a).
        register_box qreg clones (QubitRegister): The clones of the input register
        box.

    Args:
    ----
        register_box (RegisterBox): The oracle box to be controlled.
        n_ancilla (int): The number of ancilla qubits to be used in th
        control_qreg_str (str): The string of the control qreg.
            default - 'a'.
        control_index (int): The binary control index to be used in the control

    """

    def __init__(
        self,
        register_box: RegisterBox,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ):
        """Initialise the QControlOracleBox."""
        from qtmlib.circuits.core import extend_new_qreg_dataclass

        circ = register_box.initialise_circuit()
        circ.name = f"C{n_control}{register_box.__repr__()}"
        qubits = circ.qubits

        control_qreg = circ.add_q_register(control_qreg_str, n_control)
        qubits = control_qreg.to_list() + qubits

        qreg = extend_new_qreg_dataclass(
            "QCntrlQRegs", register_box.qreg, {"control": control_qreg}
        )

        qc_box = QControlBox(register_box.to_circbox(), n_control)

        circ.add_gate(qc_box, cast(list[UnitID], qubits))

        super().__init__(register_box, qreg, circ, n_control, control_index)
