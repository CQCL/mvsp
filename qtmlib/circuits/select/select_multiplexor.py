"""Module containing the SelectMultiplexorBox class."""

from qtmlib.circuits.select import SelectBox
from pytket.circuit import MultiplexedTensoredU2Box
from pytket.utils.operators import QubitPauliOperator
from qtmlib.circuits.core import (
    QControlRegisterBox,
    extend_new_qreg_dataclass,
)


class SelectMultiplexorBox(SelectBox):
    """SelectCustomBox Concrete class."""

    """This class inherits from SelectBox.
    It sets the select box to be any pytket circbox.
    To then be passed to the parent class SelectBox.
    Where the circuit construction logic is done.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).

    Args:
        operator (QubitPauliOperator): The operator to be used.
        n_state_qubits (int): The number of state qubits.
        prepare_qreg_str (str): The prepare register string. Defaults to "p".
        state_qreg_str (str): The state register string. Defaults to "q".
    """

    def __init__(
        self,
        operator: QubitPauliOperator,
        n_state_qubits: int,
        prepare_qreg_str: str = "p",
        state_qreg_str: str = "q",
    ) -> None:
        """Initialise the SelectCustomBox."""
        from qtmlib.circuits.lcu.process_operator import MulitplexedOperator
        from qtmlib.circuits.utils import int_to_bits

        self._operator = operator
        self._multi_op = MulitplexedOperator(operator, n_state_qubits)
        op_map = [
            (int_to_bits(i, self._multi_op.n_prep_qubits), term.op_list)
            for i, term in enumerate(self._multi_op.terms)
        ]

        select_box = MultiplexedTensoredU2Box(op_map)

        super().__init__(
            select_box,
            n_state_qubits,
            prepare_qreg_str=prepare_qreg_str,
            state_qreg_str=state_qreg_str,
            is_hermitian=self._multi_op.is_hermitian,
        )

    @property
    def n_prep_qubits(self) -> int:
        """Return the number of prepare qubits."""
        return self._multi_op.n_prep_qubits

    @property
    def operator(self) -> QubitPauliOperator:
        """Return the operator."""
        return self._operator

    @property
    def multi_op(self):
        """Return the multi_op."""
        return self._multi_op

    def qcontrol(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ) -> QControlRegisterBox:
        """Return a controlled QControlSelectMultiplexorBox.

        If the number of ancilla qubits is less than 5 then the
        QControlSelectMultiplexorBox is returned. Else the
        PytketQControlRegisterBox is returned.

        Args:
        ----
            n_control (int): The number of ancilla qubits.
            control_qreg_str (str): The string of the control qreg.
                default - 'a'.
            control_index (int): The binary control index to be used in the control

        Returns:
        -------
            QControlRegisterBox: The controlled QControlSelectMultiplexorBox.

        """
        return QControlSelectMultiplexorBox(
            self, n_control, control_qreg_str, control_index
        )


class QControlSelectMultiplexorBox(QControlRegisterBox):
    """Constructs a Controlled QControlSelectMultiplexorBox.

    Via adding an extra qubit to the opmap in the SelectMultiplexorBox.
    There increasing the size of the multiplexor a qubit. Where the extra
    control terms are given the Identity operator.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).
        state_qreg (QubitRegister): The state register (default - q).
        control_qreg (QubitRegister): The control register (default - a).

    Args:
    ----
        select_box (SelectMultiplexorBox): The SelectMultiplexorBox.
        n_control (int): The number of ancilla qubits.
        control_qreg_str (str): The string of the control qreg.
            default - 'a'.
        control_index (int): The binary control index to be used in the control

    """

    def __init__(
        self,
        select_box: SelectMultiplexorBox,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ):
        """Initialise the QControlSelectMultiplexorBox."""
        from qtmlib.circuits.utils import int_to_bits

        circ = select_box.initialise_circuit()
        circ.name = f"Q{n_control}C{select_box.__class__.__name__}"
        control_qreg = circ.add_q_register(control_qreg_str, n_control)

        qregs = extend_new_qreg_dataclass(
            "QControlSelectQRegs", select_box.qreg, {"control": control_qreg}
        )

        op_map = [
            (int_to_bits(i, select_box.n_prep_qubits + n_control), term.op_list)
            for i, term in enumerate(select_box.multi_op.terms)
        ]

        qc_select_box = MultiplexedTensoredU2Box(op_map)

        for cq in control_qreg:
            circ.X(cq)
        circ.add_gate(qc_select_box, [*circ.qubits])
        for cq in control_qreg:
            circ.X(cq)

        super().__init__(select_box, qregs, circ, n_control, control_index)
