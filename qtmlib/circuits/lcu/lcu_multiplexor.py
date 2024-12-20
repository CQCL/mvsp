"""LCUBox for a Pauli operator using multiplexors."""

from qtmlib.circuits.lcu import LCUBox
from qtmlib.circuits.prepare import PrepareMultiplexorBox
from qtmlib.circuits.select import SelectMultiplexorBox
from pytket.utils.operators import QubitPauliOperator


class LCUMultiplexorBox(LCUBox):
    """LCUBox for a Pauli operator using multiplexing.

    RegisterBox that contains a StatePreparationBox, a MultiplexedTensoredU2Box
    and StatePreparationBox.dagger. Upon post selection the |0..0> state of the
    qubits in the prepare register is the state of the operator. It can only be
    applied to a circuit with a prepare register and a select register previously
    initialed by initialise_lcu_registers.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - q)
        state_qreg (QubitRegister): The state register (default - q)

    Args:
    ----
        operator (QubitPauliOperator): The operator to be applied
        n_state_qubits (int): The number of qubits in the state register
        prepare_qreg_str (str): The prepare register string. Defaults to p"
        state_qreg_str (str): The state register string. Defaults to "q"

    """

    def __init__(
        self,
        operator: QubitPauliOperator,
        n_state_qubits: int,
        prepare_qreg_str: str = "p",
        state_qreg_str: str = "q",
    ):
        """Initialise the LCUMultiplexorBox."""
        select_box = SelectMultiplexorBox(
            operator,
            n_state_qubits,
            prepare_qreg_str=prepare_qreg_str,
            state_qreg_str=state_qreg_str,
        )
        prepare_box = PrepareMultiplexorBox(
            select_box.multi_op.magnitudes, prepare_qreg_str=prepare_qreg_str
        )
        super().__init__(prepare_box, select_box)
