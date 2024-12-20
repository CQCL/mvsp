from pytket.circuit import QubitRegister, Op
from qtmlib.circuits.core import RegisterBox, RegisterCircuit
from dataclasses import dataclass
from typing import Any


@dataclass
class PrepareQRegs:
    """PrepareBox qubit registers.

    Attributes
    ----------
        prepare (QubitRegister): The prepare register (default - p)

    """

    prepare: QubitRegister


class PrepareBox(RegisterBox):
    """Abstract PrepareBox base class.

    Inherits from RegisterBox. Circuit is constructed by adding the prepare
    box to the prepare register. Circuit construction logic is done in this class.
    Child classes set the box to be used in the prepare box.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).

    Args:
    ----
        prepare_box (CircBox): The prepare box to be used.
        prepare_qreg_str (str): The prepare register string. Defaults to "p".

    """

    def __init__(self, prepare_box: Op, prepare_qreg_str: str = "p"):
        """Initialise the PrepareRegisterBox."""
        circ = RegisterCircuit(f"{self.__repr__()}")

        self._prepare_box = prepare_box

        prepare_qreg = circ.add_q_register(prepare_qreg_str, self.prepare_box.n_qubits)
        qreg = PrepareQRegs(prepare_qreg)

        circ.add_gate(self.prepare_box, circ.qubits)

        super().__init__(qreg, circ)

    @property
    def prepare_box(self) -> Op:
        """Return the prepare box."""
        return self._prepare_box

    @property
    def l1_norm(self) -> Any:
        """Return the L1 norm of the prepare box."""
        raise NotImplementedError("l1_norm cannot be defined for PrepareCustomBox")
