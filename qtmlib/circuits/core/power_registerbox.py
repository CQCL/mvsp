"""PowerRegisterBox class."""

from __future__ import annotations
from typing import TYPE_CHECKING
from copy import copy

from qtmlib.circuits.core import RegisterBox, QControlRegisterBox
from pytket.circuit import Qubit

if TYPE_CHECKING:
    from qtmlib.circuits.core import PowerBox


class PowerBox(RegisterBox):
    """PowerBox base class.

    This class generates a circuit with with the register box
    applied npower times. This is equivalent to the unitary of the
    register box raised to the npower. The registers are stored
    in the register box attribute. This is the default behaviour
    for the power method of a register box. However it can be overridden
    by a subclass, if a better method is known.

    Registers:
        The clones of the input register box.

    Args:
    ----
        register_box (RegisterBox): The register box to be powered.
        power (int): The power of the register box.

    """

    def __init__(self, register_box: RegisterBox | QControlRegisterBox, power: int):
        """Initialise the PowerBox."""
        self._register_box = register_box

        self._power = power

        circ = register_box.initialise_circuit()
        circ.name = f"{register_box.__repr__()}^{power}"

        for _ in range(power):
            circ.add_registerbox(register_box)

        super().__init__(register_box.qreg, circ)

    @property
    def register_box(self) -> RegisterBox | QControlRegisterBox:
        """Return the Register box which is controlled.

        All the register information is stored in oracle_box.
        """
        return self._register_box

    @property
    def postselect(self) -> dict[Qubit, int]:
        """Return the postselect dictionary."""
        return self.register_box.postselect

    @property
    def dagger(self) -> PowerBox:
        """Return the dagger of the PowerBox."""
        new_register_box = copy(self.register_box)
        return PowerBox(new_register_box.dagger, self._power)

    def qcontrol(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int | None = None,
    ) -> QControlRegisterBox:
        """Return the QControlRegisterBox of the PowerBox.

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
        from qtmlib.circuits.core import extend_new_qreg_dataclass

        circ = self.register_box.initialise_circuit()
        circ.name = f"Q{n_control}C{self.register_box.__repr__()}^{self._power}"
        control_qreg = circ.add_q_register(control_qreg_str, n_control)

        qreg = extend_new_qreg_dataclass(
            "QCPowerBoxQReg", self._register_box.qreg, {"control": control_qreg}
        )

        for _ in range(self._power):
            circ.add_registerbox(
                self.register_box.qcontrol(n_control, control_qreg_str)
            )

        return QControlRegisterBox(
            self.register_box, qreg, circ, n_control, control_index
        )
