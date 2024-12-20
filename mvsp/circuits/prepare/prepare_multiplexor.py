"""PrepareMultiplexor class."""

import numpy
from pytket.circuit import StatePreparationBox
from math import ceil, log2
from mvsp.circuits.prepare import PrepareBox


class PrepareMultiplexorBox(PrepareBox):
    """PrepareCustomBox Concrete class."""

    """This class inherits from PrepareRegisterBox. It sets the prepare box to be any
    pytket circbox. To then be passed to the parent class PrepareRegisterBox. Where
    the circuit construction logic is done.

    Registers:
        prepare_qreg (QubitRegister): The prepare register (default - p).

    Args:
        unnorm_state (list): The unnormalised state to be prepared.
        prepare_qreg_str (str): The prepare register string. Defaults to "p".
    """

    def __init__(self, unnorm_state: list[float], prepare_qreg_str: str = "p") -> None:
        """Initialise the PrepareCustomBox."""
        self._l1_norm: float = sum(unnorm_state)

        self._lcu_state = numpy.sqrt(numpy.array(unnorm_state) / self._l1_norm).tolist()

        full_state_nqubits = int(ceil(log2(len(unnorm_state))))

        self._lcu_state.extend(
            (numpy.zeros((2**full_state_nqubits) - len(unnorm_state))).tolist()
        )

        prepare_box = StatePreparationBox(self._lcu_state)
        super().__init__(prepare_box, prepare_qreg_str=prepare_qreg_str)

    @property
    def l1_norm(self) -> float:
        """Return the L1 norm of the prepare box."""
        return self._l1_norm
