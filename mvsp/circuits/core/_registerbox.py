"""Abstract RegisterBox base class."""

from __future__ import annotations
from pytket.circuit import QubitRegister, Qubit, UnitID
from pytket._tket.circuit import CircBox, Circuit
from mvsp.measurement.utils import unitary_postselect, statevector_postselect
from numpy.typing import NDArray
import numpy as np
from copy import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mvsp.circuits.core import RegisterCircuit
    from mvsp.circuits.core import PowerBox
    from mvsp.circuits.core import QControlRegisterBox

from dataclasses import is_dataclass


class RegisterBox:
    """RegisterBox base class.

    This class is used to represent a register box. It is a container for
    circuits and qubit registers. It's purpose is to allow users to create
    custom register boxes. Which can be used to define circuits abstractly
    without explicitly defining the circuit and using register composition.
    It is the central class of mvsp.

    It is the base class for all register boxes. All boxes should inherit from
    it. When inheriting from this class, the user must define the registers and
    sub boxes as properties.

    Args:
    ----
        qreg_names (dataclass): The qubit register names.
        reg_circuit (RegisterCircuit): The register circuit.

    Raises:
    ------
        ValueError: If the register_box qubit registers are not a subset of the
            RegisterCircuit qubit registers.
        ValueError: If the QRegMap RegisterCircuit qubit registers are not a
            subset of the register_box qubit registers.

    """

    def __init__(self, qreg: Any, reg_circuit: RegisterCircuit):
        """Initialise the RegisterBox."""
        self._reg_circuit = reg_circuit

        self._verify_qreg_dataclass(qreg)

        self._qreg = qreg
        try:
            self._postselect = self._postselect
        except (NameError, AttributeError):
            self._postselect: dict[Qubit, int] = {}

    def _verify_qreg_dataclass(self, qreg: Any) -> None:
        """Verify the qreg input.

        Must be a dataclass of QubitRegisters or list[QubitRegister].

        Args:
        ----
            qreg (Any): The qreg input

        """
        if not is_dataclass(qreg):
            raise ValueError(
                f"qreg input must be a dataclass of \
                    QubitRegisters not type {type(qreg)}."
            )

        def verify_qreg_in_circ(qubit_register: QubitRegister | Qubit):
            if isinstance(qubit_register, QubitRegister):
                if qubit_register not in self.reg_circuit.q_registers:
                    raise ValueError(
                        f"QReg dataclass attribute {qubit_register} not a \
                        QubitRegister or list[QubitRegister]in RegisterCircuit input."
                    )
            else:
                if qubit_register not in self.reg_circuit.qubits:
                    raise ValueError(
                        f"QReg dataclass attribute {qubit_register} not a \
                        Qubit in RegisterCircuit input."
                    )

        for qreg_attr in qreg.__dict__.values():
            if isinstance(qreg_attr, list):
                for qreg in qreg_attr:
                    verify_qreg_in_circ(qreg)
            else:
                verify_qreg_in_circ(qreg_attr)

    @property
    def reg_circuit(self) -> RegisterCircuit:
        """Return the circuit used in the oracle box."""
        return self._reg_circuit

    @property
    def qreg(self) -> Any:
        """Return the qubit registers dataclass used in the RegisterBox."""
        return self._qreg

    def rename_q_registers(self, new_qreg_names: dict[QubitRegister, str]):
        """Rename qubit registers of RegisterBox.

        This will raname the underlying qubits of the RegisterBox. Which
        will also rebame the registers. The qreg dataclass will then also
        get correspondingly renamed.

        Args:
        ----
            new_qreg_names (dict[QubitRegister, str]): The new qubit register
                names.

        Raises:
        ------
            ValueError: If the qubit register is not in the RegisterBox.

        """
        # Rename the qubits in the circuit
        rename_qubits: dict[Qubit | UnitID, Qubit | UnitID] = {}
        for qreg, new_name in new_qreg_names.items():
            if qreg not in self.q_registers:
                raise ValueError(f"Qubit register {qreg} not in RegisterBox.")
            for i, q_old in enumerate(qreg.to_list()):
                rename_qubits[q_old] = Qubit(new_name, i)

        self._reg_circuit.rename_units(rename_qubits)

        # Rename the qubit registers in the qreg dataclass
        qreg_old_data = self._qreg.__dict__
        new_q_registers = self._reg_circuit.q_registers

        qreg_new_data: dict[str, QubitRegister] = {}
        for qreg_action, old_qreg in qreg_old_data.items():
            for new_qreg in new_q_registers:
                if new_qreg_names[old_qreg] == new_qreg.name:
                    qreg_new_data[qreg_action] = new_qreg

        QREGDataclass: Any = type(self._qreg)
        self._qreg = QREGDataclass(**qreg_new_data)

        # Rename the postselect dictionary
        new_postselect: dict[Qubit, int] = {}
        for qubit, value in self.postselect.items():
            new_postselect[rename_qubits[qubit]] = value  # type: ignore

        self._postselect = new_postselect

    @property
    def q_registers(self) -> list[QubitRegister]:
        """Return the qubit registers used in the RegisterBox."""
        return self.reg_circuit.q_registers

    @property
    def qubits(self) -> list[Qubit]:
        """Return the list of qubits used in the RegisterBox."""
        return self.reg_circuit.qubits

    @property
    def n_qubits(self) -> int:
        """Return the number of qubits used in the RegisterBox."""
        return self.reg_circuit.n_qubits

    def initialise_circuit(self) -> RegisterCircuit:
        """Initialise a circuit with the qubit registers used in the RegisterBox."""
        from mvsp.circuits.core import RegisterCircuit

        qreg_circ = RegisterCircuit()
        for qreg in self.q_registers:
            qreg_circ.add_q_register(qreg)
        return qreg_circ

    @property
    def postselect(self) -> dict[Qubit, int]:
        """Return the postselect dictionary."""
        return self._postselect

    def __repr__(self):
        """Return a string representation of the RegisterBox."""
        return f"{self.__class__.__name__}"

    def qcontrol(
        self,
        n_control: int,
        control_qreg_str: str = "a",
        control_index: int = 1,
    ) -> QControlRegisterBox:
        """Return a QControlRegisterBox with n_ancilla ancilla qubits.

        Defaults to pytket QControlBox.

        Args:
        ----
            n_control (int): The number of ancilla qubits.
            control_qreg_str (str): The name of the ancilla qubit register.
                defaults to 'a'.
            control_index (int): The binary index of the control qubit
                defaults to 1.

        """
        from mvsp.circuits.core import PytketQControlRegisterBox

        return PytketQControlRegisterBox(
            self, n_control, control_qreg_str, control_index
        )

    def power(self, power: int) -> PowerBox | RegisterBox:
        """Return a PowerRegisterBox with power.

        Args:
        ----
            power (int): The power of the oracle box.

        """
        from mvsp.circuits.core import PowerBox

        return PowerBox(self, power)

    @property
    def dagger(self) -> RegisterBox:
        """Return the dagger of the RegisterBox."""
        new = copy(self)
        new._reg_circuit = new._reg_circuit.dagger()
        new._reg_circuit.name = f"{self._reg_circuit.name}†"
        return new

    def get_circuit(self) -> RegisterCircuit:
        """Return the circuit used in the RegisterBox."""
        return self.reg_circuit

    def get_unitary(
        self,
        post_select_dict: dict[Qubit, int] | None = None,
        pre_select_dict: dict[Qubit, int] | None = None,
    ) -> NDArray[np.complex128]:
        """Return the unitary matrix of the RegisterBox.

        If the function is called with a postselect dictionary, returns a postselected
        matrix of the RegisterBox (assumes the postselected qubits are initialised
        in the 0 state).

        Args:
        ----
            post_select_dict (dict[Qubit, int]): The post select dictionary.
            pre_select_dict (dict[Qubit, int]): The pre select dictionary.

        """
        if post_select_dict is None and pre_select_dict is not None:
            raise ValueError("Cannot preselect without post_select_dict.")
        elif post_select_dict is not None:
            return unitary_postselect(
                self.reg_circuit.qubits,
                self.reg_circuit.get_unitary(),
                post_select_dict.copy(),
                pre_select_dict.copy() if pre_select_dict is not None else None,
            )
        else:
            return self.reg_circuit.get_unitary()

    def get_statevector(
        self, post_select_dict: dict[Qubit, int] | None = None
    ) -> NDArray[np.complex128]:
        """Return the statevector of the RegisterBox.

        Assumes the state starts out in the 0 state. If the function is called with a
        postselect dictionary, returns a postselected statevector of the RegisterBox.

        Args:
        ----
            post_select_dict (dict[Qubit, int]): The post select dictionary.

        """
        if post_select_dict is None:
            return self.reg_circuit.get_statevector()
        else:
            postselected_statevector = statevector_postselect(
                self.reg_circuit.qubits,
                self.reg_circuit.get_statevector(),
                post_select_dict,
            )
            if np.linalg.norm(postselected_statevector) != 0:
                return postselected_statevector / np.linalg.norm(
                    postselected_statevector
                )
            else:
                return postselected_statevector

    def to_circbox(self) -> CircBox:
        """Return the CircBox of the RegisterBox.

        Bewrare this wil flatten the registers and should only be used
        for compatibility with other pytket libraries. add_registerbox
        should be used where possible.
        """
        circ = self._reg_circuit.copy()
        circ.flatten_registers()
        return CircBox(circ)

    @classmethod
    def from_CircBox(
        cls,
        circbox: Any,
        assign_registers: dict[str, list[int]] | None = None,
        circuit_name: str | None = None,
    ) -> RegisterBox:
        """Initialise a RegisterBox from a CircBox Input.

        This class method should be used to convert pytket CircBoxes into qlibs
        RegisterBoxes.

        Adds the extra qreg dataclass functionality to the CircBox class. This is
        necessary because CircBoxes only have a single qreg. If no assign_registers
        is supplied, the default qreg name is 'default'.

        If assign_registers is given, the keys are the name of the qubit registers and
        the values are the qubit indices. The qreg names must be unique and the integer
        qubit indices for each qubit register must be the complete set of qubit indices
        in the CircBox. The qubits will be renamed to the qreg names and indices.

        Args:
        ----
            circbox (Any): The CircBox input.
            assign_registers (dict[str, list[int]] | None, optional): The qubit register
                assignment. Defaults to None.
            circuit_name (str | None, optional): The name of the circuit.

        Raises:
        ------
            ValueError: If the assign_registers are incomplete or overlapping.

        Returns:
        -------
            RegisterBox: The RegisterBox.

        """
        from mvsp.circuits.core import RegisterCircuit
        from mvsp.circuits.core import make_qreg_dataclass

        circ = RegisterCircuit(circbox.n_qubits)
        circ.append(circbox.get_circuit().copy())

        if circuit_name is not None:
            circ.name = circuit_name

        if assign_registers is not None:
            if {*range(circ.n_qubits)} != {
                qubit for qubits in assign_registers.values() for qubit in qubits
            }:
                raise ValueError(
                    "Incomplete or overlapping qubits \
                                 in register assignment."
                )

            rename_dict: dict[UnitID, UnitID] = {}
            for qreg_name, qubit_list in assign_registers.items():
                for i, qubit in enumerate(qubit_list):
                    rename_dict[Qubit("q", qubit)] = Qubit(qreg_name, i)
            circ.rename_units(rename_dict)

            dataclass_input: dict[str, QubitRegister | list[QubitRegister]] = {
                qreg_name: circ.get_q_register(qreg_name)
                for qreg_name in assign_registers.keys()
            }
            qregs = make_qreg_dataclass(dataclass_input)

        else:
            dataclass_input: dict[str, QubitRegister | list[QubitRegister]] = {
                "default": circ.q_registers[0]
            }
            qregs = make_qreg_dataclass(dataclass_input)

        return cls(qregs, circ)

    @classmethod
    def from_Circuit(
        cls,
        circ: Circuit,
        qreg_attrs: dict[str, QubitRegister] | None = None,
    ) -> RegisterBox:
        from mvsp.circuits.core import make_qreg_dataclass
        from mvsp.circuits.core import RegisterCircuit

        """Initialise a RegisterBox from a Circuit Input.

        This class method should be used to convert pytket RegisterCircuits into qlibs
        RegisterBoxes. An optional qreg_attrs_names dictionary can be supplied to
        rename the qubit registers. The keys are the qreg attribute (symbolic) names
        and the values are the QubitRegisters.

        Args:
            circ (Circuit): The Circuit input.
            qreg_attrs (dict[str, str]| None, optional): The qubit register
                assignment. Defaults to None. The keys are the qreg attribute (symbolic)
                names and the values are the QubitRegisters.

        Returns:
            RegisterBox: The RegisterBox.
        """
        if qreg_attrs is None:
            qreg_attrs_names: dict[str, QubitRegister] = {
                qreg.name: qreg for qreg in circ.q_registers
            }
        else:
            qreg_attrs_names: dict[str, QubitRegister] = qreg_attrs
        qregs = make_qreg_dataclass(qreg_attrs_names)

        reg_circ = RegisterCircuit()
        if circ.name is not None:
            reg_circ.name = circ.name
        for qreg in qreg_attrs_names.values():
            if not isinstance(qreg, list):
                reg_circ.add_q_register(qreg)
        reg_circ.append(circ.copy())

        return cls(qregs, reg_circ)
