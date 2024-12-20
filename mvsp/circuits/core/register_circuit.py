"""Oracle Circuit for Abstract Circuit Construction."""

from mvsp.circuits.core import RegisterBox
from pytket.circuit import QubitRegister, Qubit
from pytket._tket.circuit import Circuit, CircBox
from dataclasses import dataclass
from copy import deepcopy
from typing import Self
from collections.abc import Sequence
from collections import Counter


MAP_INPUT_TYPES = QubitRegister | Qubit | list[Qubit]


@dataclass
class RegisterMapElement:
    """Qubit Register Map Element.

    This class is used to map the qubit registers of an register_box to the
    qubit registers of a RegisterCircuit. The register_box qubit registers are
    the input qubit registers of the register_box. It can also map individual
    qubits and lists of qubits to RegisterCircuit.

    Args:
    ----
        box (QubitRegister | Qubit | list[Qubit]): The qubit register or qubits
            to be mapped.
        circ (QubitRegister | Qubit | list[Qubit]): The qubit register or qubits
            to be mapped to.

    Raises:
    ------
        ValueError: If the box registers/qubits and registers/qubits are not the
            same type.
        ValueError: If the box registers/qubits and registers/qubits are not the
            same size.

    """

    box: QubitRegister | Qubit | list[Qubit]
    circ: QubitRegister | Qubit | list[Qubit]

    def __post_init__(self):
        """Initialise the RegisterMapElement."""
        if isinstance(self.box, (QubitRegister | list)) and isinstance(
            self.circ, (QubitRegister | list)
        ):
            if len(self.box) != len(self.circ):
                raise ValueError(
                    f"box qreg {self.box} and circuit qreg {self.circ} are"
                    "not the same size"
                )


class QRegMap:
    """Qubit Register Maps for RegisterBoxes to RegisterCircuits.

    This class is used to map the list[QubitRegister | Qubit | list[Qubit]]) of an
    register_box to the corresponding list[QubitRegister | Qubit | list[Qubit]])
    of a RegisterCircuit. Each element of the box_qreg maps to the same index
    element in the other list. They must be the same size and type.

    Args:
    ----
        box_qregs (list[QubitRegister | Qubit | list[Qubit]]): List of
            QubitRegister objects for the register_box.
        reg_circ_qregs (list[QubitRegister | Qubit | list[Qubit]]): List of
            QubitRegister objects for the RegisterCircuit.

    Raises:
    ------
        ValueError: If the number of qubits in the register_box qubit registers and
            the RegisterCircuit qubit registers are not the same.
        ValueError: If there are duplicates in the register_box qubit registers.

    """

    def __init__(
        self,
        box_qregs: Sequence[QubitRegister | Qubit | list[Qubit]],
        reg_circ_qregs: Sequence[QubitRegister | Qubit | list[Qubit]],
    ) -> None:
        """Initialise the QRegMap."""
        self.items = [
            RegisterMapElement(box, qregcirc)
            for box, qregcirc in zip(box_qregs, reg_circ_qregs, strict=True)
        ]
        self.box_qregs = box_qregs
        self.circ_qregs = reg_circ_qregs

        self._box_qubits = self.qubit_list(box_qregs)
        self._circ_qubits = self.qubit_list(reg_circ_qregs)

    @property
    def qubit_map(self) -> dict[Qubit, Qubit]:
        """Return the qubit map."""
        return dict(zip(self.box_qubits, self.circ_qubits, strict=True))

    @property
    def box_qubits(self) -> list[Qubit]:
        """Return the qubits in the register_box."""
        return self._box_qubits

    @property
    def circ_qubits(self) -> list[Qubit]:
        """Return the qubits in the RegisterCircuit."""
        return self._circ_qubits

    def qubit_list(self, map_qreg: Sequence[MAP_INPUT_TYPES]) -> list[Qubit]:
        """Convert the map_qreg to a set of qubits."""
        qubits: list[Qubit] = []
        for element in map_qreg:
            if isinstance(element, QubitRegister):
                qubits.extend(element.to_list())
            elif isinstance(element, list):
                qubits.extend(element)
            else:
                qubits.append(element)

        counts = Counter(qubits)
        for item, count in counts.items():
            if count > 1:
                raise ValueError(f"Qubit {item} appears more than once in the input")

        return qubits

    @classmethod
    def from_dict(cls, qreg_map_dict: dict[MAP_INPUT_TYPES, MAP_INPUT_TYPES]) -> Self:
        """Create a QRegMap from a dictionary.

        Args:
        ----
            qreg_map_dict (dict): Dictionary of QubitRegister objects for the
                register_box (.keys()) and RegisterCircuit (.values()).§

        """
        box_qregs = list(qreg_map_dict.keys())
        qregcirc_qregs = list(qreg_map_dict.values())
        return cls(box_qregs, qregcirc_qregs)

    @classmethod
    def from_QRegMap_list(cls, qreg_map_list: list[Self]) -> Self:
        """Create a QRegMap from a list of QRegMaps.

        Args:
        ----
            qreg_map_list (list[QRegMap]): List of QRegMap objects.

        """
        box_qubits = [qreg_map.box_qubits for qreg_map in qreg_map_list]
        circ_qubits = [qreg_map.circ_qubits for qreg_map in qreg_map_list]
        return cls(box_qubits, circ_qubits)

    def __repr__(self):
        """Return string representation of the QRegMap."""
        mapping_str = "\n"
        for item in self.items:
            if isinstance(item.box, QubitRegister) and isinstance(
                item.circ, QubitRegister
            ):
                mapping_str += f"QREG: {item.box.name} [{len(item.box)}] -> \
                    {item.circ.name} [{len(item.circ)}]\n"

            elif isinstance(item.box, Qubit) and isinstance(item.circ, Qubit):
                mapping_str += f"QUBIT: {item.box.reg_name} ({item.box.index}) -> \
                      {item.circ.reg_name} ({item.circ.index})\n"

            elif isinstance(item.box, list) and isinstance(item.circ, list):
                box_qubits_str = "".join(
                    [f"{qubit.reg_name} ({qubit.index})," for qubit in item.box]
                )
                circ_qubits_str = "".join(
                    [f"{qubit.reg_name} ({qubit.index})," for qubit in item.circ]
                )
                mapping_str += (
                    f"QUBITS: {box_qubits_str[:-1]} -> {circ_qubits_str[:-1]}\n"
                )

        return f"QRegMap (box -> circ):\n{mapping_str}"


class RegisterCircuit(Circuit):
    """OracleCircuit for Abstract Circuit Construction.

    This class inherits from Circuit. It is used to construct the circuit
    but add the abililty to add an register_box to the circuit using a QRegMap.
    Which maps the registers of the register_box to the registers of the
    RegisterCircuit. Register Circuits have all the functionality of pytket.Circuit
    and can be used in the same way. With individual gates and qubits. But the
    main purpose is to add an register_box to the circuit just use register maps.
    """

    def add_registerbox(
        self, register_box: RegisterBox, qreg_map: QRegMap | None = None
    ) -> Self:
        """Add an register_box to the RegisterCircuit.

        Adds an register_box to the RegisterCircuit using a QRegMap to map the
        register_box qubit registers to the RegisterCircuit qubit registers.
        If no QRegMap is provided, the register_box qubit registers will be
        mapped to the RegisterCircuit qubit registers automatically if they
        match in name and size.

        Args:
        ----
            register_box (register_box): The register_box to be added to the circuit.
            qreg_map (QRegMap, optional): The QRegMap to map the register_box
                qubit registers to the RegisterCircuit qubit registers.
                Defaults to None.Box registers matching circuit names of the
                 same size will be mapped automatically, if None is passed.

        Raises:
        ------
            ValueError: If the register_box qubit registers are not a subset of the
                RegisterCircuit qubit registers (When no QregMap is provided).
            ValueError: If the QRegMap RegisterCircuit qubit registers are not a
            subset of the register_box qubit registers.

        Returns:
        -------
            RegisterCircuit: The RegisterCircuit with the register_box added.

        """
        if qreg_map is None:
            # if not set(register_box.q_registers).issubset(set(self.q_registers)):
            #     raise ValueError(
            #         "register_box QubitRegisters are not a subset of "
            #         "circuit QubitRegisters of the same size"
            #     )
            if not set(register_box.qubits).issubset(set(self.qubits)):
                raise ValueError(
                    "register_box qubits are not a subset of circuit qubits"
                )
            qubits = register_box.qubits

        else:
            if not set(qreg_map.box_qubits).issubset(set(register_box.qubits)):
                raise ValueError("qreg map box qubits are not a subset of box qubits")

            if not set(qreg_map.circ_qubits).issubset(set(self.qubits)):
                raise ValueError("qreg map circ qubits are not a subset of circ qubits")

            # Orders the map in the same order as the box qregs
            # Then form the qubit input list

            qubits = [qreg_map.qubit_map[q_regbox] for q_regbox in register_box.qubits]

        circ = register_box.get_circuit().copy()
        circ.flatten_registers()

        self.add_gate(CircBox(circ), qubits)

        return self

    def copy(self) -> Self:
        """Return a copy of the RegisterCircuit."""
        return deepcopy(self)
