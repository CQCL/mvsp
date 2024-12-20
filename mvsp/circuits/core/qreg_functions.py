"""Functions for QubitRegister dataclasses."""

from dataclasses import make_dataclass
from pytket.circuit import QubitRegister
from typing import Any
from collections.abc import Mapping


def make_qreg_dataclass(
    qreg_dict: Mapping[str, QubitRegister | list[QubitRegister]],
    dataclass_name: str = "QRegs",
) -> Any:
    """Dynamically build a QubitRegister dataclass.

    From a dictionary of qubit registers, this function will dynamically
    build a QubitRegister dataclass from the keys of the dictionary. Where
    the attribute names of the dataclass are the keys of the dictionary and
    the attribute values are the QubitRegisters or list of QubitRegisters.

    Args:
    ----
        qreg_dict (dict[str, QubitRegister|list[QubitRegister]]): The dictionary of
        QubitRegisters or list of QubitRegisters to be used in the dataclass.
        dataclass_name (str): The name of the to be created dataclass

    Returns:
    -------
        Any: The QubitRegister dataclass.

    """
    data_class_input: list[tuple[str, type]] = []
    for qreg_name, qreg_instance in qreg_dict.items():
        if isinstance(qreg_instance, list):
            data_class_input.append((qreg_name, list[QubitRegister]))
        else:
            data_class_input.append((qreg_name, QubitRegister))

    QRegs = make_dataclass(dataclass_name, data_class_input)
    qregs = QRegs(*list(qreg_dict.values()))
    return qregs


def extend_new_qreg_dataclass(
    data_class_name: str,
    qreg_old: Any,
    extend_attrs: dict[str, QubitRegister | list[QubitRegister]],
) -> Any:
    """Extend and create new QubitRegister dataclass.

    Input qreg data class which you want to extend. Checks if that
    attribute name already exists and throws an error if it does. Then
    extends the dataclass with the new attributes and creates a new
    dataclass.

    Args:
    ----
        data_class_name (str): The name of the to be created dataclass
        qreg_old (Any): The old QubitRegister dataclass.
        extend_attrs (dict[str,QubitRegister|list[QubitRegister]]): The new attributes
        to be added to the dataclass.

    Returns:
    -------
        Any: The new QubitRegister dataclass.

    Raises:
    ------
        ValueError: If the attribute name already exists in the dataclass.

    """
    for key in extend_attrs:
        if key in qreg_old.__dict__.keys():
            raise ValueError(
                f"QubitRegister attribute {key} already exists in {qreg_old}."
            )
    return make_qreg_dataclass({**qreg_old.__dict__, **extend_attrs}, data_class_name)
