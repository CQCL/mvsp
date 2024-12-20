"""Block encoding boxes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pytket.circuit import QubitRegister

from qtmlib.circuits.core import (
    PowerBox,
    QControlRegisterBox,
    QRegMap,
    RegisterBox,
    RegisterCircuit,
)
from qtmlib.circuits.lcu import LCUMultiplexorBox
from qtmlib.circuits.qubitisation import QubitiseBox
from qtmlib.circuits.utils.block_encoding_utils import (
    generate_diagonal_block_encoding,
)
from qtmlib.circuits.utils.lcu_state_preparation_utils import Rz_jkn

if TYPE_CHECKING:
    from qtmlib.circuits.lcu_state_preparation.lcu_state_preparation import (
        LCUStatePreparationQReg,
    )


class BaseLCUStatePreparationBlockEncoding(ABC):
    """Base class for the LCUStatePreparation block encodings.

    It ensures that all block encoding classes for the LCU state preparation circuit
    have the same interface.

    All subclasses must implement the generate_basis_element and the
    add_controlled_block_encoding_sequence methods. These are used for the construction
    of the LCU state preparation circuit.

    """

    def __init__(self, n_block_qubits: int):
        """Initialise abstract block encoding.

        Args:
        ----
            n_block_qubits (int): Number of qubits.

        """
        self._n_block_qubits = n_block_qubits
        self._n_qubits = (
            n_block_qubits  # This is redefined in the concrete child classes.
        )

    @property
    def n_block_qubits(self) -> int:
        """Return the number of block qubits."""
        return self._n_block_qubits

    @property
    def n_qubits(self) -> int:
        """Return the total number of qubits."""
        return self._n_qubits

    @abstractmethod
    def generate_basis_element(self, basis_index: int) -> PowerBox | RegisterBox:
        """Abstract method for the generate_basis_element method."""
        pass

    @abstractmethod
    def _generate_shift_qreg_map(
        self,
        control_idx: int,
        qreg: LCUStatePreparationQReg,
        shift_box: RegisterBox,
    ) -> QRegMap:
        pass

    @abstractmethod
    def _generate_control_qreg_map(
        self,
        control_idx: int,
        k: int,
        qreg: LCUStatePreparationQReg,
        be_box: QControlRegisterBox,
    ) -> QRegMap:
        pass

    def add_controlled_block_encoding_sequence(
        self,
        n_qubits_coeffs: int,
        basis_indices: list[int],
        circ: RegisterCircuit,
        qreg: LCUStatePreparationQReg,
        control_idx: int,
    ):
        """Add sequence of controlled block encodings to LCUStatePreparationBox.

        Args:
        ----
            n_qubits_coeffs (int): Number of qubits of the coefficient register.
            basis_indices (list[int]): Indices of the basis elements with nonzero
                coefficients.
            circ (RegisterCircuit): Circuit to which the controlled block encodings are
                added.
            qreg (LCUStatePreparationQReg): QReg containing the
                registers of circ.
            control_idx (int): Control index k. The 2**k-th basis element block encoding
                is added and controlled on the k-th qubit in the coefficient register.

        """
        control_map = {
            (n_qubits_coeffs - 1 - k): self.generate_basis_element(2**k)
            for k in range(n_qubits_coeffs)
        }
        if basis_indices[0] != 0:
            shift_box = self.generate_basis_element(basis_indices[0])
            qreg_map = self._generate_shift_qreg_map(control_idx, qreg, shift_box)
            circ.add_registerbox(shift_box, qreg_map)

        for k in control_map.keys():
            be_box = control_map[k].qcontrol(1)
            qreg_map = self._generate_control_qreg_map(control_idx, k, qreg, be_box)
            circ.add_registerbox(be_box, qreg_map)


@dataclass
class ChebychevBlockEncodingQRegs:
    """Chebychev block encoding qubit registers.

    Attributes
    ----------
        block (QubitRegister): The register of the encoded block.
        ancilla (QubitRegister): The ancilla register.

    """

    block: QubitRegister
    ancilla: QubitRegister


class ChebychevBlockEncoding(BaseLCUStatePreparationBlockEncoding):
    """ChebychevBlockEncoding class.

    This class represents a general block-encoding of the Chebychev basis and contains
    a method called ```generate_basis_element``` that, given an index k, generates a
    register box containing the block encoding of the k-th Chebychev basis function.
    It initially generates an LCU box with the Chebyshev Hamiltonian and for each basis
    element powers a QubitiseBox containing such LCU.

    Args:
    ----
        n_block_qubits (int): Number of block qubits

    """

    def __init__(self, n_block_qubits: int):
        """Initialise ChebychevBlockEncoding."""
        super().__init__(n_block_qubits=n_block_qubits)

        # Create LCU Register Box for the Chebyshev encoding
        chebyshev_hamiltonian = generate_diagonal_block_encoding(
            n_qubits=n_block_qubits
        )
        lcu_box = LCUMultiplexorBox(
            chebyshev_hamiltonian, n_state_qubits=n_block_qubits
        )
        self._n_ancilla_qubits = len(lcu_box.qreg.prepare)
        self._control_state = [False for _ in range(self._n_ancilla_qubits)]
        self._lcu_box = lcu_box
        self._n_qubits = self.n_block_qubits + self._n_ancilla_qubits

    @property
    def n_ancilla_qubits(self) -> int:
        """Return the number of ancilla qubits."""
        return self._n_ancilla_qubits

    @property
    def control_state(self) -> list[bool]:
        """Return the control bitstring of the block encoding."""
        return self._control_state

    @property
    def n_qubits(self) -> int:
        """Return the total number of qubits."""
        return self._n_qubits

    def generate_basis_element(self, basis_index: int) -> PowerBox:
        """Generate a RegisterBox containing the block-encoding.

        Given the chebychev index k, this method generates the block-encoding of the
        k-th Chebychev basis function using instances of QubitiseBox.

        Registers:
            ancilla_qreg (QubitRegister): Ancilla register block encoding.
            block_qreg (QubitRegister): Block register of block encoding.

        Args:
        ----
            basis_index (int): Index of the basis element.

        Returns:
        -------
            RegisterBox: Chebychev-index-th Chebychev basis block encoding.

        """
        assert basis_index >= 0, "basis_index must be larger than 0."

        power_box = QubitiseBox(self._lcu_box).power(power=basis_index)

        new_qreg_names = {
            power_box.qreg.state: "block",
            power_box.qreg.prepare: "ancilla",
        }
        power_box.rename_q_registers(new_qreg_names)
        return power_box

    def _generate_shift_qreg_map(
        self,
        control_idx: int,
        qreg: LCUStatePreparationQReg,
        shift_box: RegisterBox,
    ) -> QRegMap:
        qreg_map = QRegMap(
            [shift_box.qreg.state, shift_box.qreg.prepare],
            [
                qreg.state[control_idx],
                qreg.block[control_idx],
            ],
        )
        return qreg_map

    def _generate_control_qreg_map(
        self,
        control_idx: int,
        k: int,
        qreg: LCUStatePreparationQReg,
        be_box: QControlRegisterBox,
    ) -> QRegMap:
        qreg_map = QRegMap(
            [
                be_box.qreg.state,
                be_box.qreg.prepare,
                be_box.qreg.control[0],
            ],
            [
                qreg.state[control_idx],
                qreg.block[control_idx],
                qreg.coeffs[control_idx][k],
            ],
        )
        return qreg_map


@dataclass
class FourierBlockEncodingQRegs:
    """Fourier block encoding qubit registers.

    Attributes
    ----------
        block (QubitRegister): The register of the encoded block.

    """

    block: QubitRegister


class FourierBlockEncoding(BaseLCUStatePreparationBlockEncoding):
    """FourierBlockEncoding class.

    This class represents a general block-encoding of the Fourier basis and contains a
    method called ```generate_basis_element``` that, given an index k, generates a
    register box containing the block encoding of the k-th Fourier basis function.

    Args:
    ----
        n_block_qubits (int): Number of block qubits

    """

    def __init__(self, n_block_qubits: int):
        """Initialise FourierBlockEncoding."""
        super().__init__(n_block_qubits=n_block_qubits)
        self._control_state = None
        self._n_qubits = self._n_block_qubits

    @property
    def control_state(self) -> None:
        """Return the control bitstring of the block encoding."""
        return self._control_state

    @property
    def n_qubits(self) -> int:
        """Return the total number of qubits."""
        return self._n_qubits

    def generate_basis_element(self, basis_index: int) -> RegisterBox:
        """Generate a RegisterBox containing the block-encoding.

        Given the fourier index k, this method generates the block-encoding of the k-th
        Fourier basis function using Z-rotation operators.

        Registers:
            ancilla_qreg (QubitRegister): Ancilla register block encoding.
            block_qreg (QubitRegister): Block register of block encoding.

        Args:
        ----
            basis_index (int): Index of the basis element.

        Returns:
        -------
            RegisterBox: Fourier-index-th Fourier basis block encoding.

        """
        pauli_list = [
            Rz_jkn(j, basis_index, self.n_block_qubits)
            for j in range(self.n_block_qubits - 1, -1, -1)
        ]
        circ = RegisterCircuit(self.__class__.__name__)
        block_qreg = circ.add_q_register(name="block", size=self.n_block_qubits)
        for j in range(self.n_block_qubits):
            circ.add_gate(pauli_list[j], [block_qreg[j]])
        circ.add_phase(basis_index / 2)

        qregs = FourierBlockEncodingQRegs(block=block_qreg)

        return RegisterBox(qregs, circ)

    def _generate_shift_qreg_map(
        self,
        control_idx: int,
        qreg: LCUStatePreparationQReg,
        shift_box: RegisterBox,
    ) -> QRegMap:
        qreg_map = QRegMap(
            [shift_box.qreg.block],  # TODO why .block here? This is because Fourier
            # box is generated differently, so it does not has a state register but
            # a block register. Probably good to change this.
            [qreg.state[control_idx]],
        )
        return qreg_map

    def _generate_control_qreg_map(
        self,
        control_idx: int,
        k: int,
        qreg: LCUStatePreparationQReg,
        be_box: QControlRegisterBox,
    ) -> QRegMap:
        qreg_map = QRegMap(
            [be_box.qreg.block, be_box.qreg.control[0]],
            [
                qreg.state[control_idx],
                qreg.coeffs[control_idx][k],
            ],
        )
        return qreg_map
