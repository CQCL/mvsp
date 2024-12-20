"""LCUBox for a Pauli operator using multiplexors."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pytket._tket.unit_id import QubitRegister
from pytket.circuit import (
    DiagonalBox,
    StatePreparationBox,
)

from mvsp.circuits.core import QRegMap, RegisterBox
from mvsp.circuits.core.register_circuit import RegisterCircuit
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation_block_encoding import (
    BaseLCUStatePreparationBlockEncoding,
)
from mvsp.circuits.utils import (
    extend_functions,
    generate_diagonal_entries,
)


@dataclass
class LCUStatePreparationQReg:
    """LCUStatePreparationBox qubit registers.

    Attributes
    ----------
        coeffs (list[QubitRegister]): Register for the
            Fourier/Chebychev coefficients.
        state (list[QubitRegister]): The state register.
        block (list[QubitRegister]): The block encoding register.

    """

    coeffs: list[QubitRegister]
    state: list[QubitRegister]
    block: list[QubitRegister]


class LCUStatePreparationBox(RegisterBox):
    """LCUStatePreparationBox using multiplexing.

    This box prepares a quantum state whose amplitudes encode multivariate functions by
    linearly combining block-encodings of Fourier and Chebyshev basis functions,
    following https://arxiv.org/pdf/2405.21058.

    Registers:
        coeffs_qreg (list[QubitRegister]): The coefficients registers.
        state_qreg (list[QubitRegister]): The state registers.
        block_qreg (list[QubitRegister]): The block encoding registers.

    Args:
    ----
        coeffs (NDArray): Coefficients of the expansion in terms of the
            block encoded basis functions.
        basis_function_block_encoding
            (list[BaseLCUStatePreparationBlockEncoding]): Block encodings
            for each dimension. Every Block Encoding in this list needs a method
            called generate_basis_element.
        min_basis_indices (list | None, optional): List of minimal Fourier
            indices, e.g. needed if negative frequencies are used.
            All Fourier coefficients of indices from
            min_basis_indices[i] to
            min_basis_indices[i] + dims_fourier_variables[i]
            must be specified.
            If None they are automatically set to zero.
            Defaults to None.

    """

    def __init__(
        self,
        coeffs: NDArray[np.complex128 | np.floating[Any]],
        basis_function_block_encoding: list[BaseLCUStatePreparationBlockEncoding],
        min_basis_indices: list[int] | None = None,
    ):
        """Initialise the LCUStatePreparationBox."""
        dims_variables: list[int] = []
        for be in basis_function_block_encoding:
            dims_variables.append(2**be.n_block_qubits)

        self._basis_function_block_encoding = basis_function_block_encoding

        coeffs_padded = self._get_coeffs_padded(coeffs)
        dims_coeffs_variables = coeffs_padded.shape
        coeffs_abs = np.abs(coeffs_padded)
        coeffs_phi = np.angle(coeffs_padded)

        self._dims_coeffs_variables = dims_coeffs_variables
        self._dims_variables = dims_variables
        self._min_basis_indices = min_basis_indices
        basis_indices = self._get_basis_indices()
        self._basis_indices = basis_indices

        _, _, n_qubits, n_qubits_coeffs, coeffs_abs, coeffs_phi = extend_functions(
            dims_variables, coeffs_abs, coeffs_phi
        )
        self._coeffs_abs = coeffs_abs
        self._coeffs_phi = coeffs_phi

        self._n_qubits_state = n_qubits
        self._n_qubits_coeffs = n_qubits_coeffs

        dims_coeffs_variables_extended = [2**n for n in n_qubits_coeffs]
        self._dims_coeffs_variables_extended = dims_coeffs_variables_extended

        n_qubits_block_encoding = self._get_n_qubits_block_encoding()
        self._n_qubits_block_encoding = n_qubits_block_encoding

        circ = RegisterCircuit(self.__class__.__name__)

        # Define registers
        coeffs_qreg = [
            circ.add_q_register(f"Coeffs_register_{i}", n)
            for i, n in enumerate(n_qubits_coeffs)
        ]
        state_qreg = [
            circ.add_q_register(f"State_register_{i}", n)
            for i, n in enumerate(n_qubits)
        ]
        block_qreg = [
            circ.add_q_register(
                f"Block_encoding_register_{i}", n_qubits_block_encoding[i]
            )
            for i in range(len(n_qubits))
            if n_qubits_block_encoding[i] > 0
        ]
        qreg = LCUStatePreparationQReg(
            coeffs=coeffs_qreg, state=state_qreg, block=block_qreg
        )

        # Hadamards
        for q in [x for y in qreg.state for x in y]:
            circ.H(q)

        # State Prep
        coeffs_normalised = np.sqrt(coeffs_abs.flatten() / np.sum(coeffs_abs.flatten()))
        A = StatePreparationBox(coeffs_normalised)
        reg_box_A = RegisterBox.from_CircBox(A, circuit_name="StatePreparationBox")
        qreg_map_A = QRegMap(reg_box_A.qubits, [x for y in qreg.coeffs for x in y])
        circ.add_registerbox(reg_box_A, qreg_map_A)

        # Diagonal
        D = DiagonalBox(self._get_diagonal_entries())
        reg_box_D = RegisterBox.from_CircBox(D, circuit_name="DiagonalBox")
        qreg_map_D = QRegMap(reg_box_D.qubits, [x for y in qreg.coeffs for x in y])
        circ.add_registerbox(reg_box_D, qreg_map_D)

        # Block Encoding
        for i in range(len(n_qubits)):
            basis_function_block_encoding[i].add_controlled_block_encoding_sequence(
                n_qubits_coeffs=n_qubits_coeffs[i],
                basis_indices=basis_indices[i],
                circ=circ,
                qreg=qreg,
                control_idx=i,
            )

        # State Prep Dagger
        reg_box_A_dagger = reg_box_A.dagger
        circ.add_registerbox(reg_box_A_dagger, qreg_map_A)

        super().__init__(qreg, circ)

        self._postselect.update({q: 0 for qreg in self.qreg.coeffs for q in qreg})
        self._postselect.update({q: 0 for qreg in self.qreg.block for q in qreg})

    @property
    def n_qubits_state(self) -> list[int]:
        """Return the number of state qubits for each variable."""
        return self._n_qubits_state

    @property
    def n_qubits_coeffs(self) -> list[int]:
        """Return the number of coeffs qubits for each variable."""
        return self._n_qubits_coeffs

    @property
    def n_qubits_block_encoding(self) -> list[int]:
        """Return the number of block-encoding qubits for each variable."""
        return self._n_qubits_block_encoding

    @property
    def coeffs_abs(self) -> NDArray[np.floating[Any]]:
        """Return the amplitude of the complex coefficients."""
        return self._coeffs_abs

    @property
    def coeffs_phi(self) -> NDArray[np.floating[Any]]:
        """Return the phases of the complex coefficients."""
        return self._coeffs_phi

    @property
    def dims_variables(self) -> list[int]:
        """Return the dimension of each variable."""
        return self._dims_variables

    @property
    def dims_coeffs_variables_extended(self) -> list[np.floating[Any]]:
        """Return the dimensions of the coefficients."""
        return self._dims_coeffs_variables_extended

    @staticmethod
    def _get_coeffs_padded(
        coeffs: NDArray[np.complex128 | np.floating[Any]],
    ) -> NDArray[np.complex128 | np.floating[Any]]:
        """Extend coefficients.

        This method extends the array of coefficients to the corresponding shape.

        Args:
        ----
            coeffs (NDArray): Coefficients of the expansion in terms of the
                block encoded basis functions.

        Returns:
        -------
            NDArray: The extended coefficients.

        """
        if min(coeffs.shape) == 1:
            # We have to extend the coefficients.
            padder: list[tuple[int, int]] = []
            for i in coeffs.shape:
                if i == 1:
                    padder.append((0, 1))
                else:
                    padder.append((0, 0))
            coeffs_padded = np.pad(coeffs, pad_width=padder)
        else:
            coeffs_padded = coeffs

        return coeffs_padded

    def _get_basis_indices(self) -> list[list[int]]:
        """Compute the basis indices for each variable."""
        if self._min_basis_indices is None:
            self._min_basis_indices = [0 for _ in range(len(self._dims_variables))]
        basis_indices = [
            list(
                range(
                    self._min_basis_indices[i],
                    self._min_basis_indices[i] + self._dims_coeffs_variables[i],
                )
            )
            for i in range(len(self._min_basis_indices))
        ]
        return basis_indices

    def _get_n_qubits_block_encoding(self) -> list[int]:
        """Compute the number of qubits required for block encoding."""
        return [
            self._basis_function_block_encoding[i].n_qubits - self._n_qubits_state[i]
            for i in range(len(self._n_qubits_state))
        ]

    def _get_diagonal_entries(self) -> NDArray[np.complex128]:
        """Compute the entries for the Diagonal Box."""
        return np.array(
            [
                generate_diagonal_entries(
                    self._coeffs_phi[k],
                )
                for k in product(
                    *[range(dims) for dims in self._dims_coeffs_variables_extended]
                )
            ]
        )


class SeparableLCUStatePreparationBox(RegisterBox):
    """SeparableLCUStatePreparationBox using multiplexing.

    Special case of LCUStatePreparationBox where the multivariate function is separable.
    In this case the protocol can be simplified to encode each variable independently.

    Registers:
        coeffs_qreg (list[QubitRegister]): The coefficients registers.
        state_qreg (list[QubitRegister]): The state registers.
        block_qreg (list[QubitRegister]): The block encoding registers.

    Args:
    ----
        coeffs (NDArray): Coefficients of the expansion in terms of the
            block encoded basis functions.
        basis_function_block_encoding
            (list[BaseLCUStatePreparationBlockEncoding]): Block encodings
            for each dimension. Every Block Encoding in this list needs a method
            called generate_basis_element.
        min_basis_indices (list | None, optional): List of minimal Fourier
            indices, e.g. needed if negative frequencies are used.
            All Fourier coefficients of indices from
            min_basis_indices[i] to
            min_basis_indices[i] + dims_fourier_variables[i]
            must be specified.
            If None they are automatically set to zero.
            Defaults to None.

    """

    def __init__(
        self,
        coeffs: list[NDArray[np.complex128]] | list[NDArray[np.floating[Any]]],
        basis_function_block_encoding: list[BaseLCUStatePreparationBlockEncoding],
        min_basis_indices: list[int] | None = None,
    ):
        """Initialise the SeparableLCUStatePreparationBox."""
        if min_basis_indices is None:
            min_basis_indices = [0 for _ in range(len(basis_function_block_encoding))]

        lcu_state_preparation_boxes = [
            LCUStatePreparationBox(
                coeffs=coeffs[i],
                basis_function_block_encoding=[basis_function_block_encoding[i]],
                min_basis_indices=[min_basis_indices[i]],
            )
            for i in range(len(basis_function_block_encoding))
        ]

        n_qubits_total = np.sum([lcu.n_qubits for lcu in lcu_state_preparation_boxes])
        n_qubits_state = [lcu.n_qubits_state[0] for lcu in lcu_state_preparation_boxes]
        n_qubits_coeffs = [
            lcu.n_qubits_coeffs[0] for lcu in lcu_state_preparation_boxes
        ]
        n_qubits_block_encoding = [
            lcu.n_qubits_block_encoding[0] for lcu in lcu_state_preparation_boxes
        ]

        circ = RegisterCircuit(self.__class__.__name__)
        coeffs_qreg = [
            circ.add_q_register(f"Coeffs_register_{i}", n)
            for i, n in enumerate(n_qubits_coeffs)
        ]
        state_qreg = [
            circ.add_q_register(f"State_register_{i}", n)
            for i, n in enumerate(n_qubits_state)
        ]
        block_qreg = [
            circ.add_q_register(f"Block_encoding_register_{i}", n)
            for i, n in enumerate(n_qubits_block_encoding)
        ]

        qreg_maps = [
            QRegMap(
                lcu.qubits,
                list(block_qreg[idx]) + list(coeffs_qreg[idx]) + list(state_qreg[idx]),
            )
            for idx, lcu in enumerate(lcu_state_preparation_boxes)
        ]
        for lcu, qreg_map in zip(lcu_state_preparation_boxes, qreg_maps, strict=True):
            circ.add_registerbox(lcu, qreg_map)

        clean_block_qreg: list[QubitRegister] = []
        for i, qr in enumerate(block_qreg):
            if n_qubits_block_encoding[i] > 0:
                clean_block_qreg.append(qr)

        qreg = LCUStatePreparationQReg(
            coeffs=coeffs_qreg, state=state_qreg, block=clean_block_qreg
        )
        super().__init__(qreg, circ)

        self.n_qubits_total = n_qubits_total
        self._coeffs_abs = [lcu.coeffs_abs for lcu in lcu_state_preparation_boxes]
        self._coeffs_phi = [lcu.coeffs_phi for lcu in lcu_state_preparation_boxes]
        self._dims_coeffs_variables_extended = [
            lcu.dims_coeffs_variables_extended for lcu in lcu_state_preparation_boxes
        ]
        self._dims_variables = [
            lcu.dims_variables for lcu in lcu_state_preparation_boxes
        ]
        self._n_qubits_state = n_qubits_state
        self._n_qubits_coeffs = n_qubits_coeffs
        self._n_qubits_block_encoding = n_qubits_block_encoding

    @property
    def n_qubits_state(self) -> list[int]:
        """Return the number of state qubits for each variable."""
        return self._n_qubits_state

    @property
    def n_qubits_coeffs(self) -> list[int]:
        """Return the number of coeffs qubits for each variable."""
        return self._n_qubits_coeffs

    @property
    def n_qubits_block_encoding(self) -> list[int]:
        """Return the number of block-encoding qubits for each variable."""
        return self._n_qubits_block_encoding

    @property
    def coeffs_abs(self) -> list[NDArray[np.floating[Any]]]:
        """Return the amplitude of the complex coefficients."""
        return self._coeffs_abs

    @property
    def coeffs_phi(self) -> list[NDArray[np.floating[Any]]]:
        """Return the phases of the complex coefficients."""
        return self._coeffs_phi

    @property
    def dims_variables(self) -> list[list[int]]:
        """Return the dimension of each variable."""
        return self._dims_variables

    @property
    def dims_coeffs_variables_extended(self) -> list[list[np.floating[Any]]]:
        """Return the dimensions of the coefficients."""
        return self._dims_coeffs_variables_extended
