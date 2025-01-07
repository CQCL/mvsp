import itertools
import os
from time import time

import numpy as np
from numpy.linalg import eigh
from pandas import DataFrame
from pytket.extensions.qiskit import AerStateBackend
from scipy.linalg import ishermitian

from mvsp.applications.chemistry import plane_wave_hamiltonian
from mvsp.applications.chemistry.lattices import center
from mvsp.applications.chemistry.wavefunctions import plane_wave, plane_wave_renorm
from mvsp.circuits.lcu_state_preparation import (
    FourierBlockEncoding,
    LCUStatePreparationBox,
)
from mvsp.measurement.utils import recursive_statevector_postselect

k_neg = 4
k_min = -k_neg
k_max = k_neg - 1
ks = np.arange(k_min, k_max + 1)
k_dim = len(ks)
k_point_grid = itertools.product(ks, ks, ks)
k_point_grid_array = [np.array(k) for k in k_point_grid]

cell_length = 1.0
cell_volume = cell_length**3

center_lattice = center(cell_length)

n_space_qubits = 4
space_dim = 2**n_space_qubits
num_points = space_dim
L = 1
xs = np.linspace(0, L, num_points)
ys = np.linspace(0, L, num_points)
zs = np.linspace(0, L, num_points)
X, Y, Z = np.meshgrid(xs, ys, zs)
R = np.stack((X, Y, Z), axis=-1)

dims_variables = (space_dim, space_dim, space_dim)
n_qubits = [int(np.ceil(np.log2(d))) for d in dims_variables]

data_path = os.path.abspath("data/electron_in_Coulomb_potential/")
os.makedirs(data_path, exist_ok=True)


def lcu_circ(coeffs_array, title):
    """Generate the LCU state preparation circuit for a given set of coefficients."""
    coeffs_array = coeffs_array.reshape((k_dim, k_dim, k_dim))

    coeffs_array = np.swapaxes(coeffs_array, 0, -1)
    coeff_t = np.transpose(coeffs_array, (1, 2, 0))

    lcuspbox = LCUStatePreparationBox(
        coeff_t,
        [FourierBlockEncoding(n) for n in n_qubits],
        min_basis_indices=[k_min, k_min, k_min],
    )

    print("BOX Qubits:", lcuspbox.n_qubits)

    # SV backend

    backend = AerStateBackend()
    circ_compiled = backend.get_compiled_circuit(
        lcuspbox.get_circuit().copy(), optimisation_level=0
    )
    print("2Q gates: ", circ_compiled.n_2qb_gates())
    print("N Gates: ", circ_compiled.n_gates)
    print("depth: ", circ_compiled.depth())
    print("depth 2q: ", circ_compiled.depth_2q())

    # save the above prints to dataframe
    df = DataFrame.from_dict(
        {
            "N qubits": [circ_compiled.n_qubits],
            "2Q gates": [circ_compiled.n_2qb_gates()],
            "N Gates": [circ_compiled.n_gates],
            "Depth": [circ_compiled.depth()],
            "Depth 2Q": [circ_compiled.depth_2q()],
        }
    )

    filename = os.path.join(data_path, f"{title}_{space_dim}s_{k_dim}k_circ_data.csv")
    df.to_csv(filename)

    vec = backend.run_circuit(circ_compiled).get_state()

    vec_projected = recursive_statevector_postselect(
        circ_compiled.qubits, vec, lcuspbox.postselect.copy()
    )

    vec_projected_scaled = vec_projected
    res = vec_projected_scaled.reshape(dims_variables)
    res_sv = res / np.linalg.norm(res)

    return res_sv


def numpy_res(coeff_dict):
    """Generate the numpy array for the plane wave function."""
    pw_3d = plane_wave(coeff_dict)
    pw_3d_r = pw_3d(R)
    pw_3d_renorm = plane_wave_renorm(pw_3d_r)

    return pw_3d_renorm


def compute_3d_data(r_pos, title):
    """Plot the 3D scatter plot for the crystal.

    For the given crystal, plot the numerical and circuit wavefunctions.

    Args:
    ----
        r_pos (list): List of numpy arrays representing the positions of the crystal
        title (str): Title of the

    """
    # Consider the first `num_ev` wave functions
    num_ev = 2
    t0 = time()
    H = plane_wave_hamiltonian(k_point_grid_array, cell_volume, r_pos)
    print("Time taken", time() - t0)
    print(f"Is H Hermitian? {ishermitian(H)}")

    # Use eigh since H is Hermitian (verify from output above). This guarantees
    # that eigenvectors are ordered by eigenvalue.
    e, c = eigh(H)

    filename = os.path.join(data_path, f"{title}_{space_dim}s_{k_dim}k_eigen.npz")
    np.savez(filename, e=e[:num_ev], c=c[:, :num_ev], allow_pickle=False)

    print("Verify eigenvalues/-vectors")
    for i in range(num_ev):
        print(f"e_{i}={e[i]}")
        print(
            f"Is H * v_{i} = e_{i} * v_{i}?",
            np.allclose(e[i] * c[:, i], H @ c[:, i]),
        )

    coeffs = []
    for i in range(num_ev):
        coeffs.append(
            {
                tuple(k): coeff
                for k, coeff in zip(k_point_grid_array, c[:, i], strict=False)
            }
        )

    res_list = [(c[:, i], coeffs[i], None) for i in range(num_ev)]

    for i, res in enumerate(res_list):
        np_res = numpy_res(res[1])
        filename = os.path.join(
            data_path, f"{title}_{i:02}_{space_dim}s_{k_dim}k_np_result.npy"
        )
        np.save(filename, np_res)

        circ_res = lcu_circ(res[0], title)
        filename = os.path.join(
            data_path, f"{title}_{i:02}_{space_dim}s_{k_dim}k_circ_result.npy"
        )
        np.save(filename, circ_res)


if __name__ == "__main__":
    compute_3d_data(center_lattice, "center")
