import itertools
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from numpy.linalg import eig, eigh
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


def luc_circ(coeffs_array, title):
    """Generate the LCU state preparation circuit for a given set of coefficients."""
    coeffs_array = coeffs_array.reshape((k_dim, k_dim, k_dim))

    coeffs_array = np.swapaxes(coeffs_array, 0, -1)
    coeff_t = np.transpose(coeffs_array, (1, 2, 0))

    lcuspbox = LCUStatePreparationBox(
        coeff_t,
        [FourierBlockEncoding(n) for n in n_qubits],
        min_basis_indices=[k_min, k_min, k_min],
    )
    lcuspbox.get_circuit().n_qubits

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
    filename = os.path.abspath(
        f"data/electron_in_Coulomb_potential/{title}_{space_dim}s_{k_dim}k_circ_data.csv"
    )
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


def plot_3d_scatter_crystal(r_pos, title):
    """Plot the 3D scatter plot for the crystal.

    For the given crystal, plot the numerical and circuit wavefunctions.

    Args:
    ----
        r_pos (list): List of numpy arrays representing the positions of the crystal
        title (str): Title of the

    """
    t0 = time()
    H = plane_wave_hamiltonian(k_point_grid_array, cell_volume, r_pos)
    print(H)
    print(f"Is H Hermitian? {ishermitian(H)}")
    print("Time taken", time() - t0)

    # eig is not guaranteed to return sorted eigenvalues but we assume correct
    # ordering below
    e, c = eig(H)

    print("Eigenvalues", e)
    print("Coefficients", c)
    for i in range(2):
        print(
            f"Is {i}th eigenvalue/vector correct?",
            np.allclose(e[i] * c[:, i], H @ c[:, i]),
        )

    coeffs = []
    for i in range(len(e)):
        coeffs.append(
            {
                tuple(k): coeff
                for k, coeff in zip(k_point_grid_array, c[i], strict=False)
            }
        )

    res_list = [(c[0], coeffs[0], None), (c[1], coeffs[1], None)]

    fig = plt.figure(figsize=(3.2, 3), constrained_layout=True)
    gs = GridSpec(
        len(res_list), 3, figure=fig, width_ratios=[1, 1, 0.08], wspace=0.32
    )  # 3 columns: 2 for plots, 1 for colorbar

    for i, res in enumerate(res_list):
        ax1 = fig.add_subplot(gs[i, 0], projection="3d")  # First plot
        ax2 = fig.add_subplot(gs[i, 1], projection="3d")  # Second plot
        cax = fig.add_subplot(gs[i, 2])  # Colorbar

        np_res = numpy_res(res[1])
        scatter1 = ax1.scatter(X, Y, Z, c=np_res.flatten().real, cmap="RdBu_r")

        # Set labels
        ax1.set_xlabel(r"$X$", labelpad=-8, fontsize=7)
        ax1.set_ylabel(r"$Y$", labelpad=-8, fontsize=7)
        ax1.set_zlabel(r"$Z$", labelpad=-8, fontsize=7)

        ax1.tick_params(labelsize=7, pad=-3)

        if i == 0:
            ax1.set_title(r"Numerical $\Psi_0(\mathbf{r})$", fontsize=8)
        else:
            ax1.set_title(r"Numerical $\Psi_1(\mathbf{r})$", fontsize=8)

        res = luc_circ(res[0], title)
        scatter2 = ax2.scatter(X, Y, Z, c=res.flatten().real, cmap="RdBu_r")
        cbar = plt.colorbar(
            scatter2,
            cax=cax,
            aspect=2,
            format=ticker.FuncFormatter(lambda x, pos: f"{x * 1e4:.0f}"),
        )  # Use cax for colorbar
        cax.set_title(r"   $\times 10^{-4}$", fontsize=8)
        (cbar.set_label(r"$\Psi(\mathbf{r})$", rotation=0, labelpad=10),)

        # Set labels
        ax2.set_xlabel(r"$X$", labelpad=-8, fontsize=7)
        ax2.set_ylabel(r"$Y$", labelpad=-8, fontsize=7)
        ax2.set_zlabel(r"$Z$", labelpad=-8, fontsize=7)

        ax2.tick_params(labelsize=7, pad=-3)

        if i == 0:
            ax2.set_title(r"Circuit $\Psi_0(\mathbf{r})$", fontsize=8)
        else:
            ax2.set_title(r"Circuit $\Psi_1(\mathbf{r})$", fontsize=8)

    filename = os.path.abspath(
        f"plots/electron_in_Coulomb_potential/{title}_01_{space_dim}s_{k_dim}k.png"
    )
    fig.savefig(filename, dpi=300)


if __name__ == "__main__":
    plot_3d_scatter_crystal(center_lattice, "center")
