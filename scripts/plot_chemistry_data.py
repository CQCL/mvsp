import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import ticker
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


def transform(a: npt.ArrayLike) -> npt.ArrayLike:
    """Transform the raw input wavefunction into the quantity we want to plot.

    Here, we want to plot the modulus square.

    Args:
        a (npt.ArrayLike): Input wavefunction

    Returns:
        npt.ArrayLike: Output quantity to plot

    """
    return np.abs(a) ** 2


def set_vmin_vmax(data: npt.ArrayLike) -> tuple[float, float]:
    """Compute the minimum and maximum of the input dataset.

    This is used as `vmin` and `vmax` respectively in plotting. If minimum and
    maximum have the same sign, assume the input data are probabilities and we
    want to plot them from 0, i.e. set `vmin=0`.

    Args:
        data (npt.ArrayLike): Input data

    Returns:
        tuple[float, float]: Suitable minimum and maximum values of the input
        data.

    """
    vmin, vmax = np.min(np.abs(data)), np.max(np.abs(data))
    if np.sign(np.min(data)) != np.sign(np.max(data)):
        vmin = -vmax
        vmax = vmax
    else:
        vmin = 0
    return vmin, vmax


plt.style.use(os.path.abspath("plots/paper.mplstyle"))
data_path = os.path.abspath("data/electron_in_Coulomb_potential/")
plot_path = os.path.abspath("plots/electron_in_Coulomb_potential/")

title = "center"
n_space_qubits = 7
space_dim = 2**n_space_qubits
num_points = space_dim
k_dim = 2**3

filenames = [
    os.path.join(data_path, f"{title}_00_{space_dim}s_{k_dim}k_np_result.npy"),
    os.path.join(data_path, f"{title}_01_{space_dim}s_{k_dim}k_np_result.npy"),
    os.path.join(data_path, f"{title}_00_{space_dim}s_{k_dim}k_circ_result.npy"),
    os.path.join(data_path, f"{title}_01_{space_dim}s_{k_dim}k_circ_result.npy"),
]
datasets = []
for filename in filenames:
    # swapaxes and mesgrid with indexing='ij' needed to put the 3D input data
    # into the right format for ploting 2D slices of the 3D data
    datasets.append(np.load(filename, allow_pickle=False).swapaxes(0, 1))

xs = np.linspace(0, 1, num_points)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")

# Specify slices to plot
# zdirs are the axes of the slices
# zss are the locations on the corresponding axes
# Finally, slices is a boolean array selecting the corresponding entries in the 3D input
zdirs = ["y", "x", "x", "x"]
zss = [xs[-1], xs[0], xs[int(num_points / 2)], xs[-1]]
slices = [
    {"x": X, "y": Y, "z": Z}[zdir] == zs for zdir, zs in zip(zdirs, zss, strict=False)
]

data = []
vmin_vmax = []
for dataset in datasets:
    for slice_ in slices:
        data.append(transform(dataset[slice_]).flatten())
    # Compute vmin, vmax from all slices that will be plotted in the same axis.
    # This includes the last `len(slices)` datasets.
    vmin_vmax.append(set_vmin_vmax(data[-len(slices) :]))

num_rows = int(len(datasets) / 2)
fig = plt.figure(figsize=(3.22, 3), constrained_layout=True)
gs = GridSpec(
    num_rows,
    3,
    figure=fig,
    width_ratios=[1, 1, 0.06],
    wspace=0.33,
)
axs = []
for i in range(num_rows):
    axs.append(fig.add_subplot(gs[0, i], projection="3d"))
    axs.append(fig.add_subplot(gs[1, i], projection="3d"))
caxs = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]

marker_size = 72 / num_points
# Only need 2D meshes rather than the 3D meshes from before
X, Y = np.meshgrid(xs, xs, indexing="ij")

plots = []
for ax, dataset, (vmin, vmax) in zip(axs, datasets, vmin_vmax, strict=False):
    for slice_, zdir, zs in zip(slices, zdirs, zss, strict=False):
        plots.append(
            ax.scatter(
                X,
                Y,
                s=marker_size,
                zs=zs,
                zdir=zdir,
                c=transform(dataset[slice_]).flatten(),
                alpha=1.0,
                edgecolor="none",
                cmap="Reds",
                norm=Normalize(vmin, vmax),
            ),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("x", labelpad=-8, fontsize=7)
    ax.set_ylabel("y", labelpad=-8, fontsize=7)
    ax.set_zlabel("z", labelpad=-8, fontsize=7)
    ax.tick_params(labelsize=7, pad=-2)

for i, ax in enumerate(axs):
    if i < num_rows:
        num_or_circ = "Numerical"
    else:
        num_or_circ = "Circuit"
    ax.set_title(rf"{num_or_circ} $|\Psi_{i % num_rows}" + r"(\mathbf{r})|^2$")

cbar = plt.colorbar(
    plots[0],
    cax=caxs[0],
    aspect=2,
    format=ticker.FuncFormatter(lambda x, pos: f"{x * 1e7:.0f}"),
)
caxs[0].set_title(
    r"   $\times 10^{-7}$",
)
# cbar.set_label(r"$\Psi(\mathbf{r})$", rotation=0, labelpad=10)

cbar = plt.colorbar(
    plots[len(slices)],
    cax=caxs[1],
    aspect=2,
    format=ticker.FuncFormatter(lambda x, pos: f"{x * 1e7:.0f}"),
)
caxs[1].set_title(
    r"   $\times 10^{-7}$",
)
# cbar.set_label(r"$\Psi(\mathbf{r})$", rotation=0, labelpad=10)

filename = os.path.join(plot_path, f"{title}_{space_dim}s_{k_dim}k_cut.png")
fig.savefig(filename, dpi=300)
