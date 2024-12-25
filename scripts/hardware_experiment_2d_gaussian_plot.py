"""Plot hardware experiment results."""

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use(os.path.abspath("plots/paper.mplstyle"))
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_colorbar(plot, fig, ax):
    cb_formatter = ScalarFormatter(useMathText=True)
    cb_formatter.set_scientific("%.2e")
    cb_formatter.set_powerlimits((-2, 2))
    cb = fig.colorbar(
        plot, ax=ax, format=cb_formatter, location="right", shrink=1.0, pad=-0.06
    )
    cb.ax.yaxis.set_offset_position("left")
    cb.ax.tick_params(labelsize="small")
    cb.ax.yaxis.get_offset_text().set(size="small")


n = 9
c = 7
mu_x = 0.5
mu_y = 0.5
s_x = 0.22
s_y = 0.18

folder_path = os.path.abspath("data/hardware_experiment/")
# Import data for uncorrelated Gaussian
print("Load data, uncorrelated Gaussian")
rho = 0.0
load_path = os.path.join(folder_path, f"density_dict_n{n}_c{c}_rho{rho}.pkl")
with open(load_path, "rb") as file:
    density_dict = pickle.load(file)


# Plot uncorrelated Gaussian
fig = plt.figure(figsize=[6.72, 2.42], constrained_layout=True)
gs0 = fig.add_gridspec(2, 8)
gs1 = gs0[:, :2].subgridspec(2, 1)
ax_raw_2D = fig.add_subplot(gs1[0, 0])
ax_kde_2D = fig.add_subplot(gs1[1, 0])
ax_raw_marginals = fig.add_subplot(gs0[:, 2:5])
ax_truth_marginals = fig.add_subplot(gs0[:, 5:8])


cmesh = ax_raw_2D.hist2d(
    density_dict["outcomes_x"], density_dict["outcomes_y"], bins=20, cmap="Reds"
)
plot_colorbar(cmesh[3], fig, ax_raw_2D)
ax_raw_2D.set_title("Raw data")
ax_raw_2D.set_xticks([0.0, 0.5, 1.0])
ax_raw_2D.set_yticks([0.0, 0.5, 1.0])
ax_raw_2D.set_ylabel("y")
ax_raw_2D.set_aspect("equal")

cmesh = ax_kde_2D.pcolormesh(
    *np.meshgrid(*density_dict["density_full_coarse_coordinates"]),
    density_dict["density_kde_full_coarse"],
    cmap="Reds",
)
plot_colorbar(cmesh, fig, ax_kde_2D)
ax_kde_2D.set_title(r"$\hat{\mathfrak{f}}_d(x,y)$")
ax_kde_2D.set_xticks([0.0, 0.5, 1.0])
ax_kde_2D.set_yticks([0.0, 0.5, 1.0])
ax_kde_2D.set_xlabel("x")
ax_kde_2D.set_ylabel("y")
ax_kde_2D.set_aspect("equal")

ax_raw_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_raw_x"],
    label=r"$w_x(x)$",
    color=color_cycle[0],
    alpha=0.35,
    linestyle="solid",
    linewidth=0.5,
)
ax_raw_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_raw_kde_x"],
    label=r"$k_x(x)$",
    color=color_cycle[0],
    linestyle="solid",
)
ax_raw_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_raw_y"],
    label=r"$w_y(y)$",
    color=color_cycle[1],
    alpha=0.35,
    linestyle="solid",
    linewidth=0.5,
)
ax_raw_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_raw_kde_y"],
    label=r"$k_y(y)$",
    color=color_cycle[1],
    linestyle="solid",
)

ax_raw_marginals.set_title("Marginal kernel density estimates")
ax_raw_marginals.set_xlabel("x, y")
ax_raw_marginals.set_ylabel(r"probability [1e-2]")
ax_raw_marginals.set_yticks([0.0 * 1e-2, 0.5 * 1e-2, 1.0 * 1e-2], [0.0, 0.5, 1.0])
ax_raw_marginals.legend(loc=1)

labelstr = r"$\hat{\mathfrak{f}}_{d,x}(x)$"

ax_truth_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_kde_x"],
    label=labelstr,
    color=color_cycle[0],
)
labelstr = "\n".join((r"$\mathfrak{f}_{d,x}(x)$",))
ax_truth_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_ground_truth_fourier_x"],
    label=labelstr,
    color=color_cycle[0],
    linestyle="dashed",
)


labelstr = r"$\hat{\mathfrak{f}}_{d,y}(y)$"
ax_truth_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_kde_y"],
    label=labelstr,
    color=color_cycle[1],
)
labelstr = "\n".join((r"$\mathfrak{f}_{d,y}(y)$",))
ax_truth_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_ground_truth_fourier_y"],
    label=labelstr,
    color=color_cycle[1],
    linestyle="dashed",
)
ax_truth_marginals.set_title("Comparison to target")
ax_truth_marginals.set_xlabel("x, y")
ax_truth_marginals.set_ylabel("density")
ax_truth_marginals.legend(loc=1)

title_string = "".join(("Uncorrelated 2D Gaussian",))
fig.suptitle(title_string, fontsize=9)
fig.text(x=0.03, y=0.92, s="(a)")
fig.text(x=0.25, y=0.92, s="(b)")
fig.text(x=0.65, y=0.92, s="(c)")

save_file = os.path.abspath(
    f"plots/2D_gauss_fourier_n{n}_c{c}_mu-{mu_x}-{mu_y}_sigma-{s_x}-{s_y}_rho-{rho}_hardware.pdf"
)
fig.savefig(
    save_file, transparent=False, bbox_inches="tight", pad_inches="layout", dpi=600
)


# ======================================================================================

# Import data for correlated Gaussian
print("Load data, uncorrelated Gaussian")
rho = 0.4
load_path = os.path.join(folder_path, f"density_dict_n{n}_c{c}_rho{rho}.pkl")
with open(load_path, "rb") as file:
    density_dict = pickle.load(file)


# Plot correlated Gaussian
fig = plt.figure(figsize=[6.72, 2.42], constrained_layout=True)


gs0 = fig.add_gridspec(2, 8)
gs1 = gs0[:, :2].subgridspec(2, 1)
ax_raw_2D = fig.add_subplot(gs1[0, 0])
ax_kde_2D = fig.add_subplot(gs1[1, 0])
ax_raw_marginals = fig.add_subplot(gs0[:, 2:5])
ax_truth_marginals = fig.add_subplot(gs0[:, 5:8])


cmesh = ax_raw_2D.hist2d(
    density_dict["outcomes_x"], density_dict["outcomes_y"], bins=20, cmap="Reds"
)
plot_colorbar(cmesh[3], fig, ax_raw_2D)
ax_raw_2D.set_title("Raw data")
ax_raw_2D.set_xticks([0.0, 0.5, 1.0])
ax_raw_2D.set_yticks([0.0, 0.5, 1.0])
ax_raw_2D.set_ylabel("y")
ax_raw_2D.set_aspect("equal")

cmesh = ax_kde_2D.pcolormesh(
    *np.meshgrid(*density_dict["density_full_coarse_coordinates"]),
    density_dict["density_kde_full_coarse"],
    cmap="Reds",
)
plot_colorbar(cmesh, fig, ax_kde_2D)
# fig.colorbar(cmesh, ax=ax_kde_2D)
ax_kde_2D.set_title(r"$\hat{\mathfrak{f}}_d(x,y)$")
ax_kde_2D.set_xticks([0.0, 0.5, 1.0])
ax_kde_2D.set_yticks([0.0, 0.5, 1.0])
ax_kde_2D.set_xlabel("x")
ax_kde_2D.set_ylabel("y")
ax_kde_2D.set_aspect("equal")

ax_raw_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_raw_x"],
    label=r"$w_x(x)$",
    color=color_cycle[0],
    alpha=0.35,
    linestyle="solid",
    linewidth=0.5,
)
ax_raw_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_raw_kde_x"],
    label=r"$k_x(x)$",
    color=color_cycle[0],
    linestyle="solid",
)
ax_raw_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_raw_y"],
    label=r"$w_y(y)$",
    color=color_cycle[1],
    alpha=0.35,
    linestyle="solid",
    linewidth=0.5,
)
ax_raw_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_raw_kde_y"],
    label=r"$k_y(y)$",
    color=color_cycle[1],
    linestyle="solid",
)
ax_raw_marginals.set_title("Marginal kernel density estimates")
ax_raw_marginals.set_xlabel("x, y")
ax_raw_marginals.set_ylabel(r"probability [1e-2]")
ax_raw_marginals.set_yticks(
    [0.0 * 1e-2, 0.3 * 1e-2, 0.6 * 1e-2, 0.9 * 1e-2, 1.2 * 1e-2],
    [0.0, 0.3, 0.6, 0.9, 1.2],
)
ax_raw_marginals.legend(loc=1)

labelstr = r"$\hat{\mathfrak{f}}_{d,x}(x)$"
ax_truth_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_kde_x"],
    label=labelstr,
    color=color_cycle[0],
)
labelstr = "\n".join((r"$\mathfrak{f}_{d,x}(x)$",))
ax_truth_marginals.plot(
    density_dict["density_x_coordinates"],
    density_dict["density_ground_truth_fourier_x"],
    label=labelstr,
    color=color_cycle[0],
    linestyle="dashed",
)

labelstr = r"$\hat{\mathfrak{f}}_{d,y}(y)$"
ax_truth_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_kde_y"],
    label=labelstr,
    color=color_cycle[1],
)
labelstr = "\n".join((r"$\mathfrak{f}_{d,y}(y)$",))
ax_truth_marginals.plot(
    density_dict["density_y_coordinates"],
    density_dict["density_ground_truth_fourier_y"],
    label=labelstr,
    color=color_cycle[1],
    linestyle="dashed",
)
ax_truth_marginals.set_title("Comparison to target")
ax_truth_marginals.set_xlabel("x, y")
ax_truth_marginals.set_ylabel("density")
ax_truth_marginals.legend(loc=1)

title_string = "".join(("Correlated 2D Gaussian",))
fig.suptitle(title_string, fontsize=9)
fig.text(x=0.03, y=0.92, s="(d)")
fig.text(x=0.25, y=0.92, s="(e)")
fig.text(x=0.65, y=0.92, s="(f)")

save_file = os.path.abspath(
    f"plots/2D_gauss_fourier_n{n}_c{c}_mu-{mu_x}-{mu_y}_sigma-{s_x}-{s_y}_rho-{rho}_hardware.pdf"
)
fig.savefig(
    save_file, transparent=False, bbox_inches="tight", pad_inches="layout", dpi=600
)
