"""Plot kernel density cross validation."""

import os
import pickle

from matplotlib import pyplot as plt

plt.style.use("../plots/paper.mplstyle")
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Set parameter
n = 9
c = 7
mu_x = 0.5
mu_y = 0.5
s_x = 0.22
s_y = 0.18

density_switcher = {
    "grid_full_uncor",
    "grid_x_uncor",
    "grid_y_uncor",
    "grid_full_cor",
    "grid_x_cor",
    "grid_y_cor",
    "bandwidths",
}

folder_path = "../data/hardware_experiment/"

# Import data
load_path = os.path.join(folder_path, f"kde_cross_validation_dict_n{n}_c{c}.pkl")
with open(load_path, "rb") as file:
    kde_dict = pickle.load(file)


fig, ax = plt.subplots(ncols=2, figsize=[6.72, 2.42], constrained_layout=True)

ax[0].plot(
    kde_dict["bandwidths"],
    kde_dict["grid_full_uncor"].cv_results_["mean_test_score"],
    color="C2",
    label="2D distribution",
)
ax[0].vlines(
    kde_dict["grid_full_uncor"].best_params_["bandwidth"],
    ymin=-5.0,
    ymax=3.0,
    color="C2",
    linestyle="dashed",
    linewidth=1.0,
)
ax[0].hlines(
    kde_dict["grid_full_uncor"].best_score_,
    xmin=1e-4,
    xmax=3.0,
    color="C2",
    linestyle="dashed",
    linewidth=1.0,
)
ax[0].plot(
    kde_dict["bandwidths"],
    kde_dict["grid_x_uncor"].cv_results_["mean_test_score"],
    color="C0",
    label="x-marginal",
)
ax[0].vlines(
    kde_dict["grid_x_uncor"].best_params_["bandwidth"],
    ymin=-5.0,
    ymax=3.0,
    color="C0",
    linestyle="dashed",
    linewidth=1.0,
)
ax[0].hlines(
    kde_dict["grid_x_uncor"].best_score_,
    xmin=1e-4,
    xmax=3.0,
    color="C0",
    linestyle="dashed",
    linewidth=1.0,
)
ax[0].plot(
    kde_dict["bandwidths"],
    kde_dict["grid_y_uncor"].cv_results_["mean_test_score"],
    color="C1",
    label="y-marginal",
)
ax[0].vlines(
    kde_dict["grid_y_uncor"].best_params_["bandwidth"],
    ymin=-5.0,
    ymax=3.0,
    color="C1",
    linestyle="dashed",
    linewidth=1.0,
)
ax[0].hlines(
    kde_dict["grid_y_uncor"].best_score_,
    xmin=1e-4,
    xmax=3.0,
    color="C1",
    linestyle="dashed",
    linewidth=1.0,
)
ax[0].legend(loc=4)
ax[0].set_xlabel("bandwidth h")
ax[0].set_ylabel(r"$q(h)$")
ax[0].set_xscale("log")
ax[0].set_ylim(-3.5, 1.5)
ax[0].set_xlim(1e-3, 1e0)
ax[0].set_title("Uncorrelated Gaussian")


ax[1].plot(
    kde_dict["bandwidths"],
    kde_dict["grid_full_cor"].cv_results_["mean_test_score"],
    color="C2",
    label="2D distribution",
)
ax[1].vlines(
    kde_dict["grid_full_cor"].best_params_["bandwidth"],
    ymin=-5.0,
    ymax=3.0,
    color="C2",
    linestyle="dashed",
    linewidth=1.0,
)
ax[1].hlines(
    kde_dict["grid_full_cor"].best_score_,
    xmin=1e-4,
    xmax=3.0,
    color="C2",
    linestyle="dashed",
    linewidth=1.0,
)
ax[1].plot(
    kde_dict["bandwidths"],
    kde_dict["grid_x_cor"].cv_results_["mean_test_score"],
    color="C0",
    label="x-marginal",
)
ax[1].vlines(
    kde_dict["grid_x_cor"].best_params_["bandwidth"],
    ymin=-5.0,
    ymax=3.0,
    color="C0",
    linestyle="dashed",
    linewidth=1.0,
)
ax[1].hlines(
    kde_dict["grid_x_cor"].best_score_,
    xmin=1e-4,
    xmax=3.0,
    color="C0",
    linestyle="dashed",
    linewidth=1.0,
)
ax[1].plot(
    kde_dict["bandwidths"],
    kde_dict["grid_y_cor"].cv_results_["mean_test_score"],
    color="C1",
    label="y-marginal",
)
ax[1].vlines(
    kde_dict["grid_y_cor"].best_params_["bandwidth"],
    ymin=-5.0,
    ymax=3.0,
    color="C1",
    linestyle="dashed",
    linewidth=1.0,
)
ax[1].hlines(
    kde_dict["grid_y_cor"].best_score_,
    xmin=1e-4,
    xmax=3.0,
    color="C1",
    linestyle="dashed",
    linewidth=1.0,
)
ax[1].legend(loc=4)
ax[1].set_xlabel("bandwidth h")
ax[1].set_ylabel(r"$q(h)$")
ax[1].set_xscale("log")
ax[1].set_ylim(-3.5, 1.5)
ax[1].set_xlim(1e-3, 1e0)
ax[1].set_title("Correlated Gaussian")


file_dir_path = os.getcwd()
file_path = f"KDE_optimization_n{n}_c{c}_mu-{mu_x}-{mu_y}_sigma-{s_x}-{s_y}.pdf"
fig.savefig(
    "../plots/" + file_path,
    transparent=False,
    bbox_inches="tight",
    pad_inches="layout",
    dpi=600,
)
