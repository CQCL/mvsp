"""Generate kernel density cross validation results."""

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

from mvsp.utils.shot_distribution import ShotDistribution, generate_outcome_dict


def classical_fidelity(density_1, density_2, normalize=True):
    if normalize:
        density_1 = density_1 / np.sum(density_1)
        density_2 = density_2 / np.sum(density_2)
    return np.dot(np.sqrt(density_1), np.sqrt(density_2)) ** 2


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


color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def generate_kde_cross_validation_data(rho: float):
    # Set parameter
    n = 9
    c = 7

    # Import shots
    read_path = f"../data/hardware_experiment/shots_n{n}_c{c}_rho{rho}.pkl"
    with open(read_path, "rb") as file:
        shots = pickle.load(file)

    n_post_select_qubits = 6
    n_state_register = 9

    mask = np.all(
        shots[:, :n_post_select_qubits] == np.zeros(n_post_select_qubits), axis=1
    )
    shots_postselect = shots[mask][:, n_post_select_qubits:]
    shots_postselect_0 = shots_postselect[:, :n_state_register]
    shots_postselect_1 = shots_postselect[:, n_state_register:]

    # Full density
    outcome_dict = generate_outcome_dict(
        n_qubits=len(shots_postselect[0]), shots_postselect=shots_postselect, dim=2
    )
    shot_distribution_full = ShotDistribution(
        shots_postselect, outcome_dict=outcome_dict
    )
    outcomes = np.array(shot_distribution_full.outcomes)

    # Marginal densities
    outcome_dict = generate_outcome_dict(
        n_qubits=len(shots_postselect_0[0]), shots_postselect=shots_postselect_0, dim=1
    )
    shot_distribution_x = ShotDistribution(
        shots_postselect_0, outcome_dict=outcome_dict
    )
    shot_distribution_y = ShotDistribution(
        shots_postselect_1, outcome_dict=outcome_dict
    )

    #### KDE cross validation sklearn
    bandwidths = 10 ** np.linspace(-3, 0, 50)
    outcomes_x = np.array(shot_distribution_x.outcomes)
    loo = LeaveOneOut()
    loo.get_n_splits(outcomes_x)

    grid_x = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=loo
    )
    _ = grid_x.fit(outcomes_x[:, None])

    outcomes_y = np.array(shot_distribution_y.outcomes)
    loo = LeaveOneOut()
    loo.get_n_splits(outcomes_y)

    grid_y = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=loo
    )
    _ = grid_y.fit(outcomes_y[:, None])

    #### KDE cross validation sklearn of 2D distribution
    outcomes_reduced = outcomes[:]
    loo = LeaveOneOut()
    loo.get_n_splits(outcomes_reduced)

    grid_full = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=loo
    )
    _ = grid_full.fit(outcomes_reduced)

    return grid_full, grid_x, grid_y, bandwidths


print("KDE for uncorrelated Gaussian")
grid_full_uncor, grid_x_uncor, grid_y_uncor, bandwidths = (
    generate_kde_cross_validation_data(rho=0.0)
)
print("KDE for correlated Gaussian")
grid_full_cor, grid_x_cor, grid_y_cor, bandwidths = generate_kde_cross_validation_data(
    rho=0.4
)


kde_cross_validation_dict = {
    "grid_full_uncor": grid_full_uncor,
    "grid_x_uncor": grid_x_uncor,
    "grid_y_uncor": grid_y_uncor,
    "grid_full_cor": grid_full_cor,
    "grid_x_cor": grid_x_cor,
    "grid_y_cor": grid_y_cor,
    "bandwidths": bandwidths,
}


n = 9
c = 7

folder_path = "../data/hardware_experiment/"
os.makedirs(folder_path, exist_ok=True)

print("Save data")
save_path = os.path.join(folder_path, f"kde_cross_validation_dict_n{n}_c{c}.pkl")
with open(save_path, "wb") as file:
    pickle.dump(kde_cross_validation_dict, file)
