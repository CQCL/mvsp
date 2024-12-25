"""Generate hardware experiment results from shot data."""

import os
import pickle

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

# from mvsp.mvsp_nexus.fourier import FourierExpansion
from mvsp.preprocessing.fourier import FourierExpansion
from mvsp.preprocessing.gauss import gauss_coefficient_function, gaussian_function
from mvsp.utils.discrete_density import DiscreteDensity
from mvsp.utils.shot_distribution import ShotDistribution, generate_outcome_dict


def generate_data(rho: float):
    # Set parameter
    n = 9
    c = 7
    mu_x = 0.5
    mu_y = 0.5
    s_x = 0.22
    s_y = 0.18

    # Import shots
    filename = os.path.abspath(f"data/hardware_experiment/shots_n{n}_c{c}_rho{rho}.pkl")
    with open(filename, "rb") as file:
        shots = pickle.load(file)

    n_post_select_qubits = 6
    n_state_register = 9

    mask = np.all(
        shots[:, :n_post_select_qubits] == np.zeros(n_post_select_qubits), axis=1
    )
    shots_postselect = shots[mask][:, n_post_select_qubits:]
    shots_postselect_0 = shots_postselect[:, :n_state_register]
    shots_postselect_1 = shots_postselect[:, n_state_register:]

    #### Ground truth
    d_X = c - 1
    d_Y = c - 1
    n_grid_X = n
    n_grid_Y = n
    n_grid = np.array([n_grid_X, n_grid_Y])
    mean = np.array([mu_x, mu_y])
    cov = np.array([[s_x**2, rho * s_x * s_y], [rho * s_x * s_y, s_y**2]])
    gauss_fun = gaussian_function(mean=mean, cov=cov)
    f_coeff = gauss_coefficient_function(mean=mean, cov=cov)
    fourier = FourierExpansion(
        func=gauss_fun,
        coefficient_function=f_coeff,
        degrees=[d_X, d_Y],
    )
    # fourier._initialize_eval()
    fourier._initialize_coeffs_dct()

    x = np.linspace(0.0, 1.0, 2 ** n_grid[0])
    y = np.linspace(0.0, 1.0, 2 ** n_grid[1])
    xx, yy = np.meshgrid(x, y)
    pos = np.stack((xx, yy), axis=-1)
    pos = pos.transpose((1, 0, 2))  # x varies along axis 0, i.e. axis 0 is the x-axis

    density_ground_truth_fourier = DiscreteDensity(
        fourier(pos), coordinates=[x, y], positivize=True
    )
    density_ground_truth_fourier_x = density_ground_truth_fourier.get_marginal(axes=[1])
    density_ground_truth_fourier_y = density_ground_truth_fourier.get_marginal(axes=[0])

    # Full density
    outcome_dict = generate_outcome_dict(
        n_qubits=len(shots_postselect[0]), shots_postselect=shots_postselect, dim=2
    )
    shot_distribution_full = ShotDistribution(
        shots_postselect, outcome_dict=outcome_dict
    )
    outcomes = np.array(shot_distribution_full.outcomes)
    x_vals = outcomes[:, 0]
    y_vals = outcomes[:, 1]

    # Marginal densities
    outcome_dict = generate_outcome_dict(
        n_qubits=len(shots_postselect_0[0]), shots_postselect=shots_postselect_0, dim=1
    )

    shot_distribution_x = ShotDistribution(
        shots_postselect_0, outcome_dict=outcome_dict
    )
    outcome_density_x = shot_distribution_x.outcome_density

    shot_distribution_y = ShotDistribution(
        shots_postselect_1, outcome_dict=outcome_dict
    )
    outcome_density_y = shot_distribution_y.outcome_density

    density_raw_x = DiscreteDensity(
        np.array(list(outcome_density_x.values())),
        coordinates=[np.array(list(outcome_density_x.keys()))],
        normalize=False,
    )
    density_raw_y = DiscreteDensity(
        np.array(list(outcome_density_y.values())),
        coordinates=[np.array(list(outcome_density_y.keys()))],
        normalize=False,
    )

    #### KDE cross validation sklearn
    bandwidths = 10 ** np.linspace(-3, 0, 50)
    outcomes_x = np.array(shot_distribution_x.outcomes)
    loo = LeaveOneOut()
    loo.get_n_splits(outcomes_x)
    grid_x = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=loo
    )
    kdr_estimate_x = grid_x.fit(outcomes_x[:, None])
    outcomes_y = np.array(shot_distribution_y.outcomes)
    loo = LeaveOneOut()
    loo.get_n_splits(outcomes_y)
    grid_y = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=loo
    )
    kdr_estimate_y = grid_y.fit(outcomes_y[:, None])
    kdr_estimate_x = grid_x.best_estimator_
    kdr_estimate_y = grid_y.best_estimator_

    x_d = np.linspace(0, 1, len(outcome_density_x.values()))
    y_d = np.linspace(0, 1, len(outcome_density_y.values()))
    probs_x = np.exp(kdr_estimate_x.score_samples(x_d[:, None]))
    probs_y = np.exp(kdr_estimate_y.score_samples(y_d[:, None]))

    density_kde_x = DiscreteDensity(np.sqrt(probs_x), coordinates=[x_d])
    density_kde_y = DiscreteDensity(np.sqrt(probs_y), coordinates=[y_d])
    density_raw_kde_x = DiscreteDensity(
        probs_x / np.sum(probs_x),
        coordinates=[x_d],
        normalize=False,
    )
    density_raw_kde_y = DiscreteDensity(
        probs_y / np.sum(probs_x),
        coordinates=[y_d],
        normalize=False,
    )

    #### KDE cross validation sklearn of 2D distribution
    bandwidths = 10 ** np.linspace(-3, 0, 50)
    outcomes_reduced = outcomes[:]
    loo = LeaveOneOut()
    loo.get_n_splits(outcomes_reduced)
    grid_full = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=loo
    )
    kdr_estimate_full = grid_full.fit(outcomes_reduced)
    kdr_estimate_full = grid_full.best_estimator_

    x_d = np.linspace(0, 1, len(outcome_density_x.values()))
    y_d = np.linspace(0, 1, len(outcome_density_y.values()))
    xx, yy = np.meshgrid(x_d, y_d)
    pos = np.stack([xx, yy], axis=-1).reshape((-1, 2))
    probs_full = np.exp(
        kdr_estimate_full.score_samples(pos).reshape((x_d.shape[0], y_d.shape[0]))
    )
    probs_full = np.sqrt(probs_full)
    probs_full = probs_full / np.sum(probs_full)

    x_d_coarse = np.linspace(0, 1, len(outcome_density_x.values()) // 4)
    y_d_coarse = np.linspace(0, 1, len(outcome_density_y.values()) // 4)
    xx, yy = np.meshgrid(x_d_coarse, y_d_coarse)
    pos = np.stack([xx, yy], axis=-1).reshape((-1, 2))
    probs_full_coarse = np.exp(
        kdr_estimate_full.score_samples(pos).reshape(
            (x_d_coarse.shape[0], y_d_coarse.shape[0])
        )
    )
    probs_full_coarse = np.sqrt(probs_full_coarse)
    probs_full_coarse = probs_full_coarse / np.sum(probs_full_coarse)
    density_kde_full_coarse = DiscreteDensity(
        probs_full_coarse.transpose(), coordinates=[x_d_coarse, y_d_coarse]
    )

    density_dict = {
        "density_kde_full_coarse": density_kde_full_coarse.data.transpose(),
        "density_kde_x": density_kde_x.data,
        "density_raw_x": density_raw_x.data,
        "density_raw_kde_x": density_raw_kde_x.data,
        "density_kde_y": density_kde_y.data,
        "density_raw_y": density_raw_y.data,
        "density_raw_kde_y": density_raw_kde_y.data,
        "density_ground_truth_fourier_x": density_ground_truth_fourier_x.data,
        "density_ground_truth_fourier_y": density_ground_truth_fourier_y.data,
        "density_x_coordinates": density_raw_x.coordinates[0],
        "density_y_coordinates": density_raw_y.coordinates[0],
        "density_full_coarse_coordinates": density_kde_full_coarse.coordinates,
        "outcomes_x": x_vals,
        "outcomes_y": y_vals,
    }

    folder_path = os.path.abspath("data/hardware_experiment/")
    os.makedirs(folder_path, exist_ok=True)

    print("Save data")
    save_path = os.path.join(folder_path, f"density_dict_n{n}_c{c}_rho{rho}.pkl")
    with open(save_path, "wb") as file:
        pickle.dump(density_dict, file)


# Hardware: N=9, c=7, rho=0.0
print("Generate and save data for N=9, c=7, rho=0.0")
rho = 0.0
generate_data(rho=rho)

# Hardware: N=9, c=7, rho=0.4
print("Generate and save data for N=9, c=7, rho=0.4")
rho = 0.4
generate_data(rho=rho)
