from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import moment as scipy_moment


class ShotDistribution:
    def __init__(
        self,
        shots: list[tuple] | NDArray,
        outcome_dict: dict,
    ):
        self.shots = shots
        self.outcome_dict = outcome_dict
        self._outcomes = None
        self._shot_density = None
        self._outcome_density = None

    @property
    def outcomes(self) -> list:
        if self._outcomes is None:
            self._outcomes = [self.outcome_dict[tuple(shot)] for shot in self.shots]
        return self._outcomes

    @property
    def shot_density(self):
        if self._shot_density is None:
            bit_tuples = list(product([0, 1], repeat=len(self.shots[0])))
            shots_tuples = list(map(tuple, self.shots))
            shot_density = {}
            for tup in bit_tuples:
                shot_density[tup] = shots_tuples.count(tup) / len(shots_tuples)
            self._shot_density = shot_density
        return self._shot_density

    @property
    def outcome_density(self):
        if self._outcome_density is None:
            shot_density = self.shot_density
            outcomes_ordered = list(self.outcome_dict.values())
            if not isinstance(outcomes_ordered[0], (int, float)):
                print("Transform outcomes to tuples.")
                outcomes_ordered = list(map(tuple, outcomes_ordered))
            self._outcome_density = dict(zip(outcomes_ordered, shot_density.values()))
        return self._outcome_density

    def filtered_outcome_density(self, sigma=1.0, dimension=1, shape=None):
        keys = list(self.outcome_density.keys())
        vals = list(self.outcome_density.values())
        if dimension == 1:
            filtered_vals = gaussian_filter(vals, sigma=sigma)
            res = {}
            res["outcomes"] = keys
            res["values"] = filtered_vals
        else:
            keys_reshape = np.array(keys).reshape(tuple(list(shape) + [dimension]))
            vals_reshape = np.array(vals).reshape(shape)
            filtered_vals_reshape = gaussian_filter(vals_reshape, sigma=sigma)
            res = {}
            res["outcomes"] = keys_reshape
            res["values"] = filtered_vals_reshape
        return res

    def statistical_moment(self, moment: int, center: float | None = None):
        return scipy_moment(
            self.outcomes,
            moment=moment,
            center=center,
        )


def generate_outcome_dict(n_qubits, shots_postselect, dim=1):
    if dim == 1:
        bit_tuples = np.array(list(product([0, 1], repeat=n_qubits)))
        n_all_outcome = 2**n_qubits

        x_min = 0
        x_max = 1
        outcomes = np.linspace(x_min, x_max, n_all_outcome)

        assert len(outcomes) == len(bit_tuples)
        outcome_dict = dict(zip(map(tuple, bit_tuples), outcomes))
    elif dim == 2:
        assert not n_qubits % 2, "n_qubits must be even for dim = 2."

        bit_tuples = np.array(list(product([0, 1], repeat=len(shots_postselect[0]))))
        n_all_outcome = 2 ** (n_qubits // 2)
        print(n_all_outcome)

        x_min = 0
        x_max = 1
        x_space = np.linspace(x_min, x_max, n_all_outcome)
        xx, yy = np.meshgrid(x_space, x_space)
        pos = np.stack((xx, yy), axis=-1).transpose((1, 0, 2))
        outcomes = pos.reshape((np.prod(pos.shape[:2]), 2))

        assert len(outcomes) == len(bit_tuples)
        outcome_dict = dict(zip(map(tuple, bit_tuples), outcomes))

    return outcome_dict
