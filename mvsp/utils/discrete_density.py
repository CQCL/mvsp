"""Implementation of the discrete density module."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal


def gaussian_function(mean=None, cov=None):
    rv = multivariate_normal(mean=mean, cov=cov)
    return rv.pdf


class DiscreteDensity:
    def __init__(
        self,
        discrete_positive_function: NDArray,
        coordinates: list[NDArray] | None = None,
        dx: list[float] | None = None,
        domain: list[tuple[float, float]] | None = None,
        normalize: bool = True,
        positivize: str | bool = False,
    ):
        if dx is not None:
            assert len(dx) == len(discrete_positive_function.shape)
            if domain is None:
                raise Exception("If dx is defined you also need to define domain.")
        elif coordinates is not None:
            assert len(coordinates) == len(discrete_positive_function.shape)
        else:
            raise Exception("Either coordinates or dx must be defined.")

        self.coordinates = coordinates
        self.dx = dx
        self.domain = domain
        self.normalize = normalize
        self.positivize = positivize
        if positivize is True:
            self.positivize = "absolute"
        if self.positivize == "absolute":
            self.unnormalized_density = np.abs(discrete_positive_function)
        elif self.positivize == "absolute square":
            self.unnormalized_density = np.abs(discrete_positive_function) ** 2
        else:
            self.unnormalized_density = discrete_positive_function

        self.normalization = self._get_normalization_constant()
        if normalize:
            self.data = self.unnormalized_density / self.normalization
        else:
            self.data = self.unnormalized_density
        self.dim = len(self.unnormalized_density.shape)

    def _get_normalization_constant(self):
        integrator = self.unnormalized_density
        for i in range(len(self.unnormalized_density.shape) - 1, -1, -1):
            if self.dx is not None:
                integrator = np.trapz(integrator, dx=self.dx[i], axis=i)
            else:
                integrator = np.trapz(integrator, x=self.coordinates[i], axis=i)
        return integrator

    def get_marginal(self, axes: tuple, normalize: bool | None = None):
        marginal_discrete_density = self.data.copy()

        if normalize is None:
            normalize = self.normalize

        for i in range(len(axes) - 1, -1, -1):
            axis = axes[i]
            if self.dx is not None:
                marginal_discrete_density = np.trapz(
                    marginal_discrete_density, dx=self.dx[axis], axis=axis
                )
            else:
                marginal_discrete_density = np.trapz(
                    marginal_discrete_density, x=self.coordinates[axis], axis=axis
                )

        new_axes = list(set(range(len(self.data.shape))) - set(axes))
        if self.dx is None:
            new_dx = None
            new_domain = None
        else:
            new_dx = [self.dx[axis] for axis in new_axes]
            new_domain = [self.domain[axis] for axis in new_axes]
        if self.coordinates is None:
            new_coordinates = None
        else:
            new_coordinates = [self.coordinates[axis] for axis in new_axes]
        return DiscreteDensity(
            marginal_discrete_density,
            coordinates=new_coordinates,
            dx=new_dx,
            domain=new_domain,
            normalize=normalize,
        )

    def integrate_discrete_function(self, discrete_function):
        assert discrete_function.shape == self.data.shape
        integrator = discrete_function * self.data
        for i in range(len(discrete_function.shape) - 1, -1, -1):
            if self.dx is not None:
                integrator = np.trapz(integrator, dx=self.dx[i], axis=i)
            else:
                integrator = np.trapz(integrator, x=self.coordinates[i], axis=i)

        return integrator

    def _single_axis_meshgrid(self, axis):
        assert (
            len(self.data.shape) < 53
        ), "Dimension of density must be smaller than 53."

        from string import ascii_lowercase, ascii_uppercase

        alphabet = list(ascii_lowercase) + list(ascii_uppercase)

        other_axes = list(set(range(len(self.data.shape))) - set([axis]))

        einsum_input_string = (
            "".join([alphabet[i] for i in other_axes]) + f",{alphabet[axis]}"
        )
        einsum_output_string = "".join(
            [alphabet[i] for i in range(len(self.data.shape))]
        )
        einsum_string = einsum_input_string + "->" + einsum_output_string

        if self.dx is not None:
            array_list = [np.ones(np.array(self.data.shape)[other_axes])] + [
                np.linspace(
                    self.domain[axis][0],
                    self.domain[axis][1],
                    round((self.domain[axis][1] - self.domain[axis][0]) / self.dx[axis])
                    + 1,
                )
            ]
        else:
            array_list = [np.ones(np.array(self.data.shape)[other_axes])] + [
                self.coordinates[axis]
            ]

        return np.einsum(einsum_string, *array_list)

    def mean_value_of_axis(self, axis):
        integrator = self._single_axis_meshgrid(axis=axis)

        return self.integrate_discrete_function(integrator)

    def mean_value(self):
        return [self.mean_value_of_axis(axis) for axis in range(len(self.data.shape))]

    def covariance_matrix(self):
        mean = self.mean_value()

        X = [
            self._single_axis_meshgrid(axis=axis)
            for axis in range(len(self.data.shape))
        ]

        integrator = [
            [(X[i] - mean[i]) * (X[j] - mean[j]) for j in range(len(mean))]
            for i in range(len(mean))
        ]

        covariance_matrix = np.array(
            [
                [
                    self.integrate_discrete_function(integrator[i][j])
                    for j in range(len(mean))
                ]
                for i in range(len(mean))
            ]
        )

        return covariance_matrix

    def correlation_matrix(self):
        covariance_matrix = self.covariance_matrix()
        standard_deviations = np.sqrt(covariance_matrix.diagonal())
        std_outer = np.outer(standard_deviations, standard_deviations)
        return covariance_matrix / std_outer
