import json
import os
import pickle
import re
import unicodedata

import numpy as np
import numpy.typing as npt
import scipy
from pytket.extensions.qiskit import AerStateBackend
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation import (
    LCUStatePreparationBox,
)
from scipy.integrate import dblquad

NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int_]


def get_projector_csr_matrix(
    projection_qubits: list[int],
    identity_qubits: list[int],
    proj_bit_string: str | None = None,
) -> NDArrayFloat:
    """Projector onto subspace of qubits.

    Generalises get_projection_matrix.
    TODO remove get_projection_matrix by this version

    Generate a projection matrix useful to validate LCU and QSP.
    In comparison to get_projector_matrix, this function takes as input
    a list of qubit which are not changed and a list of qubits which are
    projected onto the a given bit string. If no bit string is provided,
    we project onto the |0> state. The proj_bit_string is ordered according
    to the index in projection_qubit and not the actual qubit.

    Args:
    ----
        projection_qubits (list[int]): List of qubit indices which are projected out.
        identity_qubits (list[int]): List of qubit indices that remain after projection.
        proj_bit_string (str | None, optional): String specifying a specific state onto
            which it should be projected. Defaults to None, in which case the
            proj_bit_string is set to "0...0".

    Returns:
    -------
        np.ndarray: Matrix representation of projector.

    """
    all_qubits = list(set(projection_qubits + identity_qubits))
    assert min(all_qubits) == 0
    assert max(all_qubits) == len(projection_qubits) + len(identity_qubits) - 1
    assert len(all_qubits) == len(projection_qubits) + len(identity_qubits)
    if proj_bit_string is None:
        proj_bit_string = len(projection_qubits) * "0"
    else:
        assert len(proj_bit_string) == len(projection_qubits)

    n_all_qubits = len(all_qubits)
    n_qubits_to_keep = len(identity_qubits)
    idx_to_keep = []
    # Assume certain qubit ordering:
    # AAA'BB
    # where AAA is the projection bitstring (here 3 qubits) and BB are qubits to
    # keep (here 2 qubits). The following line bit shifts to the left the projection
    # bitstring by the number of qubits to keep. The subsequent loop
    # applies binary or to set all combinations of lower-order bitstings for the
    # qubits to keep, i.e. AAA is fixed, BB loops through all possible length-2
    # bitstrings.
    # This could be generalised to nonconsecutive projection qubits by
    # fixing bits at the position of each projected qubit.
    proj_bit_mask = int(proj_bit_string, 2) << n_qubits_to_keep
    for i in range(2**n_qubits_to_keep):
        idx = i | proj_bit_mask
        idx_to_keep.append(idx)

    return scipy.sparse.csr_matrix(
        ([1.0] * len(idx_to_keep), (idx_to_keep, range(len(idx_to_keep)))),
        shape=(2**n_all_qubits, 2**n_qubits_to_keep),
    )


def get_projector_matrix3(
    projection_qubits: list[int],
    identity_qubits: list[int],
    proj_bit_string: str | None = None,
) -> np.ndarray:
    """Projector onto subspace of qubits.

    Generalises get_projection_matrix.
    TODO remove get_projection_matrix by this version

    Generate a projection matrix useful to validate LCU and QSP.
    In comparison to get_projector_matrix, this function takes as input
    a list of qubit which are not changed and a list of qubits which are
    projected onto the a given bit string. If no bit string is provided,
    we project onto the |0> state. The proj_bit_string is ordered according
    to the index in projection_qubit and not the actual qubit.

    Args:
    ----
        projection_qubits (list[int]): List of qubit indices which are projected out.
        identity_qubits (list[int]): List of qubit indices that remain after projection.
        proj_bit_string (str | None, optional): String specifying a specific state onto
            which it should be projected. Defaults to None, in which case the
            proj_bit_string is set to "0...0".

    Returns:
    -------
        np.ndarray: Matrix representation of projector.

    """
    all_qubits = list(set(projection_qubits + identity_qubits))
    assert min(all_qubits) == 0
    assert max(all_qubits) == len(projection_qubits) + len(identity_qubits) - 1
    assert len(all_qubits) == len(projection_qubits) + len(identity_qubits)
    if proj_bit_string is None:
        proj_bit_string = len(projection_qubits) * "0"
    else:
        assert len(proj_bit_string) == len(projection_qubits)

    bra = {
        "0": np.matrix([[1], [0]]),
        "1": np.matrix([[0], [1]]),
    }
    id_mat = np.identity(2)

    switcher = {}
    for idx in range(len(projection_qubits)):
        switcher[projection_qubits[idx]] = bra[proj_bit_string[idx]]
    for qb in identity_qubits:
        switcher[qb] = id_mat

    projector = 1.0
    for idx in range(len(all_qubits)):
        # todo: make this more efficient using sparse kronecker product
        projector = np.kron(projector, switcher[idx])
    return projector


class EvaluateLCU:
    def __init__(
        self,
        lcu_state_prep: LCUStatePreparationBox,
        backend,
        compile_only: bool | None = False,
        compiler_options: dict | None = None,
    ):
        if compiler_options is None:
            compiler_options = {"optimisation_level": 0}

        self.backend = backend
        self.lcu_state_prep = lcu_state_prep

        self.circ_compiled = self.backend.get_compiled_circuit(
            self.lcu_state_prep.get_circuit(), **compiler_options
        )

        self.n_qubits_main_register = sum(self.lcu_state_prep.n_qubits_state)
        self.n_qubits_coeffs = sum(self.lcu_state_prep.n_qubits_coeffs)
        self.n_qubits_block_encoding = sum(self.lcu_state_prep.n_qubits_block_encoding)

        self._projector = None
        self._state_vector = None
        self._projected_state_vector = None
        self._success_probability = None
        self._projected_state_vector_scaled = None
        self.compile_only = compile_only

        if not self.compile_only:
            qubit_ordering = [
                y
                for x in self.lcu_state_prep.qreg.coeffs
                + self.lcu_state_prep.qreg.block
                + self.lcu_state_prep.qreg.state
                for y in list(x)
            ]
            statevector_backend = AerStateBackend()
            statevec_simulation_circ = statevector_backend.get_compiled_circuit(
                self.lcu_state_prep.get_circuit(),
                optimisation_level=0,
            )
            self._state_vector = statevector_backend.run_circuit(
                statevec_simulation_circ
            ).get_state(qubit_ordering)
            # TODO: replace this by shot sampling of probabilities using other backends

            self._projector = get_projector_csr_matrix(
                list(range(self.n_qubits_coeffs + self.n_qubits_block_encoding)),
                list(
                    range(
                        self.n_qubits_coeffs + self.n_qubits_block_encoding,
                        self.n_qubits_coeffs
                        + self.n_qubits_block_encoding
                        + self.n_qubits_main_register,
                    )
                ),
            )

            self._projected_state_vector = (
                self.projector.transpose() @ self.state_vector
            )
            self._success_probability = np.linalg.norm(self.projected_state_vector) ** 2
            self._projected_state_vector_scaled = self.projected_state_vector / np.sqrt(
                self.success_probability
            )

    @property
    def n_2qb_gates(self):
        return self.circ_compiled.n_2qb_gates()

    @property
    def depth(self):
        return self.circ_compiled.depth()

    @property
    def n_1qb_gates(self):
        return self.circ_compiled.n_1qb_gates()

    @property
    def n_qubits(self):
        return self.circ_compiled.n_qubits

    @property
    def success_probability(self):
        return self._success_probability

    @property
    def projector(self):
        return self._projector

    @property
    def state_vector(self):
        return self._state_vector

    @property
    def projected_state_vector(self):
        return self._projected_state_vector

    @property
    def projected_state_vector_scaled(self):
        return self._projected_state_vector_scaled

    def success_probability_analytical(self, series_eval, series_coeffs):
        # Should pass the Chebyshev object but 1D and 2D cases are evaluated
        # slightly differently so have to pass the evaluated array
        success_probability = np.sum(np.abs(series_eval) ** 2) / (
            2**self.n_qubits_main_register * np.sum(np.abs(series_coeffs)) ** 2
        )
        if self.success_probability is None:
            self._success_probability = success_probability

        return success_probability

    def grid_points(
        self, interval_min: list | None = None, interval_max: list | None = None
    ):
        if interval_min is None:
            interval_min = [
                -1.0 for _ in range(len(self.lcu_state_prep.n_qubits_state))
            ]
        if interval_max is None:
            interval_max = [1.0 for _ in range(len(self.lcu_state_prep.n_qubits_state))]
        grid_sizes = 2 ** np.array(self.lcu_state_prep.n_qubits_state)
        grids = []
        for i in range(len(grid_sizes)):
            grid_size = grid_sizes[i]
            n_qubits = int(np.ceil(np.log2(grid_size)))
            grid = [
                interval_min[i]
                + (interval_max[i] - interval_min[i]) * j / (2**n_qubits - 1)
                for j in range(2**n_qubits)
            ]
            grids.append(grid)

        return np.meshgrid(*grids)

    def __call__(self) -> np.ndarray:
        if not self.compile_only:
            # n_qubits_state is a list of length given my the number of dimensions
            grid_sizes = 2 ** np.array(self.lcu_state_prep.n_qubits_state)
            res = self.projected_state_vector_scaled.reshape(grid_sizes)
            res = res.transpose()
            return res
        else:
            return np.array(None)


def slugify(value, allow_unicode=False):
    """Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s\.-]", "", value.lower())
    return re.sub(r"[-\s]+", "_", value).strip("-_")


def make_suffix(*values):
    return "_".join(str(v) for v in values).rstrip("_")


def load_pickle(prefix, func, n, d, vendor, device_name, optimisation_level, path):
    path = os.path.abspath(path)
    filename = (
        f"{prefix}_{func}_{n}_{d}_{vendor}_{device_name}_{optimisation_level}.pickle"
    )
    input_file = os.path.join(path, filename)
    with open(input_file, "rb") as f:
        resources = pickle.load(f)

    return resources


def load_json(basename, path):
    path = os.path.abspath(path)
    filename = f"{basename}.json"
    input_file = os.path.join(path, filename)
    with open(input_file) as f:
        resources = json.load(f)

    return resources


def load_resource_file(func, n, d, vendor, device_name, optimisation_level, path):
    return load_pickle("RS", func, n, d, vendor, device_name, optimisation_level, path)


def load_simulation_file(func, n, d, vendor, device_name, optimisation_level, path):
    return load_pickle("sim", func, n, d, vendor, device_name, optimisation_level, path)


def merge_files_along_degree(
    prefix,
    func,
    series_type,
    n,
    ds,
    vendor,
    device_name,
    optimisation_level,
    compile_only,
    func_args,
    path,
):
    resources = []
    for i, d in enumerate(ds):
        id_suffix = make_suffix(
            prefix,
            func,
            series_type,
            n,
            d,
            vendor,
            device_name,
            optimisation_level,
            compile_only,
            slugify(func_args),
        )

        try:
            res = load_json(id_suffix, path)
        except FileNotFoundError:
            if i > 0:
                res = {k: np.nan for k, _ in resources[-1].items()}
            else:
                raise FileNotFoundError
        resources.append(res)

    return [[i[key] for i in resources] for key in resources[0]]


def resources_along_degree(
    func,
    series_type,
    n,
    ds,
    vendor,
    device_name,
    optimisation_level,
    compile_only,
    func_args,
    path,
):
    if func_args is None:
        func_args = {}
    return merge_files_along_degree(
        "RS",
        func,
        series_type,
        n,
        ds,
        vendor,
        device_name,
        optimisation_level,
        compile_only,
        func_args,
        path,
    )


def load_max_errors(
    func,
    series_type,
    path,
):
    suffix = make_suffix("max_error", func, series_type)
    path = os.path.abspath(path)
    filename = suffix + ".json"
    input_file = os.path.join(path, filename)
    with open(input_file) as f:
        max_errors = json.load(f)

    return max_errors


def load_data(
    n_discretization,
    degrees,
    keys,
    data_path,
    str_rho=0.0,
    func_name="cauchy2d",
    vendor="quantinuum",
    device_name="H2-1E",
    optimisation_level=2,
):
    data = {}
    for key in keys:
        data[key] = {}
    for n in n_discretization:
        for key in keys:
            data[key][str(n)] = []

        for degree in degrees:
            filename = f"RS_{func_name}_rho_{str_rho}_{n}_{degree}_{vendor}_{device_name}_{optimisation_level}.pickle"
            input_file = os.path.join(data_path, filename)
            with open(input_file, "rb") as f:
                resources = pickle.load(f)
                for key in keys:
                    data[key][str(n)].append(resources[key])

    return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def asymptotic_p_success(func, series, bounds, series_kwargs=None, logger=None):
    if series_kwargs is None:
        series_kwargs = {"degree": 63}
    if len(bounds) < 4:
        raise ValueError(f"Need 4 boundary values (was: {bounds})")

    length_x = bounds[1] - bounds[0]
    length_y = bounds[3] - bounds[2]
    approx = series(func, **series_kwargs)
    f_quad = dblquad(lambda x, y: np.abs(func(x, y)) ** 2, *bounds)
    coeff_norm = np.sum(np.abs(approx.coeffs)) ** 2
    if logger is not None:
        logger.debug(f"    {f_quad=}, {coeff_norm=}")
    return f_quad[0] / (coeff_norm * length_x * length_y)
