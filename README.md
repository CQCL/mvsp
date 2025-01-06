# Quantum state preparation for multivariate functions

[mvsp](https://github.com/CQCL/mvsp) is a Python implementation of the protocols presented in [Quantum state preparation for multivariate functions](https://arxiv.org/abs/2405.21058). The protocols are based on function approximations with finite Fourier or Chebyshev series, efficient block encodings of Fourier and Chebyshev basis functions, and the linear combination of unitaries (LCU).

## Installation

To install the project, clone the repository and run:

```sh
python -m pip install --upgrade pip
python -m pip install uv
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Circuit Construction

All circuits are implemented with [TKET](https://github.com/CQCL/tket). The main code implementing the protocols is in `mvsp/circuits/lcu_state_preparation` with the following files:

- [lcu_state_preparation_block_encoding.py](mvsp/circuits/lcu_state_preparation/lcu_state_preparation_block_encoding.py): circuits to implement the block encodings for Fourier and Chebyshev basis functions.
- [lcu_state_preparation.py](mvsp/circuits/lcu_state_preparation/lcu_state_preparation.py): given the basis coefficients as input, implements the circuit for multivariate state preparation (see Figs. 4 and 5 from the paper).
  
The circuit construction is based on two elements, the `RegisterBox` and `QRegs`. `RegisterBox` contains the gates and operations and `QRegs` is the quantum register that keeps track which qubits the gates act on. The code can be found in `mvsp/circuits/core` and one can check the [RegisterBox example notebook](examples/circuits/intro_registerbox.ipynb) for further details.

The [LCU state preparation example notebook](examples/circuits/LCUStatePreparationBox_example.ipynb) describes how to use these circuit constructions for quantum state preparation.

In addition to the main code we also provide multiple primitives like qubitisation, select and prepare boxes, and measurement utility functions.

## Reproducing results from the paper

The repo contains all scripts and data for reproducing the figures in the paper. We also provide the code to generate the data making it straightforward to extend the results to other target functions. In addition, we include the exact circuits that were run on the Quantinuum H2-1 device for Fig. 10 in the paper.

### Basis functions (Fig. 6)

To reproduce Fig. 6 run

```sh
python scripts/plot_basis_functions.py
```

The code generates the block encodings for Fourier and Chebyshev basis functions using our protocols and plots them together with the numerical evaluation of the basis functions.

### 2D Ricker wavelet via Chebyshev series (Fig. 7)

To reproduce Fig. 7 run

```sh
python scripts/plot_chebyshev_resources.py
```

The code uses the data stored in [`data/Chebyshev_ricker2d_resource_scaling`](data/Chebyshev_ricker2d_resource_scaling).

All data can be generated with the following 3 scripts.

1. [scripts/max_errors.py](scripts/max_errors.py): numerically computes the uniform error of the Chebyshev approximation on $[-1,1]^2$. Run with the following arguments to reproduce the figure. For more options, run with `-h`.

    ```sh
    python scripts/max_errors.py -f ricker2d -t chebyshev -d `seq 63`
    ```

2. [scripts/run_chebyshev_resources.sh](scripts/run_chebyshev_resources.sh): calls [`scripts/compute_resources.py`](scripts/compute_resources.py) for varying polynomial degress and numbers of qubits. This computes the two-qubit gate counts and other resources compiled for the Quantinuum H2-1 native gate set. Compilation details can be adjusted via command line arguments, see the `python scripts/compute_resources.py -h` for details.
3. [scripts/run_chebyshev_simulations.sh](scripts/run_chebyshev_simulations.sh): works similarly but now also simulates the circuit outputs with the IBM Aer backend. This is separate from the previous script because the simulation takes longer so we generate the data for fewer degrees.

### Bivariate Student's t-distribution via Fourier series (Fig. 8)

To reproduce Fig. 8 run

```sh
python scripts/plot_fourier_resources.py
```

The code uses the data stored in [`data/Fourier_cauchy2d_resource_scaling`](data/Fourier_cauchy2d_resource_scaling).

The data generation is similar to the one described in section [2D Ricker wavelet via Chebyshev series (Fig. 7)](#2d-ricker-wavelet-via-chebyshev-series-fig-7).

Numerically compute the uniform error of the Fourier approximation of the bivariate Student's t-distribution over $[-1, 1]^2$:

```sh
python scripts/max_errors.py -f cauchy2d -t fourier -d `seq 63`
```

The shell scripts for generating resources and simulation data are [`scripts/run_fourier_resources.sh`](scripts/run_fourier_resources.sh) an [`scripts/run_fourier_simulations.sh`](scripts/run_fourier_simulations.sh).

### Single electron in periodic 3D Coulomb potential (Fig. 9)

To reproduce Fig. 9 run

```sh
python scripts/plot_chemistry_data.py
```

The code uses the data stored in [`data/electron_in_Coulomb_potential`](data/electron_in_Coulomb_potential). This data folder also contains .csv files with the number of qubits and 2-qubit gates used in Table 2 of the paper, extracted from the generated circuits.

The folder  [`mvsp/applications/chemistry`](mvsp/applications/chemistry) provides the methods to reproduce the circuits for the chemistry experiments. A single particle plane wavefunction was constructed using the Fourier state preparation using a nuclear lattice hamiltonian. Various lattices are provided in [`mvsp/applications/chemistry/lattices/lattice.py`](mvsp/applications/chemistry/lattices/lattice.py). We also provide a [tutorial on how to construct plane wave circuit from a lattice Hamiltonian](examples/circuit_plane_waves.ipynb).

Supplementary data and plots are also provided in the folder [`data/electron_in_Coulomb_potential`](data/electron_in_Coulomb_potential) and [`plots/electron_in_Coulomb_potential`](plots/electron_in_Coulomb_potential).

The data can be generated with

```sh
python scripts/compute_chemistry_data.py
```

Various parameters such as spatial grid, number of plane waves etc. can be adjusted in the file.


### Bivariate Gaussian on quantum hardware (Figs. 10-11)

To reproduce Fig. 10 run

```sh
python scripts/hardware_experiment_2d_gaussian_plot.py
```

The shot data from the hardware experiments are stored in `data/hardware_experiment/shots_n9_c7_rho0.*.pkl`. Given this shot data, the data for Fig. 10 (density estimates etc.) can be generated with

```sh
python scripts/hardware_experiment_2d_gaussian.py
```

This can take several minutes to run.

To reproduce Fig. 11 in the appendix (kernel density estimate cross validation) run

```sh
python scripts/kde_cross_validation_plot.py
```

The data can be generated with

```sh
python scripts/kde_cross_validation.py
```

This can take several minutes to run.

#### Circuits for hardware experiments

The folder [`data/hardware_experiments/circuits`](data/hardware_experiments/circuits) contains the circuits used for the hardware experiments performed on the H2-1 trapped-ion quantum computer. We have both the original and compiled circuits, along with iamges of the circuits. This supplementary data is not shown in the paper.

## Citation

If you use this code in your work, please consider citing our corresponding research paper.

```text
@misc{rosenkranzQuantumStatePreparation2024,
  title = {Quantum State Preparation for Multivariate Functions},
  author = {Rosenkranz, Matthias and Brunner, Eric and {Marin-Sanchez}, Gabriel and Fitzpatrick, Nathan and Dilkes, Silas and Tang, Yao and Kikuchi, Yuta and Benedetti, Marcello},
  year = {2024},
  number = {arXiv:2405.21058},
  publisher = {arXiv},
  eprint = {2405.21058},
  archiveprefix = {arXiv},
  primaryClass = {quant-ph}
}
```
