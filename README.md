# Quantum state preparation for multivariate functions

This is a Python implementation of the protocols presented in [Quantum state preparation for multivariate functions](https://arxiv.org/abs/2405.21058). They are based on function approximations with finite Fourier or Chebyshev series, efficient block encodings of Fourier and Chebyshev basis functions, and the linear combination of unitaries (LCU).

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

The repo contains all scripts and data for reproducing the figures in the paper. In addition we provide the code to generate the data making it straightforward to extend the results to other target functions. Moreover, we include as supplementary material the exact circuits that were run on the Quantinuum H2-1 device for Fig. 10 in the paper.

### Basis functions (Fig. 6)

The following snippet generates Fig. 6.

```sh
python scripts/plot_basis_functions.py
```

The code generates the block encodings for Fourier and Chebyshev basis functions using our protocols and plots them together with the numerical evaluation of the basis functions.

### 2D Ricker wavelet with Chebyshev approach (Fig. 7)

The following snippet generates Fig. 7.

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

### 2D Student's t-distribution with Fourier approach (Fig. 8)

The following snippet generates Fig. 8.

```sh
python scripts/plot_fourier_resources.py
```

The code uses the data stored in [`data/Fourier_cauchy2d_resource_scaling`](data/Fourier_cauchy2d_resource_scaling).

The data generation is similar to the one described in section [2D Ricker wavelet with Chebyshev approach (Fig. 7)](#2d-ricker-wavelet-with-chebyshev-approach-fig-7).

Numerically compute the uniform error of the Fourier approximation of the bivariate Student's t-distribution over $[-1, 1]^2$:

```sh
python scripts/max_errors.py -f cauchy2d -t fourier -d `seq 63`
```

The shell scripts for generating resources and simulation data are [`scripts/run_fourier_resources.sh`](scripts/run_fourier_resources.sh) an [`scripts/run_fourier_simulations.sh`](scripts/run_fourier_simulations.sh).

### Chemistry experiments (Fig. 9)

The folder  [`mvsp/applications/chemistry`](mvsp/applications/chemistry) provides the methods to reproduce the circuits for the chemistry experiments. A single particle plane wavefunction was constructed using the Fourier state preparation using a nuclear lattice hamiltonian. Various lattices are provided in [`mvsp/applications/chemistry/lattices/lattice.py`](mvsp/applications/chemistry/lattices/lattice.py). Files to reproduce the paper results are:

- [examples/circuit_plane_waves.ipynb](mvsp/applications/chemistry/examples/circuit_plane_waves.ipynb): tutorial on how to construct the plane wave circuit from a lattice Hamiltonian
- [plotting/chemistry_plotting_script.py](mvsp/applications/chemistry/plotting/chemistry_plotting_script.py): to reproduce the plots in the paper for the chemistry experiments. Data and plots are also provided in the [`mvsp/applications/chemistry/plotting/paper_data`](mvsp/applications/chemistry/plotting/paper_data) and [`mvsp/applications/chemistry/plotting/paper_plots_new`](mvsp/applications/chemistry/plotting/paper_plots_new) folders.

### Hardware experiments (Figs. 10-11)

The folder [`data/hardware_experiments/circuits`](data/hardware_experiments/circuits) contains the circuits considered for the hardware experiments performed on the H2-1 trapped-ion quantum computer. We have both the original and compiled circuits, along with iamges of the circuits.

The hardware experiment results shown in Fig. 10 of [Quantum state preparation for multivariate functions](https://arxiv.org/abs/2405.21058) are generated with the scripts [`scripts/hardware_experiment_2d_gaussian.py`](scripts/hardware_experiment_2d_gaussian.py) (extraction and processing of shot data, takes around 5 minutes to run) and [`scripts/hardware_experiment_2d_gaussian_plot.py`](scripts/hardware_experiment_2d_gaussian_plot.py) (plotting the extracted results). The plots resulting plots are saved in `plots/`. The shot data is contained in `data/hardware_experiment/shots_n9_c7_rho0.*.pkl`.

The results for the kernel density cross validation, shown in Fig. 11, are generated with [`scripts/kde_cross_validation.py`](scripts/kde_cross_validation.py) (takes around 5 minutes). Plots are generated with [`scripts/kde_cross_validation_plot.py`](scripts/kde_cross_validation_plot.py).

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
