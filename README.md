# mvsp

This is a Python ```3.11```, ```3.12``` app called ```mvsp``` based on ```tket``` that implements the protocol presented in [Quantum state preparation for multivariate functions](https://arxiv.org/abs/2405.21058).

## Installation

To install the project, clone the repository and run:

```sh
python -m pip install --upgrade pip
python -m pip install uv
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip install pre-commit
pre-commit run --all-files
```

## Circuit Construction

The circuit construction is based on two elements, the ```RegisterBox``` and ```QRegs```. The first one contains the gates and operations and the second one is the quantum register that keeps track which qubit the gates act on. The code can be found in ```mvsp/circuits/core``` and one can check the [example notebook](https://github.com/CQCL/mvsp/blob/main/examples/circuits/intro_registerbox.ipynb) for further details.

The main code for this project is found in ```mvsp/circuits/lcu_state_preparation``` with the following files:

- **lcu_state_preparation_block_encoding.py**: contains the circuits to implement the block encodings for both Fourier and Chebyshev basis functions.
- **lcu_state_preparation.py**: given the basis coefficients, implements the circuit for multivariate state preparation (see Figs. 4 and 5 from the paper).

See the mvsp [example notebook](https://github.com/CQCL/mvsp/blob/main/examples/circuits/LCUStatePreparationBox_example.ipynb) to see how to use these boxes.

In addition to the main code we also provide multiple primitives like qubitisation, select and prepare boxes, and measurement utility functions.

## Hardware experiments
In ```experiments/hardware``` we have the circuits considered for the hardware experiments performed on the H2-1 trapped-ion quantum compute. We have both the original and compiled circuits, along with their images.

## Chemistry experiments
In ```mvsp/applications/chemistry``` we have the circuits considered for the chemistry experiments. Where a single particle plane wavefunction was constructed using the Fourier state preparation using a nuclear lattice hamiltonian. Various lattices are provided in ```/lattices/lattice.py```. Files to reproduce the paper results are:
- **/examples/circuit_plane_waves.ipynb**: [Tutorial on how to construct the plane wave circuit from a lattice hamiltonian](https://github.com/CQCL/mvsp/blob/main/mvsp/applications/chemistry/examples/circuit_plane_waves.ipynb).
- **/plotting/chemistry_plotting_script.py**: [To reproduce the plots in the paper for the chemistry experiments.](https://github.com/CQCL/mvsp/blob/main/mvsp/applications/chemistry/plotting/chemistry_plotting_script.py) Data and plots are also provided in the ```paper_data``` and ```paper_plots_new``` folders. 