import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from numpy.polynomial.chebyshev import chebval
from mvsp.circuits.lcu_state_preparation.lcu_state_preparation_block_encoding import (
    ChebychevBlockEncoding,
    FourierBlockEncoding,
)

from mvsp.utils.paper_utils import get_projector_matrix3

plt.style.use(os.path.abspath("plots/paper.mplstyle"))

n = 5
num_functions = 5


def unit_vec(size: int, index: int):
    e = np.zeros(size)
    e[index] = 1.0
    return e


def evaluate_block_encoding(n: int, index: int = 1, series_type: str = "fourier"):
    if series_type == "fourier":
        W_box = FourierBlockEncoding(n).generate_basis_element(index)
        n_anc_qubits = 0
    elif series_type == "chebyshev":
        W_box = ChebychevBlockEncoding(n).generate_basis_element(index)
        n_anc_qubits = len(W_box.qreg.prepare)
    unitary = W_box.get_circuit().get_unitary()

    projector = np.array(
        get_projector_matrix3(
            list(range(n_anc_qubits)), list(range(n_anc_qubits, n_anc_qubits + n))
        )
    )
    unitary_projected = projector.transpose() @ unitary @ projector
    h = unitary_projected - np.diag(unitary_projected.diagonal())
    print(np.abs(h).max())
    return unitary_projected.diagonal()


interval_min = {"chebyshev": -1.0, "fourier": 0.0}

fig, axs = plt.subplots(
    num_functions,
    2,
    figsize=[3.22, 2.1],
    sharey=True,
    sharex="col",
    constrained_layout=True,
)
fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

for col, series_type in enumerate(["fourier", "chebyshev"]):
    for i in range(num_functions):
        ax = axs[i, col]

        evaluated_be = evaluate_block_encoding(n, i, series_type)

        x_grid = np.linspace(interval_min[series_type], 1, len(evaluated_be))
        xs = np.linspace(interval_min[series_type], 1, 300)
        ax.plot(x_grid, evaluated_be.real, ".", markersize=2)
        if series_type == "chebyshev":
            ax.plot(
                xs,
                chebval(xs, unit_vec(num_functions, i)),
                "-",
                color="C0",
                linewidth=0.7,
            )
        if series_type == "fourier":
            ax.plot(
                xs,
                np.cos(np.pi * i * xs),
                "-",
                label="Real",
                color="C0",
                linewidth=0.7,
            )
            ax.plot(
                xs,
                np.sin(np.pi * i * xs),
                "--",
                label="Imag.",
                color="C1",
                linewidth=0.7,
            )
            ax.plot(
                x_grid,
                evaluated_be.imag,
                ".",
                markersize=2,
            )

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        if col == 1:
            ax.annotate(
                f"$k={i}$",
                (-0.95, 0),
                (-0.95, 0),
                fontsize="small",
                va="center",
                ha="right",
            )
        if i == num_functions - 1:
            ax.set_xlabel("$x$")
        if i < num_functions - 1:
            ax.spines.bottom.set_visible(False)
            ax.tick_params(which="both", bottom=False)
        if col > 0:
            ax.spines.left.set_visible(False)
            ax.tick_params(which="both", left=False)


axs[0, 0].legend(loc="lower center", ncol=2, borderpad=0, borderaxespad=0)
fig.suptitle("Block encodings")
axs[0, 0].set_title(r"Fourier ($e^{i\pi k x}$)")
axs[0, 1].set_title(r"Chebyshev ($T_k(x)$)")
plt.figtext(0.0, 0.89, "(a)", fontsize="large")
plt.figtext(0.53, 0.89, "(b)", fontsize="large")

plot_file = os.path.abspath("plots/block_encodings.pdf")
fig.savefig(plot_file, transparent=False, bbox_inches="tight", pad_inches="layout")
