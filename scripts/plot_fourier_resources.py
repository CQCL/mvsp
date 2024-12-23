import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

from mvsp.preprocessing.fourier import Fourier2D
from mvsp.utils.plot_utils import adjust_lightness, plot_colorbar
from mvsp.target_functions import cauchy2d
from mvsp.utils.paper_utils import (
    asymptotic_p_success,
    load_max_errors,
    resources_along_degree,
)

plt.style.use("plots/paper.mplstyle")

ds = np.array([1, 3, 7, 15, 31])
ns = np.arange(4, 7, dtype=int)

RS = {}
for n in ns:
    RS[n] = resources_along_degree(
        "cauchy2d",
        "fourier",
        [n, n],
        ds,
        "quantinuum",
        "H2-1E",
        optimisation_level=2,
        compile_only=True,
        func_args=None,
        path="data/Fourier_cauchy2d_resource_scaling",
    )

sim = {}
for n in ns:
    sim[n] = resources_along_degree(
        "cauchy2d",
        "fourier",
        [n, n],
        ds,
        "ibm",
        "Aer",
        optimisation_level=0,
        compile_only=False,
        func_args=None,
        path="data/Fourier_cauchy2d_resource_scaling",
    )
max_errors = load_max_errors(
    "cauchy2d", "fourier", path="data/Fourier_cauchy2d_resource_scaling"
)


def scaling(a, d, n, dim):
    """Worst-case scaling of the Fourier approach with free parameters `a`

    Args:
    ----
        a (_type_): _description_
        d (_type_): _description_
        n (_type_): _description_
        dim (_type_): _description_

    Returns:
    -------
        _type_: _description_

    """
    # dim could be inside the ceil but implementation doesn't work that way
    return (
        a[0]
        + a[1] * dim * n * np.ceil(np.log2(2 * d + 1))
        + a[2] * 2 ** (dim * np.ceil(np.log2(2 * d + 1)))
    )


def scaling_minimizer(a, d, n2qb, n, dim):
    """Cost function for least square fit.

    Args:
    ----
        a (_type_): _description_
        d (_type_): _description_
        n2qb (_type_): _description_
        n (_type_): _description_
        dim (_type_): _description_

    Returns:
    -------
        _type_: _description_

    """
    return scaling(a, d, n, dim) - n2qb


def max_err_scaling(a, d):
    return a[0] / d


def max_err_scaling_minimizer(a, d, c):
    return max_err_scaling(a, d) - c


fig = plt.figure(figsize=[6.72, 3.52], constrained_layout=True)
gs0 = fig.add_gridspec(2, 2)
gs1 = gs0[0, 0].subgridspec(1, 2)
# Dummy axes for title across two subplots
ax_left = fig.add_subplot(gs0[0, 0])
ax_left.axis("off")

ax_func = fig.add_subplot(gs1[0, 0])
ax_diff = fig.add_subplot(gs1[0, 1])
ax_max_err = fig.add_subplot(gs0[0, 1])
ax_n2qb = fig.add_subplot(gs0[1, 0])
ax_succ = fig.add_subplot(gs0[1, 1])

labels = [f"$n = {n}$" for n in ns]

for i, (n, label) in enumerate(zip(ns, labels, strict=False)):
    # Resources from compilation only, no max_errors_*
    (
        n_1qbs,
        n_2qbs,
        n_qubits,
        success_probabilities,
        max_errors_f,
        max_errors_approx,
        eval_real,
        eval_imag,
    ) = RS[n]

    # Fit 2q gate count to worst-case scaling
    # Exclude nan from the fit
    idx = np.isfinite(ds) & np.isfinite(np.array(n_2qbs).flatten())
    x0 = [0.0, 10.0, 3.0]
    res_lsq = least_squares(
        scaling_minimizer,
        x0,
        args=(
            ds[idx],
            np.array(n_2qbs).flatten()[idx],
            np.ones_like(ds[idx]) * n,
            np.ones_like(ds[idx]) * 2,
        ),
    )
    print(
        f"Fit {n=}: Success={res_lsq.success}. a0={res_lsq.x[0]:4.2f}, a1={res_lsq.x[1]:4.2f}, a2={res_lsq.x[2]:3.2f})"
    )

    # Load max error function and compute corresponding normalisation constant
    (_, _, _, _, max_errors_f, _, _, _) = sim[n]
    xs = np.linspace(0, 1, 2**n)
    xx, yy = np.meshgrid(xs, xs)
    f_eval = np.array([[cauchy2d(x, y) for x in xx[0]] for y in yy[:, 0]])
    f_norm = np.sqrt(np.sum(np.abs(f_eval) ** 2))

    ax_n2qb.loglog(
        ds,
        n_2qbs,
        "o",
        markerfacecolor=adjust_lightness(f"C{i}", 1.3),
        markeredgecolor=f"C{i}",
        markeredgewidth=1.0,
    )
    ds_ext = np.arange(1, 64)
    ax_n2qb.loglog(
        ds_ext,
        [scaling(res_lsq.x, d, n, 2) for d in ds_ext],
        linewidth=0.0,
        color=f"C{i}",
        marker="_",
        markersize=3,
    )

    ax_max_err.loglog(
        ds,
        # Multiply by norm because numerics was saved after normalisation
        np.array(max_errors_f) * f_norm,
        "o",
        markerfacecolor=adjust_lightness(f"C{i}", 1.5),
        markeredgecolor=f"C{i}",
        markeredgewidth=1.0,
        label=label,
    )
    ax_succ.loglog(
        ds,
        success_probabilities,
        "o",
        linewidth=0.9,
        markerfacecolor=adjust_lightness(f"C{i}", 1.5),
        markeredgecolor=f"C{i}",
        markeredgewidth=1.0,
    )


# Compute & plot the asymptotic p_success. We assume that the coefficients after
# degree 63 are sufficiently small to not contribute significantly to p_success.
# We checked by computing p_success with degree 128, which is visually
# indistinguishable from p_success at degree 63.
asymp_p_success = asymptotic_p_success(
    cauchy2d, Fourier2D, (0, 1, 0, 1), series_kwargs={"degree": 63}
)
print(f"Asympt. p_success={asymp_p_success}.")
ax_succ.axhline(
    asymp_p_success,
    linestyle="-",
    color="grey",
    linewidth=0.8,
    zorder=-100,
)

ax_max_err.loglog(
    max_errors["ds"],
    max_errors["max_errors"],
    "-",
    color="grey",
    linewidth=0.8,
    zorder=-100,
)
# Least square fit to the asymptotic maximum error
# x0 = [1.0]
ds_err = np.array(max_errors["ds"])
idx = ds_err >= 10
# res_lsq = least_squares(
#     max_err_scaling_minimizer,
#     x0,
#     args=(ds_err[idx], np.array(max_errors["max_errors"])[idx]),
# )
# print(f"Fit max error: Success={res_lsq.success}. a0={res_lsq.x[0]:4.1f}")
# Plots asymptotic scaling of the max error (we show 0.5/d)=O(1/d))
ax_max_err.loglog(
    ds_err[idx],
    # [max_err_scaling(res_lsq.x, d) for d in ds_err[idx]],
    [max_err_scaling(np.array([0.5]), d) for d in ds_err[idx]],
    "--",
    color="C3",
    linewidth=0.8,
    zorder=-100,
)

# Extract n=4, d=7 from circuit simulation data for plotting
n = 4
(_, _, _, _, _, _, eval_real, eval_imag) = sim[n]
# d=7 is index 2 in `ds`
eval_real = np.array(eval_real[2])
eval_imag = np.array(eval_imag[2])
xs = np.linspace(0, 1, 2**n)
xx, yy = np.meshgrid(xs, xs)
f_eval = np.array([[cauchy2d(x, y) for x in xx[0]] for y in yy[:, 0]])
f_norm = np.sqrt(np.sum(np.abs(f_eval) ** 2))


cb_func = ax_func.pcolormesh(xx, yy, f_eval, cmap="Reds")
cb_diff = ax_diff.pcolormesh(
    xx, yy, np.abs(f_eval - (eval_real + 1j * eval_imag) * f_norm), cmap="Reds"
)
plot_colorbar(cb_func, fig, ax_func)
plot_colorbar(cb_diff, fig, ax_diff)


ax_n2qb.annotate(
    "Fit to worst-case scaling\n"
    + r"$\alpha_0 + \alpha_1 \cdot D \cdot n \lceil \log_2(2d+1) \rceil + \alpha_2\cdot 2^{D\lceil \log_2(2d+1) \rceil}$",
    (2, 200),
    (2, 70),
    fontsize="small",
    va="top",
    ha="left",
    arrowprops=dict(arrowstyle="-", linewidth=0.5, relpos=(0, 0.5)),
)
ax_max_err.annotate(
    r"Upper bound $\max_{(x,y)\in [0,1]^2} |f(x,y)-f_d(x,y)|$",
    (22, 1.5e-2),
    (17, 1.2e-2),
    fontsize="small",
    va="top",
    ha="right",
    arrowprops=dict(arrowstyle="-", linewidth=0.5, relpos=(1, 0.5)),
)
ax_max_err.annotate(
    r"$\mathcal{O}(1/d)$",
    (27, 2e-2),
    (38, 2e-2),
    fontsize="small",
    va="center",
    ha="left",
    arrowprops=dict(arrowstyle="-", linewidth=0.5, relpos=(1, 0.5)),
)
ax_succ.annotate(
    r"Asymptotic $p_{\text{success}}^*=" + f"{asymp_p_success*100:.2f}" + r"\%$",
    (33, asymp_p_success + 4e-3),
    (33, asymp_p_success + 4e-3),
    fontsize="small",
    va="bottom",
    ha="right",
)


# Added spaces so title moves to left
ax_left.set_title(
    r"State prep. Fourier approach (Grid size $2^4 \times 2^4$)       ",
    pad=0,
    x=0.52,
    y=0.95,
)
ax_func.set_aspect("equal")
ax_func.set_xlabel("x")
ax_func.set_ylabel("y")
ax_func.set_title(r"Target $f$", y=1.01)
ax_diff.set_aspect("equal")
ax_diff.set_xlabel("x")
ax_diff.set_title(r"Error $|f - \tilde g_{d=7}|$", y=1.01)
ax_n2qb.set_ylabel(r"Number of two-qubit gates")
ax_n2qb.set_xlabel(r"Degree $d$")
ax_n2qb.set_ylim(10, 6e4)
ax_max_err.legend(title=r"Data (Grid size $2^n \times 2^n$)", alignment="left")
ax_max_err.set_ylabel(r"$\mathrm{max}_{\mathrm{grid}} |f - \tilde g_d|$")
ax_max_err.set_xlabel(r"Degree $d$")
ax_succ.set_ylim(5.5e-2, 1.1)
ax_succ.set_ylabel(r"$p_\mathrm{success}$")
ax_succ.set_xlabel(r"Degree $d$")

plt.figtext(0.02, 0.96, "(a)", fontsize="large")
plt.figtext(0.51, 0.96, "(b)", fontsize="large")
plt.figtext(0.03, 0.48, "(c)", fontsize="large")
plt.figtext(0.51, 0.48, "(d)", fontsize="large")

plot_file = os.path.abspath("plots/2D_Cauchy_state_preparation_resource_scaling.pdf")
fig.savefig(plot_file, transparent=False, bbox_inches="tight", pad_inches="layout")
