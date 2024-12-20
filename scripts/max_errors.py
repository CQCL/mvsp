import argparse
import functools
import json
import logging
import os

import numpy as np
import scipy

from mvsp import target_functions
from mvsp.preprocessing.chebyshev import Chebyshev2D
from mvsp.preprocessing.fourier import Fourier2D
from mvsp.utils.paper_utils import slugify, make_suffix

# This script computes the maximum approximation error of Fourier or Chebyshev
# series over the domain of a 2D target function for varying degre. It stores
# results in a json file.
#
# Given a target function :math:`f(x, y)` and degree-`d` polynomial
# approximation :math:`f_d(x, y)` it computes
#
# .. math::
#
#   E_d = max_{x, y} |f_d(x, y) - f(x, y)|
#
# usage: max_errors.py [-h] -d D [D ...] [-f FUNCTION] [--func-args FUNC_ARGS] [-t {chebyhev,fourier}] [-o OUTPUT_PATH]
#
# Generate a file with max errors from a Fourier or Chebyshev approximation of a 2D function evaluated for varying degrees
#
# options:
#   -h, --help            show this help message and exit
#   -d D [D ...]          List of degrees for the approximating series
#   -f FUNCTION, --function FUNCTION
#                         Name of function to evaluate (default: cauchy2d)
#   --func-args FUNC_ARGS
#                         Additional arguments to pass to the function
#   -t {chebyhev,fourier}, --type {chebyhev,fourier}
#                         Type of series. Either 'chebyshev' or 'fourier' (default: chebyshev)
#   -o OUTPUT_PATH, --output_path OUTPUT_PATH
#                         Path for output files (default: ../data/)
#
# Provide a a function name with -f. This function name is imported from
# `mvsp.target_function`. Currently, it should take two positional arguments x
# and y. Other arguments should be optional. They can be passed via --func_args.
# --func_args takes a valid json string containing key-value pairs, which are
# passed the the function specified with -f. For example -f cauchy2d --func-args '{"mean":
# [0.5, 0.5]}' passes mean=[0.5, 0.5] to function cauchy2d.
#


logger = logging.getLogger("scaling")
logger.propagate = False
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(
    description="Generate a file with max errors from a Fourier or Chebyshev approximation of a 2D function evaluated for varying degrees"
)

parser.add_argument(
    "-d",
    type=int,
    nargs="+",
    required=True,
    help="List of degrees for the approximating series",
)
parser.add_argument(
    "-t",
    "--type",
    type=str.lower,
    choices=["chebyshev", "fourier"],
    default="chebyshev",
    help="Type of series. Either 'chebyshev' or 'fourier' (default: chebyshev)",
)
parser.add_argument(
    "-f",
    "--function",
    type=str,
    default="cauchy2d",
    help="Name of function to evaluate (default: cauchy2d)",
)
parser.add_argument(
    "--func-args", type=str, help="Additional arguments to pass to function"
)
parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    default="../data/",
    help="Path for output files (default: ../data/)",
)

args = parser.parse_args()
base_dir = os.getcwd()

ds = args.d
func_name = args.function
func_args = args.func_args
if args.func_args is None:
    func_args = {}
else:
    func_args = json.loads(args.func_args)
series_type = args.type
output_path = os.path.join(base_dir, args.output_path)
func_ = getattr(target_functions, func_name)
func = functools.partial(func_, **func_args)

id_suffix = make_suffix(func_name, series_type, func_args)

# Log file
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"%(asctime)s - %(name)s [{id_suffix}] - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

log_filename = slugify(f"max_error_{id_suffix}") + ".log"
log_dir_path = os.path.join(base_dir, "logs/")
log_file = os.path.join(log_dir_path, log_filename)
os.makedirs(log_dir_path, exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


if series_type == "chebyshev":
    series = Chebyshev2D
    bounds = ((-1, 1), (-1, 1))
    series_kwargs = {"run_ge": False}
else:
    series = Fourier2D
    bounds = ((0, 1), (0, 1))
    series_kwargs = {}
max_errors = []
for d in ds:
    logger.info(f"Computing degree {d=}")

    func_approx = series(func, d, **series_kwargs)
    coeffs = func_approx.coeffs

    res = scipy.optimize.shgo(
        lambda x: -np.abs(func_approx(x[0], x[1]) - func(x[0], x[1])),
        bounds=bounds,
        n=32,
        # sampling_method="sobol",
    )
    logger.info(f"Optimization successful: {res.success}")
    logger.debug(f"Max error={-res.fun} at {res.x}")
    max_errors.append(-res.fun)

output = {
    "ds": ds,
    "max_errors": max_errors,
    "series_type": series_type,
    "func_name": func_name,
    "func_args": func_args,
}

filename = slugify(f"max_error_{id_suffix}") + ".json"
output_file = os.path.join(output_path, filename)
os.makedirs(output_path, exist_ok=True)

logger.info(f"Saving to {output_file}")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

logger.info("Done")
