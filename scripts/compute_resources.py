import argparse
import functools
import json
import logging
import os
import time

from pytket.extensions.qiskit import AerStateBackend
from pytket.extensions.quantinuum import QuantinuumBackend

import mvsp.target_functions as target_functions
from mvsp.utils.resources import resource_scaling
from mvsp.utils.paper_utils import NumpyEncoder, make_suffix, slugify

# This script generates resource files (number of 2q gates, qubits etc.) and,
# optionally, simulates the circuit (controlled by flag --compile-only). In the
# latter case, the output of the evaluated circuit is also stored.
#
# Example 1:
# Compute resources for Chebyshev approach on the ricker2d function with
# compilation for Quantinuum (defaults to H2-1E) and optimisation level 2:
# python compute_resources.py -f ricker2d -d 2 -n 4 4 -t chebyshev --vendor
# quantinuum --optimisation-level 2 --compile-only
#
# Example 2:
# Compute resources and simulate the Fourier approach for the cauchy2d function
# with IBM Aer simulator:
# python compute_resources.py -f cauchy2d -d 2 -n 4 4 -t fourier --vendor ibm
# --optimisation-level 0

logger = logging.getLogger("Resources")
logger.propagate = False
logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser(
    description="Generate a file with resources for the state prep algorithm and, optionally, the simulated circuit output. For speed, it is recommended to run circuit evaluations with IBM Aer at optimisation level 0"
)
parser.add_argument(
    "-d", type=int, required=True, help="Degree of Chebyshev polynomial"
)
parser.add_argument(
    "-n",
    nargs="+",
    type=int,
    required=True,
    help="Sequence of D integers specifying number of qubits in D dimensions",
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
    type=str,
    default="ricker2d",
    help="Name of function to evaluate (default: ricker2d)",
)
parser.add_argument(
    "--func-args", type=str, help="Additional arguments to pass to function"
)
parser.add_argument(
    "--vendor",
    type=str.lower,
    default="ibm",
    help="Quantum computer vendor used for backend (default: ibm)",
)
parser.add_argument(
    "-q",
    type=str,
    help="Target device for compilation. Supports Quantinuum device names and IBM AER (default: Aer for IBM and H2-1E for Quantinuum)",
)
parser.add_argument(
    "--optimisation-level",
    type=int,
    default=0,
    help="Compiler optimisation level passed to TKET (default: 0)",
)
parser.add_argument(
    "--output-path",
    type=str,
    default="data/",
    help="Path for output files (./data/)",
)
parser.add_argument(
    "--compile-only",
    action="store_true",
    help="If set to True, only output resources. If set to False, also run the circuits and evaluate the output (default: True)",
)

args = parser.parse_args()
base_dir = os.getcwd()

d = args.d
n = args.n
func_name = args.f
func_args = args.func_args
if args.func_args is None:
    func_args = {}
else:
    func_args = json.loads(args.func_args)
series_type = args.type
vendor = args.vendor
device_name = args.q
optimisation_level = args.optimisation_level
output_path = os.path.abspath(os.path.join(base_dir, args.output_path))
compile_only = args.compile_only
func_ = getattr(target_functions, func_name)
func = functools.partial(func_, **func_args)

if vendor == "quantinuum":
    if device_name is None:
        device_name = "H2-1E"
    backend = QuantinuumBackend(device_name=device_name, machine_debug=True)
elif vendor == "ibm":
    # Currently only support Aer state vector
    device_name = "Aer"
    backend = AerStateBackend()
else:
    raise ValueError(f"Vendor not supported (was {vendor=})")

id_suffix = make_suffix(
    func_name,
    series_type,
    n,
    d,
    vendor,
    device_name,
    optimisation_level,
    compile_only,
    slugify(func_args),
)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"%(asctime)s - %(name)s [{id_suffix}] - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

basename = f"RS_{id_suffix}"
log_filename = basename + ".log"
log_dir_path = os.path.join(base_dir, "logs/")
log_file = os.path.join(log_dir_path, log_filename)
os.makedirs(log_dir_path, exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


logger.info("Computing resources")
tic = time.perf_counter()
resources = resource_scaling(
    func,
    d,
    n,
    backend,
    series_type,
    compile_only=compile_only,
    compiler_options={"optimisation_level": optimisation_level},
    logger=logger,
)
logger.info(f"Elapsed time: {time.perf_counter()-tic}s")

filename = basename + ".json"
output_file = os.path.join(output_path, filename)
os.makedirs(output_path, exist_ok=True)

logger.info(f"Saving to {output_file}")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(resources, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

logger.info("Done")
