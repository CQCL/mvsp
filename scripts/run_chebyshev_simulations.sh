#!/bin/bash

processes=3
device_name="Aer"
optimisation_level=0
out="../data/Chebyshev_ricker2d_resource_scaling"

for d in 1 3 7 15
do
	for n in 4 5 6
	do
        python compute_resources.py -d $d -n $n $n --vendor IBM --optimisation-level $optimisation_level --output-path $out &

		background=( $(jobs -p) )
    	if (( ${#background[@]} == processes )); then
        	wait -n
    	fi
	done
done
