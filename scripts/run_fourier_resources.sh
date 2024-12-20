#!/bin/bash

processes=3
vendor="quantinuum"
device_name="H2-1E"
optimisation_level=2
fun="cauchy2d"
series_type="fourier"
out="../data/Fourier_cauchy2d_resource_scaling"

for d in 1 3 5 7 15 31 63
do
	for n in 4 5 6
	do
        python compute_resources.py -d $d -n $n $n -t $series_type -f $fun --vendor $vendor -q $device_name --optimisation-level $optimisation_level --output-path $out --compile-only &
	
		background=( $(jobs -p) )
    	if (( ${#background[@]} == processes )); then
        	wait -n
    	fi
	done
done
