#!/bin/bash

RUNCONF=${1:-despmd-paper-alderlake-i7-12700.runconf}
RESULT_PREFIX="benchmark-results/$(date +%F-%H.%M.%S)"

for env in \
        cuda-chipstar-pocl-7.2-pre.env \
	cuda-chipstar-pocl-lvn.env \
	cuda-chipstar-pocl-lvn-ude0.env \
	cuda-chipstar-intel-ocl.env \
	cuda-chipstar-rusticl-llvmpipe.env \
	sycl-acpp-pocl-7.2-pre.env \
	sycl-acpp-pocl-lvn.env \
	sycl-acpp-pocl-lvn-ude0.env \
	sycl-acpp-intel-ocl.env \
	sycl-acpp-rusticl-llvmpipe.env
do

    echo '#####################################################################'
    ./despmd_paper_benchmark.bash -o "$RESULT_PREFIX/$env" \
				  environments/alderlake/$env ${RUNCONF}
done

wc -l $(find "$RESULT_PREFIX" -name \*.csv | sort)
