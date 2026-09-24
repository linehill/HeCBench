#!/bin/bash

RUNCONF=${1:-despmd-paper-alderlake-i7-12700.runconf}
RESULT_PREFIX="benchmark-results/omp-vs-ocl-$(date +%F-%H.%M.%S)"

for env in \
	sycl-acpp-pocl-7.2-pre.env \
	sycl-acpp-pocl-lvn.env \
	sycl-acpp-intel-ocl.env \
	sycl-acpp-rusticl-llvmpipe.env \
	sycl-acpp-openmp-single-threaded.env \
	sycl-acpp-pocl-7.2-pre-multi-threaded.env \
	sycl-acpp-pocl-lvn-multi-threaded.env \
	sycl-acpp-intel-ocl-multi-threaded.env \
	sycl-acpp-rusticl-llvmpipe-multi-threaded.env \
	sycl-acpp-openmp-multi-threaded.env
do

    echo '#####################################################################'
    ./despmd_paper_benchmark.bash -o "$RESULT_PREFIX/$env" \
				  environments/alderlake/$env ${RUNCONF}
done

wc -l $(find "$RESULT_PREFIX" -name \*.csv | sort)
