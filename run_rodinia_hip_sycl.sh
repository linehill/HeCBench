#!/bin/bash

LEVEL0=1
OPENCL=0
REPEATS=4

for D in $(echo *-hip *-sycl); do
  mkdir -p $D/l0_cache
done

# -------------------

if [ $OPENCL -ne 0 ]; then

  export CHIP_DEVICE_TYPE=gpu
  export CHIP_LOGLEVEL=crit

  export CHIP_BE=opencl
  #export CHIP_PLATFORM=2

  ./scripts/autohecbench.py --warmup true --repeat $REPEATS  -o test_Rodinia_4x_hip_oclBE.csv  backprop-hip b+tree-hip cfd-hip  gaussian-hip heartwall-hip hotspot3D-hip hotspot-hip hybridsort-hip kmeans-hip lavaMD-hip  lud-hip myocyte-hip nw-hip particlefilter-hip pathfinder-hip srad-hip streamcluster-hip
fi

# -------------------

if [ $LEVEL0 -ne 0 ]; then

  unset CHIP_PLATFORM
  export CHIP_BE=level0

  ./scripts/autohecbench.py --warmup true --repeat $REPEATS  -o test_Rodinia_4x_hip_l0BE.csv  backprop-hip b+tree-hip cfd-hip  gaussian-hip heartwall-hip hotspot3D-hip hotspot-hip hybridsort-hip kmeans-hip lavaMD-hip  lud-hip myocyte-hip nw-hip particlefilter-hip pathfinder-hip srad-hip streamcluster-hip
fi

#############################################

source /opt/intel/oneapi/setvars.sh

if [ $OPENCL -ne 0 ]; then
  export SYCL_DEVICE_FILTER=opencl:gpu:2
  ./scripts/autohecbench.py --warmup true --repeat $REPEATS -o test_Rodinia_4x_sycl_oclBE.csv --sycl-type opencl backprop-sycl b+tree-sycl cfd-sycl dwt2d-sycl gaussian-sycl heartwall-sycl hotspot3D-sycl hotspot-sycl hybridsort-sycl kmeans-sycl lavaMD-sycl  lud-sycl myocyte-sycl nw-sycl particlefilter-sycl pathfinder-sycl srad-sycl streamcluster-sycl
fi

if [ $LEVEL0 -ne 0 ]; then
  export SYCL_DEVICE_FILTER=ext_oneapi_level_zero:gpu:0
  ./scripts/autohecbench.py --warmup true --repeat $REPEATS -o test_Rodinia_4x_sycl_l0BE.csv --sycl-type opencl backprop-sycl b+tree-sycl cfd-sycl dwt2d-sycl gaussian-sycl heartwall-sycl hotspot3D-sycl hotspot-sycl hybridsort-sycl kmeans-sycl lavaMD-sycl  lud-sycl myocyte-sycl nw-sycl particlefilter-sycl pathfinder-sycl srad-sycl streamcluster-sycl
fi
