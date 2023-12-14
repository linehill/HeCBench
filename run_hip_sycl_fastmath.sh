#!/bin/bash

LEVEL0=1
OPENCL=0
REPEATS=4

export CHIP_L0_COLLECT_EVENTS_TIMEOUT=5
export CHIP_L0_IMM_CMD_LISTS=1
# export CHIP_DUMP_SPIRV=1
# export SYCL_DUMP_IMAGES=1

for D in $(echo *-hip *-sycl); do
  mkdir -p $D/l0_cache
done

export CHIP_DEVICE_TYPE=gpu
export CHIP_LOGLEVEL=crit

# -------------------

if [ $OPENCL -ne 0 ]; then

  export CHIP_BE=opencl
  export CHIP_PLATFORM=0
  ./scripts/autohecbench.py --warmup true --repeat ${REPEATS} --extra-compile-flags="-ffast-math" -o test_FULL_${REPEATS}_x_hip_fast_oclBE.csv hip
fi

# -------------------

if [ $LEVEL0 -ne 0 ]; then

  export CHIP_BE=level0

  ./scripts/autohecbench.py --warmup true --repeat ${REPEATS} --extra-compile-flags="-ffast-math" -o test_FULL_${REPEATS}_x_hip_fast_l0BE.csv hip
fi

#############################################

source /opt/intel/oneapi/setvars.sh

if [ $OPENCL -ne 0 ]; then
  export SYCL_DEVICE_FILTER=opencl:gpu:0
  ./scripts/autohecbench.py --warmup true --repeat ${REPEATS} --extra-compile-flags="-fp-model=fast -fno-sycl-instrument-device-code" -o test_FULL_${REPEATS}_x_sycl_fast_oclBE.csv --sycl-type opencl sycl
fi

if [ $LEVEL0 -ne 0 ]; then
  export SYCL_DEVICE_FILTER=ext_oneapi_level_zero:gpu:0
  ./scripts/autohecbench.py --warmup true --repeat ${REPEATS} --extra-compile-flags="-fp-model=fast -fno-sycl-instrument-device-code" -o test_FULL_${REPEATS}_x_sycl_fast_l0BE.csv --sycl-type opencl sycl
fi
