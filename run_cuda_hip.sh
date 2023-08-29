#!/bin/bash

CUDA=0
HIP=1
# sycl maybe later
REPEATS=1
NVIDIA_SM=50


# -------------------

if [ $CUDA -ne 0 ]; then
  ./scripts/autohecbench.py --warmup true --repeat $REPEATS --nvidia-sm $NVIDIA_SM -o test_Rodinia_${REPEATS}x_cuda.csv  backprop-cuda b+tree-cuda bfs-cuda cfd-cuda  gaussian-cuda heartwall-cuda hotspot3D-cuda hotspot-cuda hybridsort-cuda kmeans-cuda lavaMD-cuda  lud-cuda myocyte-cuda nw-cuda nn-cuda particlefilter-cuda pathfinder-cuda srad-cuda streamcluster-cuda
fi

# -------------------

if [ $HIP -ne 0 ]; then

  unset CHIP_PLATFORM
  export CHIP_BE=opencl
  export CHIP_DEVICE_TYPE=gpu
  export CHIP_LOGLEVEL=warn

  ./scripts/autohecbench.py --warmup true --repeat $REPEATS  -o test_Rodinia_${REPEATS}x_hipOclBE.csv  backprop-hip b+tree-hip cfd-hip bfs-hip	 gaussian-hip heartwall-hip hotspot3D-hip hotspot-hip hybridsort-hip kmeans-hip lavaMD-hip  lud-hip myocyte-hip nw-hip nn-hip particlefilter-hip pathfinder-hip srad-hip streamcluster-hip
fi
