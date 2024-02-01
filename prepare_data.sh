#!/bin/bash

SCRIPT=$(readlink -f $0)
HECBENCH=$(dirname $SCRIPT)

for F in bsw-cuda/test-data.tar.gz ced-hip/frames.tar.gz columnarSolver-cuda/data.tar.gz sssp-cuda/data.tar.gz geodesic-sycl/locations.tar.gz snake-cuda/Datasets.tar.gz sssp-cuda/data.tar.gz urng-sycl/image.tar.gz ss-sycl/input.tar.gz sobel-sycl/data.tar.gz ; do

  echo "############ UNPACKING $F"

  D=$(dirname $F)
  cd $HECBENCH/$D && tar xzf $HECBENCH/$F

done

##############

cd $HECBENCH/sssp-hip ; ln -s ../sssp-cuda/input . ; ln -s ../sssp-cuda/output .

cd $HECBENCH/sssp-sycl ; ln -s ../sssp-cuda/input . ; ln -s ../sssp-cuda/output .

################

cd $HECBENCH/hogbom-cuda

mkdir -p data
cd data

if [ ! -e dirty_4096.img ] ; then
  wget https://github.com/ATNF/askap-benchmarks/raw/master/data/dirty_4096.img
fi
if [ ! -e psf_4096.img ] ; then
  wget https://github.com/ATNF/askap-benchmarks/raw/master/data/psf_4096.img
fi

###################

cd $HECBENCH/svd3x3-cuda

if [ ! -e Dataset_1M.txt ]; then
  wget https://github.com/kuiwuchn/3x3_SVD_CUDA/raw/master/svd3x3/svd3x3/Dataset_1M.txt
fi
