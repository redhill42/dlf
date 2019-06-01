# This file describes the standard way to build DLF, using Docker
#
# Usage:
#
# # Assemble the full dev environment. This is slow the first time.
# docker build -t dlf-dev .
#
# # Mount your source in an iteractive container for quick testing:
# docker run -v `pwd`:/devel/dlf --privileged -ti dlf-dev bash
#
# # Run the test-suite:
# docker run --privileged dlf-dev build/make.sh test

FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake wget libprotobuf-dev protobuf-compiler libgmp-dev && \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add - && \
    sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && apt-get install -y --no-install-recommends intel-mkl-64bit-2019.4-070 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_PATH "/usr/local/cuda"
ENV CUDA_LIB_PATH "/usr/local/cuda/lib64/stubs"
ENV MKLROOT "/opt/intel/mkl"

WORKDIR /devel/dlf

# Wrap all commands in the "docker-in-docker" script to allow nested containers
ENTRYPOINT ["build/dind"]

# Upload source
COPY . /devel/dlf
