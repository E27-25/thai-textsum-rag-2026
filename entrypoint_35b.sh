#!/bin/bash

PYSITE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export LD_LIBRARY_PATH=$(python3 -c "
import glob, os, sysconfig
pysite = sysconfig.get_paths()['purelib']
dirs = (
    glob.glob(f'{pysite}/nvidia/*/lib') +
    glob.glob(f'{pysite}/torch/lib') +
    glob.glob(f'{pysite}/cuda*/lib*') +
    ['/usr/local/cuda/lib64']
)
print(':'.join(d for d in dirs if os.path.isdir(d)))
"):${LD_LIBRARY_PATH:-}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

export VLLM_USE_V1=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec python3 /model/run.py
