#!/bin/bash
PYSITE=/usr/local/lib/python3.11/site-packages
export LD_LIBRARY_PATH="\
${PYSITE}/nvidia/cuda_runtime/lib:\
${PYSITE}/nvidia/cublas/lib:\
${PYSITE}/nvidia/cudnn/lib:\
${PYSITE}/nvidia/nccl/lib:\
${PYSITE}/nvidia/nvjitlink/lib:\
${PYSITE}/nvidia/cusparse/lib:\
${PYSITE}/nvidia/cusolver/lib:\
${PYSITE}/nvidia/curand/lib:\
${PYSITE}/nvidia/cuda_nvrtc/lib:\
${PYSITE}/nvidia/cufft/lib:\
${PYSITE}/torch/lib:\
${LD_LIBRARY_PATH:-}"
export VLLM_USE_V1=0
exec python3 /model/run.py
