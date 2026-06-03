#!/bin/bash
# vllm 0.21.0 uses CUDA 13 libs at nvidia/cu13/lib/
PYSITE=/usr/local/lib/python3.11/site-packages
export LD_LIBRARY_PATH="${PYSITE}/nvidia/cu13/lib:${PYSITE}/nvidia/cublas/lib:${PYSITE}/torch/lib:${LD_LIBRARY_PATH:-}"
exec python3 /model/run.py
