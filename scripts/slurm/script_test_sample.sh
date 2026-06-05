#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 00:50:00
#SBATCH -A zz991016
#SBATCH -J test_sample
#SBATCH -o test_sample-%j.out

set -e
ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python
export PYTHONNOUSERSITE=1   # skip ~/.local — older huggingface_hub there breaks peft
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=========================================="
echo "  GPU info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="
echo "  PASS A: dense-only (no reranker)"
echo "=========================================="
USE_RERANKER=0 TOP_K_FINAL=7 OUT_PATH=sample_pred_dense.json $PYTHON test_sample_timing.py

echo "=========================================="
echo "  PASS B: dense + reranker"
echo "=========================================="
USE_RERANKER=1 TOP_K_RETRIEVE=15 TOP_K_FINAL=7 OUT_PATH=sample_pred_rerank.json $PYTHON test_sample_timing.py

echo "Done."
