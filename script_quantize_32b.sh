#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 16
#SBATCH -t 04:00:00
#SBATCH -A zz991016
#SBATCH -J quantize_32b
#SBATCH -o quantize_32b-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] AWQ quantize 32B ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON quantize_32b_awq.py

echo "=== [$(date)] Done! ==="
du -sh Qwen3-32B-SFT-v1-AWQ/
