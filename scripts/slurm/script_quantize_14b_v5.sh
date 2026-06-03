#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 16
#SBATCH -t 02:00:00
#SBATCH -A zz991016
#SBATCH -J quantize_14b_v5
#SBATCH -o quantize_14b_v5-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] AWQ quantize 14B SFT v5 ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON quantize_14b_v5_awq.py

echo "=== [$(date)] Done! ==="
du -sh Qwen3-14B-SFT-v5-AWQ/
