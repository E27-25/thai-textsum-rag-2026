#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 16
#SBATCH -t 12:00:00
#SBATCH -A zz991016
#SBATCH -J paraphrase_v3
#SBATCH -o paraphrase_v3-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] Generate Thai paraphrases (base Qwen3-32B) ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON gen_paraphrases_v3.py

echo "=== [$(date)] Done ==="
ls -la paraphrases_v3.json
