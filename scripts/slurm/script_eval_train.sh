#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 04:00:00
#SBATCH -A zz991016
#SBATCH -J eval_train
#SBATCH -o eval_train-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python
PYSITE=$ENV_BASE/lib/python3.12/site-packages

# CUDA runtime libs (nvidia-* wheels) must be on LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find $PYSITE/nvidia -name 'lib' -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON --version

echo "=== [$(date)] Step 1: Inference on training data ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON infer_train.py

echo "=== [$(date)] Step 2: Score + per-question analysis ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON eval_train.py

echo "=== [$(date)] Done! Results: submission_train.csv + eval_detail.json ==="
