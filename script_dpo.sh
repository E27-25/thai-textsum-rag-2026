#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J train_dpo_v1
#SBATCH -o train_dpo_v1-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON --version
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import torch; print('torch OK:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "=== [$(date)] Step 1: DPO Training (generate pairs + train) ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON train_dpo.py

echo "=== [$(date)] Step 2: Merge DPO LoRA ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON merge_lora_dpo.py

echo "=== [$(date)] Step 3: Copy tokenizer ==="
mkdir -p Qwen3-14B-DPO-v1
cp Qwen3-14B-SFT-v3/tokenizer_config.json Qwen3-14B-DPO-v1/
cp Qwen3-14B-SFT-v3/tokenizer.json        Qwen3-14B-DPO-v1/ 2>/dev/null || true
cp Qwen3-14B-SFT-v3/merges.txt            Qwen3-14B-DPO-v1/ 2>/dev/null || true
cp Qwen3-14B-SFT-v3/vocab.json            Qwen3-14B-DPO-v1/ 2>/dev/null || true

echo "=== [$(date)] Done! Qwen3-14B-DPO-v1 ready ==="
