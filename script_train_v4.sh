#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J train_sft_v4_oracle
#SBATCH -o train_sft_v4_oracle-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON --version
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import torch; print('torch OK:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "=== [$(date)] Step 1: Train SFT v4 Oracle ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON train_sft_v4_oracle.py

echo "=== [$(date)] Step 2: Merge LoRA ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON merge_lora_v4.py

echo "=== [$(date)] Step 3: Copy tokenizer ==="
mkdir -p Qwen3-14B-Oracle-v4
cp Qwen3-14B/tokenizer_config.json Qwen3-14B-Oracle-v4/
cp Qwen3-14B/tokenizer.json        Qwen3-14B-Oracle-v4/ 2>/dev/null || true
cp Qwen3-14B/merges.txt            Qwen3-14B-Oracle-v4/ 2>/dev/null || true
cp Qwen3-14B/vocab.json            Qwen3-14B-Oracle-v4/ 2>/dev/null || true

echo "=== [$(date)] Done! Qwen3-14B-Oracle-v4 ready ==="
