#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 06:00:00
#SBATCH -A zz991016
#SBATCH -J train_sft_v3
#SBATCH -o train_sft_v3-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON --version
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import datasets; print('datasets OK:', datasets.__version__)"
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import torch; print('torch OK:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "=== [$(date)] Step 1: Train SFT v3 ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON train_sft.py

echo "=== [$(date)] Step 2: Merge LoRA ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON merge_lora.py

echo "=== [$(date)] Step 3: Copy tokenizer from base model ==="
mkdir -p Qwen3-14B-SFT-v3
cp Qwen3-14B/tokenizer_config.json Qwen3-14B-SFT-v3/
cp Qwen3-14B/tokenizer.json        Qwen3-14B-SFT-v3/ 2>/dev/null || true
cp Qwen3-14B/merges.txt            Qwen3-14B-SFT-v3/ 2>/dev/null || true
cp Qwen3-14B/vocab.json            Qwen3-14B-SFT-v3/ 2>/dev/null || true

echo "=== [$(date)] Done! Qwen3-14B-SFT-v3 ready ==="
