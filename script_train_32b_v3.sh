#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 16
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J train_32b_v3
#SBATCH -o train_32b_v3-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [$(date)] Step 1: Train 32B SFT v3 (10 distractors + paraphrases) ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON train_sft_32b_v3.py

echo "=== [$(date)] Step 2: Merge LoRA ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON merge_lora_32b_v3.py

echo "=== [$(date)] Step 3: AWQ 4-bit quantize ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON quantize_32b_v3_awq.py

echo "=== [$(date)] Step 4: Copy tokenizer ==="
mkdir -p Qwen3-32B-SFT-v3-AWQ
cp Qwen3-32B/tokenizer_config.json Qwen3-32B-SFT-v3-AWQ/ 2>/dev/null || true
cp Qwen3-32B/tokenizer.json        Qwen3-32B-SFT-v3-AWQ/ 2>/dev/null || true
cp Qwen3-32B/merges.txt            Qwen3-32B-SFT-v3-AWQ/ 2>/dev/null || true
cp Qwen3-32B/vocab.json            Qwen3-32B-SFT-v3-AWQ/ 2>/dev/null || true

echo "=== [$(date)] Done! Qwen3-32B-SFT-v3-AWQ ready ==="
du -sh Qwen3-32B-SFT-v3-AWQ/
