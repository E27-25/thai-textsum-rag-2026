#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 16
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J dpo_32b
#SBATCH -o dpo_32b-%j.out

set -e

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PYTHON --version
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import torch; print('torch OK:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo "=== [$(date)] Step 1: Generate DPO rejected samples ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON gen_dpo_rejected.py

echo "=== [$(date)] Step 2: DPO Training ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON train_dpo_32b.py

echo "=== [$(date)] Step 3: Merge LoRA (CPU) ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON merge_lora_dpo_32b.py

echo "=== [$(date)] Step 4: AWQ 4-bit quantize ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON quantize_dpo_32b_awq.py

echo "=== [$(date)] Step 5: Copy tokenizer ==="
mkdir -p Qwen3-32B-DPO-v1-AWQ
cp Qwen3-32B/tokenizer_config.json Qwen3-32B-DPO-v1-AWQ/ 2>/dev/null || true
cp Qwen3-32B/tokenizer.json        Qwen3-32B-DPO-v1-AWQ/ 2>/dev/null || true
cp Qwen3-32B/merges.txt            Qwen3-32B-DPO-v1-AWQ/ 2>/dev/null || true
cp Qwen3-32B/vocab.json            Qwen3-32B-DPO-v1-AWQ/ 2>/dev/null || true

echo "=== [$(date)] Done! Qwen3-32B-DPO-v1-AWQ ready ==="
du -sh Qwen3-32B-DPO-v1-AWQ/
