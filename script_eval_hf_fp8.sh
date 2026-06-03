#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 02:00:00
#SBATCH -A zz991016
#SBATCH -J hf_fp8
#SBATCH -o hf_fp8-%j.out

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# FP8 MoE (3B active params) — try BF16 first, if OOM use --load-4bit
/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_hf.py \
  --model ./Qwen3.6-35B-A3B-FP8 \
  --out submission_hf_fp8.csv \
  --batch 2
