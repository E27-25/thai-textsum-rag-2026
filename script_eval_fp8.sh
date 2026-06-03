#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 01:00:00
#SBATCH -A zz991016
#SBATCH -J eval_fp8
#SBATCH -o eval_fp8-%j.out

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# FP8 weights stay in 8-bit on A100 (W8A16), fits in 40GB
/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_vllm.py \
  --model ./Qwen3.6-35B-A3B-FP8 \
  --tp 1 \
  --gpu-util 0.92 \
  --out submission_fp8.csv
