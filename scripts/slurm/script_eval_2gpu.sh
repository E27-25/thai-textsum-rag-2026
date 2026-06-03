#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=2
#SBATCH -N 1 -c 16
#SBATCH -t 01:30:00
#SBATCH -A zz991016
#SBATCH -J eval_2gpu
#SBATCH -o eval_2gpu-%j.out

# MODEL and OUT passed via --export or set here
MODEL=${MODEL:-"./Qwen3-32B"}
OUT=${OUT:-"submission_32b.csv"}

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2)"

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_vllm.py \
  --model "$MODEL" \
  --tp 2 \
  --gpu-util 0.88 \
  --out "$OUT"
