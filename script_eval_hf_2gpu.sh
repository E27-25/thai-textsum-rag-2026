#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=2
#SBATCH -N 1 -c 16
#SBATCH -t 03:00:00
#SBATCH -A zz991016
#SBATCH -J hf_2gpu
#SBATCH -o hf_2gpu-%j.out

# MODEL and OUT passed via --export
MODEL=${MODEL:-"./Qwen3-32B"}
OUT=${OUT:-"submission_hf_32b.csv"}

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2)"

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_hf.py \
  --model "$MODEL" \
  --out "$OUT" \
  --batch 2
