#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 03:00:00
#SBATCH -A zz991016
#SBATCH -J hf_32b_4bit
#SBATCH -o hf_32b_4bit-%j.out

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_hf.py \
  --model ./Qwen3-32B \
  --out submission_hf_32b.csv \
  --batch 4 \
  --load-4bit
