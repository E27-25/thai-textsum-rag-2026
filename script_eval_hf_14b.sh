#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J hf_14b
#SBATCH -o hf_14b-%j.out

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_hf.py \
  --model ./Qwen3-14B \
  --out submission_hf_14b.csv \
  --batch 4
