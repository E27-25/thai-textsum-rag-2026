#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 01:00:00
#SBATCH -A zz991016
#SBATCH -J eval_14b
#SBATCH -o eval_14b-%j.out

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_vllm.py \
  --model ./Qwen3-14B \
  --tp 1 \
  --gpu-util 0.88 \
  --out submission_14b.csv
