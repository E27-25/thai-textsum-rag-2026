#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 8
#SBATCH -t 02:00:00
#SBATCH -A zz991016
#SBATCH -J vllm_9b
#SBATCH -o vllm_9b-%j.out

module load cudatoolkit/24.11_12.6
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "vllm: $(/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python -c 'import vllm; print(vllm.__version__)')"

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python eval_vllm.py \
  --model ./Qwen3.5-9B \
  --tp 1 \
  --gpu-util 0.88 \
  --out submission_vllm_9b.csv
