#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 00:30:00
#SBATCH -A zz991016
#SBATCH -J FP8Test
#SBATCH -o fp8_test-%j.out

module reset
ml Mamba
conda deactivate
conda activate /project/zz991000-zdeva/zz991016/Arther/env
export PYTHONNOUSERSITE=1

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== FP8 single-GPU quick test (50 queries) ==="
python3 test_fp8_quick.py
