#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 64
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J MoEEval
#SBATCH -o moe_eval-%j.out

module reset
ml Mamba
conda deactivate
conda activate /project/zz991000-zdeva/zz991016/Arther/env
export PYTHONNOUSERSITE=1

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== Step 1: Inference (Qwen3.6-35B-A3B multi-GPU BF16) on train set ==="
# ใช้ 4x A100 บน Lanta แทน INT4 — ไม่ต้อง quantize
LANTA_MULTIGPU=1 python3 inference_moe_train.py

echo "=== Step 2: Evaluate against ground truth ==="
python3 eval_train.py
