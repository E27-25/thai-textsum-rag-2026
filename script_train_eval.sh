#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1 -c 64
#SBATCH -t 120:00:00
#SBATCH -A zz991016
#SBATCH -J TrainEval
#SBATCH -o train_eval-%j.out

module reset
ml Mamba
conda deactivate
conda activate /project/zz991000-zdeva/zz991016/Arther/env
export PYTHONNOUSERSITE=1

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== Step 1: Inference on train set ==="
python inference_train.py

echo "=== Step 2: Evaluate against ground truth ==="
python eval_train.py
