#!/bin/bash
#SBATCH -p gpu                      # Specify partition [Compute/Memory/GPU]
#SBATCH --gpus-per-node=4           # Specify number of GPUs
#SBATCH -N 1 -c 64                  # Specify number of nodes and CPUs
#SBATCH -t 120:00:00                # Specify maximum time limit (hour:minute:second)
#SBATCH -A zz991016                 # Specify project name
#SBATCH -J Dayoo                    # Specify job name
#SBATCH -o qwen3_14-%j.out            # Output log file

module reset
ml Mamba
# module load Mamba/23.11.0-0
# module load cuda/12.6
conda deactivate
conda activate /project/zz991000-zdeva/zz991016/Arther/env
export PYTHONNOUSERSITE=1   # avoid ~/.local override project env

python "${1:-inference.py}"