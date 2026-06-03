#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 4
#SBATCH -t 00:05:00
#SBATCH -A zz991016
#SBATCH -J diag
#SBATCH -o diag-%j.out

ENV_BASE=/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env
PYTHON=$ENV_BASE/bin/python

echo "=== sys.prefix ==="
$PYTHON -c "import sys; print(sys.prefix)"

echo "=== sys.path (default) ==="
$PYTHON -c "import sys; [print(p) for p in sys.path]"

echo "=== with PYTHONPATH set ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import sys; [print(p) for p in sys.path]"

echo "=== datasets import test ==="
PYTHONPATH=$ENV_BASE/lib/python3.12/site-packages $PYTHON -c "import datasets; print('OK:', datasets.__version__)"

echo "=== ls site-packages datasets ==="
ls $ENV_BASE/lib/python3.12/site-packages/ | grep dataset
