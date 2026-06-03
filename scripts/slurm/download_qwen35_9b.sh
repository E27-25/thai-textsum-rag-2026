#!/bin/bash
#SBATCH -p compute
#SBATCH -N 1 -c 4
#SBATCH -t 02:00:00
#SBATCH -A zz991016
#SBATCH -J dl_9b
#SBATCH -o dl_9b-%j.out

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/env/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    local_dir="./Qwen3.5-9B",
    local_dir_use_symlinks=False,
)
print("Download complete: ./Qwen3.5-9B")
EOF
