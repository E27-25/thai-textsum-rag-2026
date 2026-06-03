#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -N 1 -c 16
#SBATCH -t 02:00:00
#SBATCH -A zz991016
#SBATCH -J PodmanBuild
#SBATCH -o podman_build-%j.out

PODMAN_ROOT="/tmp/podman-root-${USER}"
PODMAN_RUN="/tmp/podman-run-${USER}"
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUN"

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== Building image on $(hostname) | /tmp: $(df -h /tmp | tail -1) ==="

podman build \
  --storage-driver=overlay \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  -t textsum:v1 \
  . 2>&1

if [ $? -ne 0 ]; then
  echo "Build failed — check errors above"
  exit 1
fi

echo "=== Build succeeded — pushing to registry ==="

podman login registry.ai.in.th \
  -u 'watin promfiy.tme5' \
  -p '09022544' \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN"

podman push \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  textsum:v1 \
  'registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026'

echo "=== Done ==="
