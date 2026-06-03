#!/bin/bash
#SBATCH -p memory                   # memory partition สำหรับงานหนัก
#SBATCH -N 1 -c 32
#SBATCH -t 04:00:00
#SBATCH -A zz991016
#SBATCH -J BuildPush
#SBATCH -o build_push-%j.out

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

IMAGE_NAME='registry.ai.in.th/2026-textsum/48f0b4ab/watin promfiy.tme5'
USER_TAG="v1"

echo "=== Building image ==="
podman build -t "${IMAGE_NAME}:${USER_TAG}" .

echo "=== Pushing image ==="
podman push "${IMAGE_NAME}:${USER_TAG}"

echo "=== Done ==="
