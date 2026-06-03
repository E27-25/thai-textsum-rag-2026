#!/bin/bash
set -e

LUSTRE_BASE="/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther"
PODMAN_ROOT="/dev/shm/podman-root-sftv3-${USER}"
PODMAN_RUN="/tmp/podman-run-sftv3-${USER}"
LUSTRE_TMP="${LUSTRE_BASE}/podman-tmp"

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

mkdir -p "$LUSTRE_TMP"
export TMPDIR="$LUSTRE_TMP"

echo "=== [$(date)] Cleaning old podman storage ==="
podman --storage-driver=overlay --root="$PODMAN_ROOT" --runroot="$PODMAN_RUN" rmi --all --force 2>/dev/null || true
podman unshare rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUN"
df -h /dev/shm

echo "=== [$(date)] Build start (32B SFT v3) ==="

podman build \
  --storage-driver=overlay \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  --storage-opt ignore_chown_errors=true \
  --ignorefile .dockerignore.32b_v3_inference \
  -t textsum-32b-sft-v3:v1 \
  -f Dockerfile.32b_sft_v3 \
  .

echo "=== [$(date)] Build done — pushing ==="

podman login registry.ai.in.th \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  -u 'watin promfiy.tme5' \
  -p '09022544'

for attempt in 1 2 3 4 5; do
  echo "=== [$(date)] Push attempt $attempt ==="
  podman push \
    --root="$PODMAN_ROOT" \
    --runroot="$PODMAN_RUN" \
    textsum-32b-sft-v3:v1 \
    'registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026-32b-sft-v3' && break
  echo "Push failed, retrying in 60s..."
  sleep 60
done

echo "=== [$(date)] Push done! ==="
echo "Tag: AI-Benchmark-Programs-2026-32b-sft-v3"
