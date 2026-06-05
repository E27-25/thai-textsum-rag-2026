#!/bin/bash
set -e

LUSTRE_BASE="/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther"
PODMAN_ROOT="/dev/shm/podman-root-14bdense-${USER}"
PODMAN_RUN="/tmp/podman-run-14bdense-${USER}"
LUSTRE_TMP="${LUSTRE_BASE}/podman-tmp"

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

# Sanity check: AWQ model must exist
if [ ! -d "Qwen3-14B-SFT-v5-AWQ" ]; then
    echo "ERROR: Qwen3-14B-SFT-v5-AWQ not found."
    exit 1
fi

mkdir -p "$LUSTRE_TMP"
export TMPDIR="$LUSTRE_TMP"

echo "=== [$(date)] Cleaning old podman storage ==="
podman --storage-driver=overlay --root="$PODMAN_ROOT" --runroot="$PODMAN_RUN" rmi --all --force 2>/dev/null || true
podman unshare rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUN"

# Also nuke any leftover from earlier builds in /dev/shm
podman unshare rm -rf /dev/shm/podman-root-14bawq-${USER} 2>/dev/null || true
podman unshare rm -rf /dev/shm/podman-root-refboost*-${USER} 2>/dev/null || true
df -h /dev/shm

echo "=== [$(date)] Build start (14B dense — no reranker + EDA fixes) ==="

podman build \
  --storage-driver=overlay \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  --storage-opt ignore_chown_errors=true \
  --ignorefile .dockerignore.14b_dense_inference \
  -t textsum-14b-dense:v2 \
  -f Dockerfile.14b_dense \
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
    textsum-14b-dense:v2 \
    'registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026-14b-dense-v2' && break
  echo "Push failed, retrying in 60s..."
  sleep 60
done

echo "=== [$(date)] Push done! ==="
echo "Tag: AI-Benchmark-Programs-2026-14b-dense-v2"
