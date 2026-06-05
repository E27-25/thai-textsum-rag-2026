#!/bin/bash
set -e

LUSTRE_BASE="/lustrefs/disk/project/zz991000-zdeva/zz991016/Arther"
PODMAN_ROOT="/dev/shm/podman-root-32bthinkv2-${USER}"
PODMAN_RUN="/tmp/podman-run-32bthinkv2-${USER}"
LUSTRE_TMP="${LUSTRE_BASE}/podman-tmp"

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

if [ ! -d "Qwen3-32B-SFT-v1-AWQ" ]; then
    echo "ERROR: Qwen3-32B-SFT-v1-AWQ not found."
    exit 1
fi

mkdir -p "$LUSTRE_TMP"
export TMPDIR="$LUSTRE_TMP"

echo "=== [$(date)] Cleaning old podman storage ==="
podman --storage-driver=overlay --root="$PODMAN_ROOT" --runroot="$PODMAN_RUN" rmi --all --force 2>/dev/null || true
podman unshare rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUN"

# Nuke leftovers from earlier builds in /dev/shm to free space (32B = 19GB)
podman unshare rm -rf /dev/shm/podman-root-14bdense-${USER} 2>/dev/null || true
podman unshare rm -rf /dev/shm/podman-root-14bawq-${USER} 2>/dev/null || true
podman unshare rm -rf /dev/shm/podman-root-refboost*-${USER} 2>/dev/null || true
df -h /dev/shm

echo "=== [$(date)] Build start (32B thinking v2 — parser fix + drop verify_refs) ==="

podman build \
  --storage-driver=overlay \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  --storage-opt ignore_chown_errors=true \
  --ignorefile .dockerignore.32b_thinking_v2_inference \
  -t textsum-32b-thinking:v2 \
  -f Dockerfile.32b_thinking_v2 \
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
    textsum-32b-thinking:v2 \
    'registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026-32b-thinking-v2' && break
  echo "Push failed, retrying in 60s..."
  sleep 60
done

echo "=== [$(date)] Push done! ==="
echo "Tag: AI-Benchmark-Programs-2026-32b-thinking-v2"
