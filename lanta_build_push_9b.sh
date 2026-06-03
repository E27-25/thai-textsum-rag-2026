#!/bin/bash
set -e

PODMAN_ROOT="/tmp/podman-root-${USER}"
PODMAN_RUN="/tmp/podman-run-${USER}"

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] Cleaning old podman storage ==="
podman unshare rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUN"
df -h /tmp

# Swap dockerignore: exclude 14B/FP8/etc, include Qwen3.5-9B
cp .dockerignore .dockerignore.bak
cat > .dockerignore << 'IGNORE'
env/
Qwen3-14B/
Qwen3-32B/
Qwen3.6-27B/
Qwen3.6-35B-A3B/
Qwen3.6-35B-A3B-FP8/
*.out
*.csv
__pycache__/
.git/
inference_train.py
eval_train.py
script*.sh
swait.sh
IGNORE

echo "=== [$(date)] Build start (9B) ==="

podman build \
  --storage-driver=overlay \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  --storage-opt ignore_chown_errors=true \
  -f Dockerfile.9b \
  -t textsum:9b \
  .

# Restore original dockerignore
cp .dockerignore.bak .dockerignore

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
    textsum:9b \
    'registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026' && break
  echo "Push failed, retrying in 60s..."
  sleep 60
done

echo "=== [$(date)] Push done! ==="
