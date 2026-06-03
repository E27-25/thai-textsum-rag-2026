#!/bin/bash
set -e

PODMAN_ROOT="/tmp/podman-root-${USER}"
PODMAN_RUN="/tmp/podman-run-${USER}"

cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther

echo "=== [$(date)] Cleaning old podman storage ==="
podman unshare rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || rm -rf "$PODMAN_ROOT" "$PODMAN_RUN" 2>/dev/null || true
mkdir -p "$PODMAN_ROOT" "$PODMAN_RUN"
df -h /tmp

# Swap dockerignore: exclude Qwen3-14B, include Qwen3.6-35B-A3B-FP8
cp .dockerignore .dockerignore.14b.bak
cat > .dockerignore << 'EOF'
env/
Qwen3-14B/
Qwen3-32B/
Qwen3.6-27B/
Qwen3.6-35B-A3B/
*.out
*.csv
__pycache__/
.git/
inference_train.py
eval_train.py
script*.sh
swait.sh
EOF

echo "=== [$(date)] Build start (FP8) ==="

podman build \
  --storage-driver=overlay \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  --storage-opt ignore_chown_errors=true \
  -f Dockerfile.fp8 \
  -t textsum:fp8 \
  .

# Restore original dockerignore
cp .dockerignore.14b.bak .dockerignore

echo "=== [$(date)] Build done — pushing ==="

podman login registry.ai.in.th \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  -u 'watin promfiy.tme5' \
  -p '09022544'

podman push \
  --root="$PODMAN_ROOT" \
  --runroot="$PODMAN_RUN" \
  textsum:fp8 \
  'registry.ai.in.th/2026-textsum/48f0b4ab/watin-promfiy.tme5:AI-Benchmark-Programs-2026'

echo "=== [$(date)] Push done! ==="
