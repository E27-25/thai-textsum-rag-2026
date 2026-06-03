#!/bin/bash
# Wait for 35B build to finish, clean up, then build Oracle v4
set -e

PODMAN_ROOT_35B="/dev/shm/podman-root-ua039"
PODMAN_RUN_35B="/tmp/podman-run-ua039"
BUILD_PID=3671930

echo "=== [$(date)] Waiting for 35B build (PID $BUILD_PID) to finish ==="
while kill -0 $BUILD_PID 2>/dev/null; do
    echo "  35B still running... ($(df -h /dev/shm | tail -1))"
    sleep 60
done
echo "=== [$(date)] 35B build done. Cleaning root ==="

podman --storage-driver=overlay --root="$PODMAN_ROOT_35B" --runroot="$PODMAN_RUN_35B" system reset --force 2>/dev/null || true
podman --storage-driver=overlay --root="$PODMAN_ROOT_35B" --runroot="$PODMAN_RUN_35B" unshare rm -rf "$PODMAN_ROOT_35B" 2>/dev/null || true
rm -rf "$PODMAN_ROOT_35B" 2>/dev/null || true
echo "=== [$(date)] Cleaned root, shm now: ==="
df -h /dev/shm

echo "=== [$(date)] Starting Oracle v4 build ==="
bash /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther/lanta_build_push_oracle_v4.sh 2>&1 | tee build_oracle_v4.log
