#!/bin/bash
# Wait for 14B SFT-v3 build (PID 127982) to finish, then start DPO build
WATCH_PID=127982
echo "=== [$(date)] Watching PID $WATCH_PID (14B SFT-v3 build) ==="
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 30
done
echo "=== [$(date)] PID $WATCH_PID done. Starting DPO build ==="
df -h /dev/shm
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther
nohup bash lanta_build_push_dpo.sh > build_dpo_v1.log 2>&1 &
echo "DPO build started (PID $!), log: build_dpo_v1.log"
