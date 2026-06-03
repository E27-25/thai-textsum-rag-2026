#!/bin/bash
WATCH_PID=127982
echo "=== [$(date)] Watching PID $WATCH_PID (14B SFT-v3 build) ==="
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 30
done
echo "=== [$(date)] PID $WATCH_PID done. Waiting 120s before starting Oracle v4 (let SFT-v3 push clear space) ==="
sleep 120
df -h /dev/shm
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther
nohup bash lanta_build_push_oracle_v4_rerank.sh > build_oracle_v4_rerank.log 2>&1 &
echo "Oracle v4 rerank build started (PID $!), log: build_oracle_v4_rerank.log"
