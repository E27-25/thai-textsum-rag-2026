#!/bin/bash
WATCH_PID=3221337
echo "=== [$(date)] Watching DPO v4 PID $WATCH_PID ==="
while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 30; done
echo "=== [$(date)] DPO done. Starting Oracle v4 ==="
df -h /dev/shm
cd /lustrefs/disk/project/zz991000-zdeva/zz991016/Arther
nohup bash lanta_build_push_oracle_v4_rerank.sh > build_oracle_v4_rerank4.log 2>&1 &
echo "Oracle v4 PID: $!"
