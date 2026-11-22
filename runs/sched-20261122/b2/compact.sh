#!/usr/bin/env bash
# Round B2: is rank 0's small-row GEMV penalty memory fragmentation? Engine stopped; hwbench both boxes (before);
# compact the head only (sync, drop caches, compact_memory); hwbench both (after); buddyinfo each step. Restore + validate.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
D=runs/sched-20261122/b2; W=colum@10.50.0.11; IMG=tensorfold-dsv41:bench-fused-next-10c9866
ARGS="--rm --gpus all --ipc=host --cpuset-cpus 15-19 --entrypoint python3"
bench() { for k in 1 2; do
  docker run $ARGS -v $PWD/$D:/b $IMG -I /b/hwbench.py 2>/dev/null | tail -1 | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print('r0 $1 $k', d['gemv_rows1_us_per_pair'], d['gemv_rows7_us_per_pair'], d['gemv_rows28_us_per_pair'], d['d2d_copy_GBps'])"
  ssh -n -o BatchMode=yes $W "docker run $ARGS -v /tmp/b2:/b $IMG -I /b/hwbench.py 2>/dev/null | tail -1" | python3 -c "import json,sys;d=json.loads(sys.stdin.read());print('r1 $1 $k', d['gemv_rows1_us_per_pair'], d['gemv_rows7_us_per_pair'], d['gemv_rows28_us_per_pair'], d['d2d_copy_GBps'])"
done; }
bi() { echo "$1 head:   $(grep Normal /proc/buddyinfo | awk '{$1=$2=$3=$4="";print}')"; echo "$1 worker: $(ssh -n -o BatchMode=yes $W "grep Normal /proc/buddyinfo" | awk '{$1=$2=$3=$4="";print}')"; }
./stop.sh > $D/cp.stop.log 2>&1; echo "$(date -u +%T) engine stopped"; sleep 5
bi stopped; echo "(cols: rows1 rows7 rows28 us/pair, d2d GB/s)"; bench before
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'; sleep 5
bi compacted-head; bench after
echo "$(date -u +%T) restoring production (head compacted)"
PREPARE=0 ./start.sh restart > $D/cp.restore.log 2>&1 || grep -q 'the smoke test request failed' $D/cp.restore.log || echo "restore restart failed"
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/cp.validate.json \
  --compare runs/matrix/base.validate.json > $D/cp.validate.log 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
bi engine-up
echo "$(date -u +%T) finished"
