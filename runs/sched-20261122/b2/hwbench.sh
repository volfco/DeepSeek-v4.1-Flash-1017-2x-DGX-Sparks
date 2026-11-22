#!/usr/bin/env bash
# Round B2: stop the engine, run hwbench.py on each Spark alone (same image, same cpuset as serving), 3 repeats each,
# then both at once (to see whether either box slows when the other runs), then restore production and validate.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
D=runs/sched-20261122/b2; W=colum@10.50.0.11; IMG=tensorfold-dsv41:bench-fused-next-10c9866
ssh -n -o BatchMode=yes $W "mkdir -p /tmp/b2"; scp -q $D/hwbench.py $W:/tmp/b2/
ARGS="--rm --gpus all --ipc=host --cpuset-cpus 15-19 --entrypoint python3"
./stop.sh > $D/hw.stop.log 2>&1; echo "$(date -u +%T) engine stopped"
for k in 1 2 3; do
  docker run $ARGS -v $PWD/$D:/b $IMG -I /b/hwbench.py 2>/dev/null | tail -1 | sed "s/^/r0 alone $k: /"
  ssh -n -o BatchMode=yes $W "docker run $ARGS -v /tmp/b2:/b $IMG -I /b/hwbench.py 2>/dev/null | tail -1" | sed "s/^/r1 alone $k: /"
done
for k in 1 2; do
  ssh -n -o BatchMode=yes $W "docker run $ARGS -v /tmp/b2:/b $IMG -I /b/hwbench.py 2>/dev/null | tail -1" | sed "s/^/r1 both $k: /" &
  docker run $ARGS -v $PWD/$D:/b $IMG -I /b/hwbench.py 2>/dev/null | tail -1 | sed "s/^/r0 both $k: /"; wait
done
echo "$(date -u +%T) restoring production"
PREPARE=0 ./start.sh restart > $D/hw.restore.log 2>&1 || grep -q 'the smoke test request failed' $D/hw.restore.log || echo "restore restart failed"
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/hw.validate.json \
  --compare runs/matrix/base.validate.json > $D/hw.validate.log 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
echo "$(date -u +%T) finished"
