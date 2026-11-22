#!/usr/bin/env bash
# suite-fused-next2-20261121 launch (on the head, from the recipe root, detached): wait until production has had no
# non-loopback request for 30 minutes (give up after 12 hours), snapshot both ranks' production env, capture the
# guarded 0215 small-row crossover timings during the idle window, then start the suite. The image is built and
# shipped beforehand by preflight.sh on the workstation.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
O=runs/suite-fused-next2-20261121
W=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
n=1
for i in $(seq 1 720); do
  n=$(docker logs --since 30m dsv41-1017-exl3-tf 2>&1 | grep 'POST /v1/' | grep -vc '\[tensorfold\] 127\.0\.0\.1 ')
  echo "$(date -u +%T) external requests in the last 30 min: $n" >> $O/launch.log
  if [ "$n" = 0 ]; then echo "$(date -u +%T) idle: starting" >> $O/launch.log; break; fi
  sleep 60
done
if [ "$n" != 0 ]; then echo "$(date -u +%T) never idle for 30 minutes in 12 hours: not started" >> $O/launch.log; exit 1; fi

echo "$(date -u +%T) snapshotting production env (both ranks)" >> $O/launch.log
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf > $O/production.rank0.env
ssh -o BatchMode=yes "$W" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf" > $O/production.rank1.env

echo "$(date -u +%T) refreshing 0215 parity state (~3 s, 36 cases; timing-burst refuses without it)" >> $O/launch.log
if (cd ~/tf-bench/pre-bench-20261121/crossover && flock ~/tf-bench/gpu.lock python3 measure.py --mode parity --cubin bin/ops.cubin --json results.json) >> $O/launch.log 2>&1; then
  echo "$(date -u +%T) capturing guarded 0215 crossover timings (~2 min)" >> $O/launch.log
  if (cd ~/tf-bench/pre-bench-20261121/crossover && python3 run_timing_loop.py --cubin bin/ops.cubin --json results.json --budget-s 120) >> $O/launch.log 2>&1; then
    rsync -a ~/tf-bench/pre-bench-20261121/crossover/results.json ~/tf-bench/pre-bench-20261121/crossover/timing.log \
      ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/pre-bench-20261121/crossover-0215/ || true
  else
    echo "$(date -u +%T) 0215 timing capture stayed busy or failed: continuing (small_nbk stays 6144)" >> $O/launch.log
  fi
else
  echo "$(date -u +%T) 0215 parity state refresh FAILED (environmental, or a byte mismatch to inspect): skipping 0215 timings, continuing to the suite" >> $O/launch.log
fi

echo "$(date -u +%T) starting suite" >> $O/launch.log
bash $O/start.sh >> $O/launch.log 2>&1
