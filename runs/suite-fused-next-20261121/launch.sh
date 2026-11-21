#!/usr/bin/env bash
# Wait until production has had no non-loopback request for 30 minutes, then start the suite (start.sh). Give up after
# 12 hours. Run detached on the head.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
O=runs/suite-fused-next-20261121
for i in $(seq 1 720); do
  n=$(docker logs --since 30m dsv41-1017-exl3-tf 2>&1 | grep 'POST /v1/' | grep -vc '\[tensorfold\] 127\.0\.0\.1 ')
  echo "$(date -u +%T) external requests in the last 30 min: $n" >> $O/launch.log
  if [ "$n" = 0 ]; then echo "$(date -u +%T) idle: starting" >> $O/launch.log; bash $O/start.sh >> $O/launch.log 2>&1; exit 0; fi
  sleep 60
done
echo "$(date -u +%T) never idle for 30 minutes in 12 hours: not started" >> $O/launch.log
