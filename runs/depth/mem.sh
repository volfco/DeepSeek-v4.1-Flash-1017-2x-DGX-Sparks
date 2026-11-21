#!/bin/bash
# MemAvailable (kB) on both machines each second; GPU clock/temp/power every 10 s. Kill with: pkill -f runs/depth/mem.sh
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/depth
ssh -o BatchMode=yes -o ServerAliveInterval=15 10.50.0.11 'while :; do echo "$(date +%s) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 1; done' >> mem.worker 2>&1 &
ssh -o BatchMode=yes -o ServerAliveInterval=15 10.50.0.11 'while :; do echo "$(date -u +%FT%TZ) worker $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"; sleep 10; done' >> clocks.log 2>&1 &
i=0
while :; do
  echo "$(date +%s) $(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> mem.head
  if [ $((i % 10)) = 0 ]; then echo "$(date -u +%FT%TZ) head $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)" >> clocks.log; fi
  i=$((i+1)); sleep 1
done
