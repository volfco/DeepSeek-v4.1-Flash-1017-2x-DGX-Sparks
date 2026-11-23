#!/usr/bin/env bash
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
until grep -q finished runs/d1-20261123/chain.log; do sleep 30; done
while pgrep -f "overnight-20261122/arms.sh" >/dev/null; do sleep 10; done
cp runs/nccl-screen-20261123/arms.sh.next runs/overnight-20261122/arms.sh
echo "$(date -u +%T) screen start"; bash runs/nccl-screen-20261123/run.sh > runs/nccl-screen-20261123/run.log 2>&1
echo "$(date -u +%T) finished"
