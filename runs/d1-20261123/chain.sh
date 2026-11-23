#!/usr/bin/env bash
# stop production, build both ranks' D1 weight files, then the D1 ABBA suite (arms.sh restores production at its end)
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
echo "$(date -u +%T) stopping production"; ./stop.sh > runs/d1-20261123/stop.log 2>&1; echo "stop rc=$?"
echo "$(date -u +%T) building caches"; bash runs/d1-20261123/build.sh caches > runs/d1-20261123/caches.log 2>&1; rc=$?
echo "caches rc=$rc"; tail -8 runs/d1-20261123/caches.log
if (( rc != 0 )); then echo "$(date -u +%T) cache build failed: restoring production"; PREPARE=0 ./start.sh restart > runs/d1-20261123/restore.log 2>&1; echo "restore rc=$?"; echo finished; exit 1; fi
echo "$(date -u +%T) suite"; bash runs/d1-20261123/run.sh > runs/d1-20261123/run.log 2>&1
grep -E "arm |exact vs|sampled vs|banners|restored|passed" runs/d1-20261123/run.log; echo "$(date -u +%T) finished"
