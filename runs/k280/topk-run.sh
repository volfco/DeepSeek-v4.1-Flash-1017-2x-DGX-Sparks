#!/usr/bin/env bash
# The 0261 on/off pair on sampled rounds (no new image). Run from the recipe root on the head, production idle.
# About 4 arms x ~12 min. Restarts the server per arm and restores production at the end, as every arms.sh suite does.
#   bash runs/k280/topk-run.sh 2>&1 | tee runs/k280/topk/run.log
# Score: python3 runs/overnight-20261122/table_gpu.py runs/k280/topk --control on1,on2 --test off1,off2
#   (control = shipped 0261; a positive "test vs control" means the fallback is slower, i.e. 0261 is a gain)
# Gates: exact IDENTICAL in all arms; sampled IDENTICAL (the two paths draw the same tokens); every one-stream
#   request's sha in <arm>.json equal across arms; the banner count is 1 in on arms and 0 in off arms.
set -euo pipefail
R=runs/overnight-20261122
mkdir -p runs/k280/topk
OUT=runs/k280/topk REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 \
  PROBE=1 PROBE_ARGS='--narrow --sampled --stream-bursts' WORKLOAD_SCRIPT=runs/k280/work.py WORKLOAD_SAMPLED=1 \
  BANNERS='TF_DS_TOPK_MERGE' ARMS_FILE=runs/k280/topk-arms.txt bash $R/arms.sh
