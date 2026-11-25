#!/usr/bin/env bash
# 0283's one-at-a-time screen of the L2 prefetch knobs (like runs/nccl-screen-20261123, with the round-time workload
# on sampled cells so the GPU clock scores it). Run from the recipe root on the head, production idle, after
# runs/k280/run.sh's ctl arms have shown the new image exact. Eight arms x about 6 min, then the restore: about 50 min.
#   mkdir -p runs/k280/l2 && setsid nohup bash runs/k280/l2-run.sh > runs/k280/l2/run.log 2>&1 < /dev/null &
# No probe and no sampled panel: the screen ranks knobs, the main suite's method confirms a winner. The exact panel
# stays on (EXACT=0 drops it, about a minute an arm): it costs little and catches a build or env mistake.
# Expected banner counts on each rank: 0 in ctl arms, 1 in every knob arm.
set -euo pipefail
R=runs/overnight-20261122
mkdir -p runs/k280/l2
OUT=runs/k280/l2 REF=runs/deploy-10c9866-20261122/exact.json EXACT=${EXACT:-1} SAMPLED=0 PROBE=0 \
  WORKLOAD_SCRIPT=runs/k280/work.py WORKLOAD_SAMPLED=1 BANNERS='the L2 prefetch waits' \
  ARMS_FILE=runs/k280/l2-arms.txt bash $R/arms.sh
