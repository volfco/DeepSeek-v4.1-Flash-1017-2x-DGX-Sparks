#!/usr/bin/env bash
R=runs/overnight-20261122
OUT=runs/k278 REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=0 \
  WORKLOAD_SCRIPT=runs/k278/work.py BANNERS='TF_DS_TAPS_PACKED|TF_DS_IDX_PICK_WIDE' ARMS_FILE=runs/k278/arms.txt bash $R/arms.sh
