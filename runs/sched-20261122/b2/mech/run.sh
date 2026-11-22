#!/usr/bin/env bash
R=runs/overnight-20261122
OUT=runs/sched-20261122/b2/mech EXACT=0 SAMPLED=0 PROBE=0 WORKLOAD_SCRIPT=runs/sched-20261122/b2/load.py \
  BANNERS='TF_DS_RING_TRACE' ARMS_FILE=runs/sched-20261122/b2/mech/arms.txt bash $R/arms.sh
