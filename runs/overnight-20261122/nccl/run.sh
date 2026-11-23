#!/usr/bin/env bash
R=runs/overnight-20261122
OUT=$R/nccl REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 PREFILL_CELL=1 \
  PROBE_ARGS='--narrow --sampled --stream-bursts' BANNERS='SOLO_OVERHEAD|NCCL_MAX_CTAS' ARMS_FILE=$R/nccl/arms.txt bash $R/arms.sh
