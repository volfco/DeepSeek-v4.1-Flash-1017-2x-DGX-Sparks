#!/usr/bin/env bash
R=runs/overnight-20261122
OUT=$R/k272 REF=$R/sched2/ctl1.exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 PROBE_ARGS='--narrow --sampled --stream-bursts' \
  BANNERS='TF_DS_COPY_MIN|TF_DS_RING_TRACE|stretch capture' ARMS_FILE=$R/k272/arms.txt bash $R/arms.sh
