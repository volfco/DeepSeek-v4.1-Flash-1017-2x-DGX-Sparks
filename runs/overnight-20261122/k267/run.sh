#!/usr/bin/env bash
R=runs/overnight-20261122
OUT=$R/k267 REF=$R/sched2/ctl1.exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 \
  BANNERS='TF_DS_MARKOV_FUSED' ARMS_FILE=$R/k267/arms.txt bash $R/arms.sh
