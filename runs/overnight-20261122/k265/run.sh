#!/usr/bin/env bash
# k265 suite: exact + sampled identity vs production's sched2 ctl1, the sampled decode probe, the round-time workload.
R=runs/overnight-20261122
OUT=$R/k265 REF=$R/sched2/ctl1.exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 PROBE_ARGS='--narrow --sampled' \
  BANNERS='TF_DS_DRAFT_HEAD_SCORED|TF_DS_COMPRESS2_PAIR_ONCE|TF_DS_DRAFT_QKV_NORM' ARMS_FILE=$R/k265/arms.txt bash $R/arms.sh
