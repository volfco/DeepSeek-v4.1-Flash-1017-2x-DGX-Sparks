#!/usr/bin/env bash
# kall suite: 0265+0266+0268 together (and +0267 when k267 was exact) on 0f89e5a vs its all-off control.
R=runs/overnight-20261122
OUT=$R/kall REF=$R/sched2/ctl1.exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 PROBE_ARGS='--narrow --sampled' \
  BANNERS='TF_DS_DRAFT_HEAD_SCORED|TF_DS_MARKOV_FUSED' ARMS_FILE=$R/kall/arms.txt bash $R/arms.sh
