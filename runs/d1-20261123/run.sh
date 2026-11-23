#!/usr/bin/env bash
R=runs/overnight-20261122
OUT=runs/d1-20261123 REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 \
  PROBE_ARGS='--narrow --sampled --stream-bursts' BANNERS='rank-cache|weight file' ARMS_FILE=runs/d1-20261123/arms.txt bash $R/arms.sh
