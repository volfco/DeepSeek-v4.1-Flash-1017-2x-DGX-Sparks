#!/usr/bin/env bash
# k296 suite (2026-11-24): sched lane 0294/0295/0296 (runs/lane-sched-20261123/REPORT.md). From the recipe root on the head:
#   mkdir -p runs/k296 && setsid nohup bash runs/k296/run.sh > runs/k296/run.log 2>&1 < /dev/null &
# Exact vs production's reference, sampled vs sched2 ctl1, the sampled narrow probe (drafts/rounds/accepted), the
# kvreuse-size workload (4K/32K/128K x 1/4 streams, GPU clock) — whose 32K/128K cells feed 0296's counter.
R=runs/overnight-20261122
OUT=runs/k296 REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 \
  PROBE_ARGS='--narrow --sampled' BANNERS='TF_DS_CONF_LEADER_ONLY=1' ARMS_FILE=runs/k296/arms.txt bash runs/k296/arms.sh
