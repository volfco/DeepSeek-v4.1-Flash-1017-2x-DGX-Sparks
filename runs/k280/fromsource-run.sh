#!/usr/bin/env bash
# The from-source control (see decisions.md 2026-11-25). Mirrors k280/run.sh's settings so pass_ms is comparable.
set -euo pipefail
R=runs/overnight-20261122
mkdir -p runs/k280/fromsource
OUT=runs/k280/fromsource REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 \
  PROBE=1 PROBE_ARGS='--narrow --sampled --stream-bursts' WORKLOAD_SCRIPT=runs/k280/work.py WORKLOAD_SAMPLED=1 \
  BANNERS='Markov bias rows cached at load|the drafter pass forks the paced L2 prefetch' \
  ARMS_FILE=runs/k280/fromsource-arms.txt bash $R/arms.sh
