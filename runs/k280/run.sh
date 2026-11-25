#!/usr/bin/env bash
# Round 0280-0283's main suite: 0281 (cached Markov bias rows) and 0282 (the drafter pass's L2 prefetch) against the
# new image's own off path. Run from the recipe root on the head, production idle. Ten arms x about 10 min, then the
# restore: about 1 h 45 min. Restarts the server each arm and restores production from .env at the end, as every
# arms.sh suite does. See runs/k280/README.md for the gates and the scoring.
#   mkdir -p runs/k280/main && setsid nohup bash runs/k280/run.sh > runs/k280/main/run.log 2>&1 < /dev/null &
# BANNERS is the two load/engagement lines' own text (no quotes or $ in it: arms.sh splices it into an ssh command).
# Expected banner counts on each rank: 0 in ctl arms, 1 in every mc and dpf arm.
set -euo pipefail
R=runs/overnight-20261122
mkdir -p runs/k280/main
OUT=runs/k280/main REF=runs/deploy-10c9866-20261122/exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 \
  PROBE=1 PROBE_ARGS='--narrow --sampled --stream-bursts' WORKLOAD_SCRIPT=runs/k280/work.py WORKLOAD_SAMPLED=1 \
  BANNERS='Markov bias rows cached at load|the drafter pass forks the paced L2 prefetch' \
  ARMS_FILE=runs/k280/arms.txt bash $R/arms.sh
