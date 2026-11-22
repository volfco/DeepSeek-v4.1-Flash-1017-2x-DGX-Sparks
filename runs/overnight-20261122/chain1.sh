#!/usr/bin/env bash
# ringprobe -> deploy DRAFT_BLOCK_KEPT=2 -> rowcost sweep. Waits on log lines, not pgrep.
set -uo pipefail
R=runs/overnight-20261122
until grep -q ' finished$' $R/ringprobe/run.log 2>/dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) ringprobe finished"
IDLE_MIN=1 bash runs/deploy-kept2-20261122/deploy.sh > runs/deploy-kept2-20261122/run.log 2>&1; echo "$(date -u +%FT%TZ) deploy rc=$?"
tail -4 runs/deploy-kept2-20261122/run.log
grep -qx 'TF_DS_DRAFT_BLOCK_KEPT=2' .env || { echo "deploy did not land; stopping"; exit 1; }
OUT=$R/rowcost EXACT=0 WORKLOAD_SCRIPT=$R/rowcost/rowcost.py ARMS_FILE=$R/rowcost/arms.txt bash $R/arms.sh > $R/rowcost/run.log 2>&1
echo "$(date -u +%FT%TZ) rowcost done"
