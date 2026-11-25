#!/usr/bin/env bash
# The pre-soak half of the ENGRAM_LEAD test: cold prefill at 32K/96K (PREFILL_CELL) and decode at 4K/32K x 1/4,
# ABBA, exact panel as the gate. The soak (>=1.5M uncached prefill tokens) and the hour of process age from
# to-benchmark.md's evaluation order are NOT covered here.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export WORKLOAD_CORPUS=runs/overnight-20261122/corpus.json
S=runs/engram-lead-20261125
mkdir -p "$S/main"
echo "$(date -u +%FT%TZ) start"
OUT="$S/main" REF=runs/deploy-10c9866-20261122/exact.json ARMS_FILE="$S/arms.txt" \
  SAMPLED=0 PROBE=0 WORKLOAD=1 WORKLOAD_SCRIPT=runs/kvreuse-size-20261122/workload.py \
  PREFILL_CELL=1 BANNERS='engram-lead' \
  bash runs/overnight-20261122/arms.sh > "$S/main/run.log" 2>&1
echo "$(date -u +%FT%TZ) rc=$?"
tail -3 "$S/main/run.log"
