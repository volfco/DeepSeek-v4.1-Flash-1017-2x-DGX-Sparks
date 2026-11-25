#!/usr/bin/env bash
# mixed-20261125: the prefill/decode interference cell, control first then the knob sweep. Production restored and
# validated after each run (arms.sh does that).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export WORKLOAD_CORPUS=runs/overnight-20261122/corpus.json
S=runs/mixed-20261125
REF=runs/deploy-10c9866-20261122/exact.json
mkdir -p "$S/control" "$S/sweep"

echo "$(date -u +%FT%TZ) control: 7 decoders alone"
MIXED_CONTROL=1 OUT="$S/control" REF="$REF" ARMS_FILE="$S/arms-control.txt" SAMPLED=0 PROBE=0 WORKLOAD=1 \
  WORKLOAD_SCRIPT="$S/mixed.py" BANNERS='the L2 prefetch waits|interleave' \
  bash runs/overnight-20261122/arms.sh > "$S/control/run.log" 2>&1
echo "$(date -u +%FT%TZ) control rc=$?"

echo "$(date -u +%FT%TZ) sweep: 9 knob pairs"
OUT="$S/sweep" REF="$REF" ARMS_FILE="$S/arms-mixed.txt" SAMPLED=0 PROBE=0 WORKLOAD=1 \
  WORKLOAD_SCRIPT="$S/mixed.py" BANNERS='the L2 prefetch waits|interleave' \
  bash runs/overnight-20261122/arms.sh > "$S/sweep/run.log" 2>&1
echo "$(date -u +%FT%TZ) sweep rc=$?"
echo "$(date -u +%FT%TZ) done"
