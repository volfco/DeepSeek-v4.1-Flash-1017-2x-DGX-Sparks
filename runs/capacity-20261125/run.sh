#!/usr/bin/env bash
# capacity-20261125: the two runs in order, production restored and validated after each (arms.sh does that).
#   bash runs/capacity-20261125/run.sh          (from the recipe root on the head)
# Runtime-only: every arm is production's own image, so no build and no patch.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export WORKLOAD_CORPUS=runs/overnight-20261122/corpus.json
S=runs/capacity-20261125
REF=runs/deploy-10c9866-20261122/exact.json

run() { # name arms-file streams depths
  local n=$1 f=$2 st=$3 dp=$4
  mkdir -p "$S/$n"
  echo "$(date -u +%FT%TZ) $n: STREAMS=$st DEPTHS=$dp arms=$f"
  OUT="$S/$n" REF="$REF" ARMS_FILE="$f" SAMPLED=1 PROBE=0 WORKLOAD=1 WORKLOAD_SCRIPT=runs/kvreuse-size-20261122/workload.py \
    DEPTHS="$dp" STREAMS="$st" MAX_TOKENS=320 PREFILL_CELL=0 \
    bash "$S/arms.sh" > "$S/$n/run.log" 2>&1
  echo "$(date -u +%FT%TZ) $n rc=$?"
}

run s8  "$S/arms-s8.txt"  8  4096,32768
run s16 "$S/arms-s16.txt" 16 4096,32768
echo "$(date -u +%FT%TZ) done"
