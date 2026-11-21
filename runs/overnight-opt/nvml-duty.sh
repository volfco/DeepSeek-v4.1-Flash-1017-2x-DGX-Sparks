#!/usr/bin/env bash
# Passive NVML sampling on both ranks. Timestamps are UTC and driver percentages are time-based.
set -euo pipefail
directory=$1
seconds=$2
mkdir -p "$directory"
options=(--query-gpu=timestamp,utilization.gpu,utilization.memory,clocks.sm,clocks.mem --format=csv,noheader,nounits --loop-ms=500)
sample() {
    local rank=$1
    shift
    local rc=0
    "$@" env TZ=UTC timeout "$seconds" stdbuf -oL nvidia-smi "${options[@]}" > "$directory/$rank-nvml.csv" || rc=$?
    [[ $rc = 0 || $rc = 124 ]]
}
sample head &
owner=$!
sample worker ssh -n -o BatchMode=yes 10.50.0.11 &
peer=$!
wait "$owner"
wait "$peer"
