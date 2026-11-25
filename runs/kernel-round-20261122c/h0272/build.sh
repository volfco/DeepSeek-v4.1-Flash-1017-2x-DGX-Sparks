#!/usr/bin/env bash
# h0272's local gate: the CPU emulation of TF_DS_RING_TRACE's device-side record ring against an independent
# reference (trace_emu.cpp, with the shipped logic spliced in by splice.py), the summarizer on a synthetic two-rank
# trace with known injected delays (gen_trace.py -> summarize.py -> check_attribution.py), and the kernels' call-site
# wiring greps. Usage: ./build.sh [WORKTREE] (default /home/colum/Workspace/tfref-k0272). Output: run.log.
set -euo pipefail
cd "$(dirname "$0")"
WORKTREE=${1:-/home/colum/Workspace/tfref-k0272}

echo "== h0272: TF_DS_RING_TRACE CPU-emulation harness =="
echo "worktree: $WORKTREE"

echo "-- call-site wiring (the kernels must call every record helper exactly where the brief lists it)"
grep -q 'tf_tr_claim(trace, q, t0)' "$WORKTREE/zig/kernels/cuda/dsv41/rdma_gather.cu"
grep -q 'tf_tr_publish(trace, q, now_ns()' "$WORKTREE/zig/kernels/cuda/dsv41/rdma_gather.cu"
grep -q 'tf_tr_entry(trace, q, now_ns())' "$WORKTREE/zig/kernels/cuda/dsv41/rdma_gather.cu"
grep -q 'tf_tr_flag(trace, q, t2)' "$WORKTREE/zig/kernels/cuda/dsv41/rdma_gather.cu"
grep -q 'tf_tr_done(trace, q, now_ns())' "$WORKTREE/zig/kernels/cuda/dsv41/rdma_gather.cu"
grep -q 'tf_tr_claim(trace, q, t0)' "$WORKTREE/zig/kernels/cuda/dsv41_shard.cu"
grep -q 'tf_tr_publish(trace, q, tf_ds_now_ns()' "$WORKTREE/zig/kernels/cuda/dsv41_shard.cu"
grep -q 'tf_tr_entry(trace, q, tf_ds_now_ns())' "$WORKTREE/zig/kernels/cuda/dsv41_shard.cu"
grep -q 'tf_tr_flag(trace, q, t2)' "$WORKTREE/zig/kernels/cuda/dsv41_shard.cu"
grep -q 'tf_tr_done(trace, q, tf_ds_now_ns())' "$WORKTREE/zig/kernels/cuda/dsv41_shard.cu"
echo "ok: rdma_gather.cu and dsv41_shard.cu call claim/publish/entry/flag/done at the listed points"

echo "-- splice: extract the shipped record logic"
python3 splice.py "$WORKTREE/zig/kernels/cuda/dsv41/ring_trace_rec.cuh"

echo "-- compile and run the record-ring emulation against the independent reference"
g++ -std=c++17 -O1 -Wall -Wextra -o trace_emu trace_emu.cpp
./trace_emu

echo "-- the summarizer on a synthetic two-rank trace with known injected delays"
python3 gen_trace.py .
python3 summarize.py --json synth_rank0.jsonl synth_rank1.jsonl > synth_summary.json
python3 summarize.py synth_rank0.jsonl synth_rank1.jsonl
python3 check_attribution.py < synth_summary.json

echo "h0272: ok"
