#!/usr/bin/env bash
# h0279: build and run the CPU harness for 0279 (TF_DS_IDX_PICK_WIDE). extract.py takes the kernel's exact source
# from the worktree's dsv41_ops.cu into pick_wide_kernel.inc; pick_wide_run.cpp emulates it (fibers, blocks
# sequential) against the independent sort reference; gate.zig checks the host gate's arm decisions against the
# worktree's real modules (including the 65,537 refusal -> unfused fallback). run.log holds both runs.
set -euo pipefail
cd "$(dirname "$0")"
WT=${WT:-/home/colum/Workspace/tfref-k0279}
ZIG=/home/colum/.local/zig-0.17.0/zig
python3 extract.py "$WT/zig/kernels/cuda/dsv41_ops.cu" pick_wide_kernel.inc
g++ -O2 -std=c++17 -Wall -o pick_wide_run pick_wide_run.cpp
{
    ./pick_wide_run
    "$ZIG" build run -Dwt="$WT"
} 2>&1 | tee run.log
