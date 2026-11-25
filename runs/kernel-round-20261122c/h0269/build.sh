#!/usr/bin/env bash
# h0269: TF_DS_DRAFT_TAPS_DIRECT's drafter input row addressing on the CPU against an independent reference.
# 0269 adds no .cu kernel (the arm skips the gather launch and mainKeys reads rd.taps rows in place at the same row
# stride dr.at had), so there is nothing to splice out of the .cu: the harness emulates the arm's row addressing
# instead (see draft_taps_direct_run.cpp's header). stubs/ is a copy of the round's fixed CUDA syntax stubs (the
# shared cuda-syntax-stubs tree predates the base and fails on it; see report-0269.md); it also holds the syntax
# check this round must pass:
#   stubs/check.sh <worktree>/zig/kernels/cuda
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
g++ -std=c++17 -O1 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -o "$here/draft_taps_direct_run" "$here/draft_taps_direct_run.cpp"
"$here/draft_taps_direct_run"
