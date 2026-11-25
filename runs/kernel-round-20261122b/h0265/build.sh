#!/usr/bin/env bash
# h0265: TF_DS_DRAFT_HEAD_SCORED's row selection / packing on the CPU against an independent reference.
# 0265 adds no .cu kernel (the packing reuses the kit's _collapse/_rmsnorm and the head's glinear at fewer rows), so
# there is nothing to splice out of the .cu: the harness emulates the call sites' per-thread row semantics instead.
# stubs/ is a copy of the round's fixed CUDA syntax stubs (the shared cuda-syntax-stubs tree predates the base and
# fails on it; see report-0265.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
g++ -std=c++17 -O1 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -o "$here/draft_head_scored_run" "$here/draft_head_scored_run.cpp"
"$here/draft_head_scored_run"
