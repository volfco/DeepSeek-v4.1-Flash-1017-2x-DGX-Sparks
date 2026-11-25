#!/usr/bin/env bash
# h0276: build and run the CPU harness for 0276 (TF_DS_CACHE_REPLACE_STALE) against the worktree's
# prompt_cache.zig; run.log holds the run (the engagement banner appears once, on stderr).
set -euo pipefail
cd "$(dirname "$0")"
ZIG=/home/colum/.local/zig-0.17.0/zig
PC=${PC:-/home/colum/Workspace/tfref-k0276/zig/src/core/prompt_cache.zig}
$ZIG build run -Dpc="$PC" 2>&1 | tee run.log
banners=$(grep -c 'dsv41: TF_DS_CACHE_REPLACE_STALE:' run.log || true)
echo "engagement banners in run.log: $banners (expected 1)" | tee -a run.log
test "$banners" -eq 1
