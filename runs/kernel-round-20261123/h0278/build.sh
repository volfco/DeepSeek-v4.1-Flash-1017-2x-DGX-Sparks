#!/bin/sh
# h0278: build and run the CPU-emulation harness against the worktree's real kernel source.
# The exercised write is tf_ds_hc_mean4_kernel (extracted verbatim from the worktree's
# dsv41_torch.cu — the splice.py trick, p0194), run as a sequential virtual-thread loop.
# No new kernel exists for 0278 (hcMean4's row stride suffices), so the emulation covers the
# write/copy LAYOUTS: planar + completion copies vs the packed write, against an independent
# reference written from BRIEF-0278's spec. No OS threads; blocks run sequentially.
set -eu
WT=${WT:-/home/colum/Workspace/tfref-k0278}
cd "$(dirname "$0")"

# extract the exercised kernel's source from the worktree, including its spec comment
sed -n '/^\/\/ A DSpark tap as the prompt forward takes it/,/^}/p' \
    "$WT/zig/kernels/cuda/dsv41_torch.cu" > hc_mean4.inc
grep -q "tf_ds_hc_mean4_kernel" hc_mean4.inc

g++ -O2 -std=c++17 -Wall -Wextra harness.cpp -o harness

{
    echo "== tf_ds_hc_mean4_kernel extracted from $WT/zig/kernels/cuda/dsv41_torch.cu =="
    cat hc_mean4.inc
    echo
    echo "== harness: planar + completion copies vs packed write vs independent reference =="
    ./harness
} 2>&1 | tee run.log
