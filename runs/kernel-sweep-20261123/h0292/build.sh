#!/usr/bin/env bash
# h0292 (TF_DS_BLOCK_MAX8): the block-maxima kernels' exact source on the CPU. old.inc is tf_ds_block_max_kernel cut
# from the production tree (tfref-kstack4, which must be at 2bb052f); new.inc is the sweep worktree's
# tf_ds_block_max_kernel (must be unchanged) and tf_ds_block_max8_kernel. block_max_run.cpp launches all three over the
# edge matrix on the launcher's grid (and on a clipped grid) and compares output bytes.
set -euo pipefail
cd "$(dirname "$0")"
OLD=${OLD:-/home/colum/Workspace/tfref-kstack4}
NEW=${NEW:-/home/colum/Workspace/tfref-ksweep}
[[ $(git -C "$OLD" rev-parse --short HEAD) == 2bb052f ]] || { echo "old tree is not 2bb052f"; exit 1; }
python3 ../emu/extract.py "$OLD/zig/kernels/cuda/dsv41_ops.cu" old.inc \
  'extern "C" __global__ void tf_ds_block_max_kernel(' "// The pool from a row's top blocks idx [rows, k]"
python3 ../emu/extract.py "$NEW/zig/kernels/cuda/dsv41_ops.cu" new.inc \
  'extern "C" __global__ void tf_ds_block_max_kernel(' "// The pool from a row's top blocks idx [rows, k]"
g++ -std=c++20 -O2 -ffp-contract=off -fno-strict-aliasing -Wall -Wno-unused-function -Wno-unused-variable -o block_max_run block_max_run.cpp
{
    echo "old: $OLD @ $(git -C "$OLD" rev-parse --short HEAD)   new: $NEW @ $(git -C "$NEW" rev-parse --short HEAD)$(git -C "$NEW" diff --quiet || echo ' + working tree')"
    ./block_max_run
} 2>&1 | tee run.log
