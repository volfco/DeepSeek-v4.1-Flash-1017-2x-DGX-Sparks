#!/usr/bin/env bash
# h0290 (TF_DS_DRAW_CUT): the sampled selectors' exact source on the CPU, old against new. old.inc is cut from the
# production tree (tfref-kstack4, which must be at 2bb052f), new.inc from the sweep worktree; draw_cut_run.cpp runs
# tf_ds_topk_select_kernel and tf_ds_draw_rows_kernel from both, and the new *_cut_kernel entry points, over the edge
# matrix under three lane orders and compares output bytes (and each list against an independent sort).
set -euo pipefail
cd "$(dirname "$0")"
OLD=${OLD:-/home/colum/Workspace/tfref-kstack4}
NEW=${NEW:-/home/colum/Workspace/tfref-ksweep}
[[ $(git -C "$OLD" rev-parse --short HEAD) == 2bb052f ]] || { echo "old tree is not 2bb052f"; exit 1; }
spans=(
  "__device__ unsigned tf_ds_block_argmax_v(const float* row, int n, float* out_v) {" "// logits [rows, n] fp32 (row stride ld), out [rows] u32; a row a block"
  "__device__ __forceinline__ uint64_t tf_ds_mix64(" "// grid (ceil(width / blockDim), streams)"
  "struct TfDsDrawRow {" "// TF_DS_TOPK_MERGE: the two ranks' lists"
)
python3 ../emu/extract.py "$OLD/zig/kernels/cuda/dsv41_ops.cu" old.inc "${spans[@]}"
python3 ../emu/extract.py "$NEW/zig/kernels/cuda/dsv41_ops.cu" new.inc "${spans[@]}"
g++ -std=c++20 -O2 -ffp-contract=off -fno-strict-aliasing -Wall -Wno-unused-function -Wno-unused-variable -o draw_cut_run draw_cut_run.cpp
SHARDS=${SHARDS:-16}
{
    echo "old: $OLD @ $(git -C "$OLD" rev-parse --short HEAD)   new: $NEW @ $(git -C "$NEW" rev-parse --short HEAD)$(git -C "$NEW" diff --quiet || echo ' + working tree')"
    for i in $(seq 0 $((SHARDS - 1))); do ./draw_cut_run "$i" "$SHARDS" > "shard-$i.out" 2>&1 & done
    wait
    cat shard-*.out | python3 ../emu/sum_stats.py "h0290 TF_DS_DRAW_CUT"
    rm -f shard-*.out
} 2>&1 | tee run.log
