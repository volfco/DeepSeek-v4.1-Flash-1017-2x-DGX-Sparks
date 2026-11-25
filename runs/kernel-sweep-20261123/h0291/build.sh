#!/usr/bin/env bash
# h0291 (TF_DS_QD_PAIR_SHFL): the pair query kernel's exact source on the CPU, old against new. old.inc is cut from
# the production tree (tfref-kstack4, which must be at 2bb052f), new.inc from the sweep worktree; qd_pair_run.cpp runs
# tf_ds_fused_rope_qd_pair_kernel from both, the new tf_ds_fused_rope_qd_pair_shfl_kernel, and the old one-element
# tf_ds_fused_rope_qd_kernel (which writes the same bytes by 0241's contract) and compares output bytes.
set -euo pipefail
cd "$(dirname "$0")"
OLD=${OLD:-/home/colum/Workspace/tfref-kstack4}
NEW=${NEW:-/home/colum/Workspace/tfref-ksweep}
[[ $(git -C "$OLD" rev-parse --short HEAD) == 2bb052f ]] || { echo "old tree is not 2bb052f"; exit 1; }
spans=(
  "// E2M1 magnitude of an already-clamped t" "// One rotary element as the fused chains rotate it"
  'extern "C" __global__ void __launch_bounds__(128) tf_ds_fused_rope_qd_kernel(' "// The prompt indexer key (prompt.zig's kv-source layers)"
)
python3 ../emu/extract.py "$OLD/zig/kernels/cuda/dsv41_torch.cu" old.inc "${spans[@]}"
python3 ../emu/extract.py "$NEW/zig/kernels/cuda/dsv41_torch.cu" new.inc "${spans[@]}"
g++ -std=c++20 -O2 -ffp-contract=off -fno-strict-aliasing -Wall -Wno-unused-function -Wno-unused-variable -o qd_pair_run qd_pair_run.cpp
SHARDS=${SHARDS:-16}
{
    echo "old: $OLD @ $(git -C "$OLD" rev-parse --short HEAD)   new: $NEW @ $(git -C "$NEW" rev-parse --short HEAD)$(git -C "$NEW" diff --quiet || echo ' + working tree')"
    for i in $(seq 0 $((SHARDS - 1))); do ./qd_pair_run "$i" "$SHARDS" > "shard-$i.out" 2>&1 & done
    wait
    cat shard-*.out | python3 sum.py
    rm -f shard-*.out
} 2>&1 | tee run.log
