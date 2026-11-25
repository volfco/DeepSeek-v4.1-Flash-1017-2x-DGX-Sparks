#!/usr/bin/env bash
# h0268: TF_DS_DRAFT_QKV_NORM's host check - the joint _q_kv_norm call's arguments against the two separate calls',
# through the kernel-0268 worktree's real tri_norm wrappers (no GPU; kernel arithmetic parity is GPU-gate only).
# Copies the worktree's sources at run time (so the check always runs the committed tree), then zig test.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
wt=${WT:-/home/colum/Workspace/tfref-k0268}
zig=${ZIG:-/home/colum/.local/zig-0.17.0/zig}
mkdir -p "$here/src/fixtures"
cp "$wt/zig/src/families/dsv41/tri.zig" "$wt/zig/src/families/dsv41/tri_norm.zig" "$wt/zig/src/families/dsv41/draft_qkv_norm.zig" "$here/src/"
cp "$wt/zig/src/families/dsv41/fixtures/launches.json" "$here/src/fixtures/"
printf 'pub const with_kernels = false;\npub const has_dsv41_port = false;\n' > "$here/src/kernel_options.zig"
cd "$here"
"$zig" test \
    --dep cuda -Mroot=src/harness.zig \
    --dep stagger --dep kernel_options -Mcuda="$wt/zig/src/cuda/root.zig" \
    -Mstagger="$wt/zig/src/core/stagger.zig" \
    -Mkernel_options=src/kernel_options.zig \
    -lc 2>&1 | tee "$here/run.log"
