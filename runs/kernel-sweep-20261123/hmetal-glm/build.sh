#!/usr/bin/env bash
# hmetal-glm: evidence for unverified/0001-glm-index-select-cut.patch. The old kernel text is glm_glue.metal at
# 2bb052f (git show, from tfref-kstack4); the new text is that file with the patch applied in a scratch directory
# (nothing is written to any worktree). metal2c.py rewrites both into C++ for emu/emu.h; glm_select_run.cpp runs
# glm_index_select old, new with cut 0, and new with cut 1 over the edge matrix under three lane orders and compares
# the key lists' bytes. This checks the kernel's source logic on a CPU. It is NOT a Metal compile: nothing here has
# been through Apple's compiler.
set -euo pipefail
cd "$(dirname "$0")"
OLD=${OLD:-/home/colum/Workspace/tfref-kstack4}
[[ $(git -C "$OLD" rev-parse --short HEAD) == 2bb052f ]] || { echo "old tree is not 2bb052f"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in zig/kernels/metal/glm_glue.metal zig/src/families/glm/engine.zig zig/src/families/glm/forward.zig; do
    mkdir -p "$tmp/old/$(dirname "$f")" "$tmp/new/$(dirname "$f")"
    git -C "$OLD" show "2bb052f:$f" > "$tmp/old/$f"
    cp "$tmp/old/$f" "$tmp/new/$f"
done
(cd "$tmp/new" && git apply "$OLDPWD/../unverified/0001-glm-index-select-cut.patch")
python3 metal2c.py "$tmp/old/zig/kernels/metal/glm_glue.metal" old.inc
python3 metal2c.py "$tmp/new/zig/kernels/metal/glm_glue.metal" new.inc
g++ -std=c++20 -O2 -ffp-contract=off -fno-strict-aliasing -Wall -Wno-unused-function -Wno-unused-variable -o glm_select_run glm_select_run.cpp
SHARDS=${SHARDS:-16}
{
    echo "old: glm_glue.metal @ 2bb052f   new: the same with unverified/0001-glm-index-select-cut.patch applied"
    for i in $(seq 0 $((SHARDS - 1))); do ./glm_select_run "$i" "$SHARDS" > "shard-$i.out" 2>&1 & done
    wait
    cat shard-*.out | python3 sum.py
    rm -f shard-*.out
} 2>&1 | tee run.log
