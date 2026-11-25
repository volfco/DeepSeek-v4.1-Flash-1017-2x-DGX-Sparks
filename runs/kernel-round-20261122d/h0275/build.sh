#!/bin/sh
# h0275: build and run the CPU-emulation harness against the worktree's real code.
# The harness is pure host (sequential; no threads, no GPU), so no block emulation is involved.
set -e
WT=/home/colum/Workspace/tfref-k0275
ZIG=/home/colum/.local/zig-0.17.0/zig
cd "$(dirname "$0")"
# extract the exercised sources from the worktree: the gate's module and the proposer it gates
cp "$WT/zig/src/families/dsv41/copy_min.zig" .
cp "$WT/zig/src/core/lanes/proposer.zig" .
echo "== zig test copy_min.zig (the module's own tests)"
"$ZIG" test copy_min.zig
echo "== zig run harness.zig"
"$ZIG" run harness.zig
