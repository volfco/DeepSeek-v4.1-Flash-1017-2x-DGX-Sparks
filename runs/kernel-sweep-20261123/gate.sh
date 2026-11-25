#!/usr/bin/env bash
# gate.sh <NNNN>: the local gate for one patch of the sweep, run from the worktree as it stands. Logs under gate-<NNNN>/.
# 1. zig build test (rerun once on failure: lane_host.test.prefill and one spill-tier timing test are known flakes)
# 2. zig build (compiles every executable)
# 3. cuda-syntax-stubs/check.sh: its output must equal the base tree's (base/check.log, paths aside)
# 4. check_all.sh: the same stub check over every .cu, the error set diffed against base/check_all.txt
set -uo pipefail
n=${1:?usage: gate.sh <NNNN>}
here=$(cd "$(dirname "$0")" && pwd)
wt=/home/colum/Workspace/tfref-ksweep
zig=/home/colum/.local/zig-0.17.0/zig
out=$here/gate-$n
mkdir -p "$out"
cd "$wt"
"$zig" build test > "$out/zig-test.log" 2>&1; t=$?
if [[ $t != 0 ]]; then cp "$out/zig-test.log" "$out/zig-test.first.log"; "$zig" build test > "$out/zig-test.log" 2>&1; t=$?; fi
"$zig" build > "$out/zig-build.log" 2>&1; b=$?
/home/colum/Workspace/cuda-syntax-stubs/check.sh "$wt/zig/kernels/cuda" > "$out/check.log" 2>&1
sed "s|$wt|WT|g" "$out/check.log" > "$out/check.norm"
sed "s|$wt|WT|g; s|/home/colum/Workspace/tfref-kstack4|WT|g" "$here/base/check.log" | grep -v '^exit ' > "$out/check.base.norm"
"$here/check_all.sh" "$wt/zig/kernels/cuda" > "$out/check_all.txt" 2>&1
{
    echo "tree: $(git -C "$wt" rev-parse --short HEAD)$(git -C "$wt" diff --quiet || echo ' + working tree')"
    echo "zig build test: exit $t"
    echo "zig build: exit $b"
    echo "check.sh errors: $(grep -c 'error:' "$out/check.norm") (base $(grep -c 'error:' "$out/check.base.norm"))"
    echo "check_all.sh: $(wc -l < "$out/check_all.txt") error lines (base $(wc -l < "$here/base/check_all.txt")); diff against base:"
    diff "$here/base/check_all.txt" "$out/check_all.txt" && echo "  identical"
} | tee "$out/gate.txt"
