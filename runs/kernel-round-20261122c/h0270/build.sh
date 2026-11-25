#!/usr/bin/env bash
# h0270: TF_DS_PROMPT_PICKS_ONCE's host check - the padded pick list every prompt layer's attention reads under the
# reuse arm against an independently written off reference, byte for byte (BRIEF-0270's case matrix). There is no new
# kernel in this patch: the harness emulates the two pad ops the arm skips (prompt.fill = cuMemsetD8Async, and
# ops.copyRows) and runs the worktree's real prompt_picks_once.zig (Tracker/padWidth) as the reuse decision.
# Copies the worktree's module at run time (so the check always runs the committed tree), then zig test.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
wt=${WT:-/home/colum/Workspace/tfref-k0270}
zig=${ZIG:-/home/colum/.local/zig-0.17.0/zig}

# source-shape guard: the harness's transcription of prompt.zig's pad block and event hooks must match the worktree
grep -q 'if (e.ops.picks_once and ch.picks.reusable(n, kk))' "$wt/zig/src/families/dsv41/prompt.zig"
# the pad predicate and its one_split/ring_mode seams: the whole case matrix rests on them
grep -q 'if (kk % 16 != 0 and (n > tri.decode_rows or one_split))' "$wt/zig/src/families/dsv41/prompt.zig"
grep -q 'const ring_mode = n <= ring_extra and !(replay_floor and below);' "$wt/zig/src/families/dsv41/prompt.zig"
grep -q 'const one_split = !ring_mode and n <= tri.decode_rows;' "$wt/zig/src/families/dsv41/prompt.zig"
grep -q 'if (li == c.layers / 2)' "$wt/zig/src/families/dsv41/model.zig"
grep -q 'try fill(e, ch.cidxp, 0xff, n \* kp \* 8);' "$wt/zig/src/families/dsv41/prompt.zig"
grep -q 'try e.ops.copyRows(e.s, ch.cidx, kk \* 8, ch.cidxp, kp \* 8, kk \* 8, n);' "$wt/zig/src/families/dsv41/prompt.zig"
grep -q 'ch.picks.held(n, kk);' "$wt/zig/src/families/dsv41/prompt.zig"
test "$(grep -c 'ch.picks.stale();' "$wt/zig/src/families/dsv41/prompt.zig")" = 3   # begin, indexer, replayCut

mkdir -p "$here/src"
cp "$wt/zig/src/families/dsv41/prompt_picks_once.zig" "$here/src/"
cd "$here"
"$zig" test src/harness.zig 2>&1 | tee "$here/run.log"
