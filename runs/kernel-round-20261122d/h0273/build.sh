#!/usr/bin/env bash
# h0273: the 0273 capture failure path and the 0274 call-gate statement on the CPU, against independent
# references written from the briefs. Neither patch adds a .cu kernel: 0273's code under test is round.zig's
# capture failure path (spliced verbatim out of round.zig below, the splice.py trick adapted to Zig: no
# preprocessor, so the extracted top-level decls are concatenated after the harness file), and 0274's is one
# statement in engine.zig (feed_gate.zig models its exact shapes; the real regression tests are
# engine_test.zig's, run by `zig build test`). Re-run after round.zig edits: the .inc is re-extracted every time.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
zig=${ZIG:-/home/colum/.local/zig-0.17.0/zig}
tree=${TREE:-/home/colum/Workspace/tfref-k0273}
round=$tree/zig/src/families/dsv41/round.zig

# the fix's source, verbatim from the worktree: StretchCapture + leaveCapture (0273)
sed -n '/^\/\/\/ What leaveCapture does/,/^};$/p' "$round" > "$here/leave_capture.inc.zig"
sed -n '/^\/\/\/ A stretch failed inside its capture/,/^}$/p' "$round" >> "$here/leave_capture.inc.zig"
cat "$here/capture_harness.zig" "$here/leave_capture.inc.zig" > "$here/gen_capture_run.zig"

{
  echo "== extracted from $round =="
  cat "$here/leave_capture.inc.zig"
  echo "== capture harness (0273): the capture model, the case matrix, the negative control =="
  "$zig" test "$here/gen_capture_run.zig"
  echo "== feed gate (0274): the statement shapes against the brief's reference =="
  "$zig" test "$here/feed_gate.zig"
  echo "== all harness cases matched their references =="
} 2>&1 | tee "$here/run.log"
