# Review round: fused-next 0213-0251 (read-only)

You are reviewing, not implementing. **Do not edit, commit, build into images, or run anything on the GPU.** Read code,
diffs and evidence; you may run read-only git commands and `zig build test` in a scratch copy if you need to.

- Diff under review: `cd ~/Workspace/tfref-fused-next && git log --oneline 3c54a9d..fused-next` (13 commits, 0213-0251)
  and `git diff 3c54a9d fused-next`. Base `3c54a9d` is production's source plus the earlier follow-ups.
- Evidence: `~/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/fused-next-20261121/wp{0,1,2,3,4}/` (each
  `RESULTS.md`, `SUPERVISOR.md`, logs/JSON) and the harness `tools/fused_bench/`. The rules the work was held to are in
  `runs/fused-next-20261121/BRIEF.md`.
- Context: this engine serves production; every change must be byte-identical to the path it replaces, and any output a
  kit kernel consumes must match the kit producer's bytes. The previous fused round shipped a change (0206) that passed
  its standalone checks and then changed 4 of 39 replies on the pair because its reference was wrong.

Run these reviewers **in parallel as subagents**, each with this brief and its own focus, then merge their findings:

1. **Bit-exactness:** every changed kernel path versus the code it replaces: reduction order, rounding points,
   NaN/inf/signed-zero handling, ties and total orders in selection (0213-0215), FP4/E8M0 encoding (0240), the query
   geometry dispatch (0241), the shuffle guard (0242), the decode/prompt compressor (0230-0231). Look for inputs or
   shapes the harness never covered.
2. **Concurrency and memory safety:** PDL ordering (0250: nothing produced by a predecessor may be read before
   `griddepcontrol.wait`), shared-memory sizing and the opt-in dynamic shared limit (0217), cross-block or cross-launch
   dependencies (0215's two-phase form, 0231's ring writes), out-of-bounds or uninitialized reads at edge shapes, warp
   primitive masks (`__reduce_max_sync`, full-mask shuffles under the new guards).
3. **Host-side call sites:** `round.zig`, `prompt.zig`, `idx_fused.zig`, `exact.zig`, `ops.zig`, `rope_store.zig`,
   `compress2_norm.zig`: dispatch thresholds and fallbacks (0215's 6144-block switch, 0217's `max_nbk`, 0241's row
   threshold), 0216's prompt bypass (the harness cannot byte-check it), 0231's ring wrap / odd trailing row / one-row
   chunk, graph capture keys, both ranks agreeing, and the unfused fallbacks still reachable and correct.
4. **Claims versus evidence:** for each commit, does the evidence actually support the stated speedup and parity (right
   reference, served shapes, interleaved timing, launch floor)? Flag anything asserted but not measured.

Write `runs/fused-next-20261121/review/REVIEW.md`: findings ranked by severity (blocker / should-fix / note), each with
file:line, the concrete failing input or scenario, and how sure you are. Write it in appends of at most ~150 lines.
Keep every response short. End with a one-paragraph verdict on whether the series is safe to benchmark on production.
