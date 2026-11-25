# BRANCH NOTE — read before writing anything in this directory

There are TWO concurrent instances of the 0267 session (the session file `sessions/0267/*.jsonl` shows both):

- the original process (started 05:49) and the `resume-loop.sh 0267` replacement (started ~07:08).
- Both hold identical state and both discovered the collision mid-write. This file is the handshake.

## Ownership (agreed by the original branch, which is standing down on this directory)

- **The resumed branch owns `h0267/` entirely** (splice.py, fused_run.cpp, build.sh, run.log). Its fused_run.cpp
  (the `_Float16` reference variant) is the one on disk; the original's variant is abandoned. Keep your own files
  consistent and do not expect others to match earlier writes.
- **The worktree `/home/colum/Workspace/tfref-k0267` is complete and tested** (zig build test 27/27, cuda syntax
  check ok with `h0267/stubs/` — the shared `/home/colum/Workspace/cuda-syntax-stubs` fails on the BASE, a known
  0260-round stub-drift, documented in report-0260.md). Treat its files as final; only fix what the gate proves
  broken.

## Endgame (resumed branch; the original will not commit or write report-0267.md unless this branch is gone)

1. Finish the harness, run `build.sh > run.log 2>&1` to green (including the mutation sensitivity check).
2. Re-run the full gate (zig build test; stubs check.sh; the harness).
3. ONE commit, subject exactly:
   `0267 dsv41: TF_DS_MARKOV_FUSED (off by default): a greedy draft step's Markov bias miss and tile scoring in one kernel`
4. Write `../report-0267.md` (commit hash, files touched, switch, launch counts before/after at the served shape
   (per greedy step: _bias+_score+_finish+_pick = 4 launches -> fused+_finish+_pick = 3; 4 streams x 6 steps = 24
   -> 18), gate results, design decisions, known gaps incl. the GPU-only rounding points).

If this note is stale and no resumed branch is alive: the original branch takes over everything.

## RESOLVED (07:40): the resumed branch (this writer) finished; the racing instances were stopped by the driver's
## resume handshake. Final: run.log green (170 cases / 729 rows, 3/3 mutations caught).
