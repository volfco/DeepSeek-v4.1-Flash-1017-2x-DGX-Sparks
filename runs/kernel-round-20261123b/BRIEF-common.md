# Kernel round 0280–0283 — common brief (read first)

**Why this round exists.** The audit review in `continue.md` ("Audit review and re-score of rounds 0259–0279") found
that the drafter pass is about 10% of a one-stream decode cycle (4.15 ms of 42.8 ms), has never been broken down, and
has three untried timing-only levers. Read that section first; it holds the numbers and file references.

**One agent, one worktree, four commits in order.** Worktree `/home/colum/Workspace/tfref-kstack5`, branch
`kernel-stack-0280`, base `2bb052f` (`kernel-stack-0278` in `~/Workspace/tfref-kstack4`: exactly what production runs,
image `tensorfold-dsv41:bench-fused-next-2bb052f`). Create it with
`git -C ~/Workspace/tfref-kstack4 worktree add ~/Workspace/tfref-kstack5 -b kernel-stack-0280 2bb052f`. You are the
sole writer there. Read other `~/Workspace/tfref-*` worktrees freely; write to none of them.

## What all four patches have in common

- **Timing only.** None may change an output bit or a draft. 0280 is instrumentation; 0281 changes which Markov bias
  rows are precomputed; 0282 and 0283 touch only the L2 prefetch, which writes nothing. Say in each report why the
  bits cannot move.
- **No CUDA source changes.** Every patch uses kernels that already exist (`tf_ds_l2_prefetch_kernel`, the kit's
  `_bias`). This matters: this workstation has no nvcc, and the bench image reuses the production image's fatbins. If
  a patch seems to need a `.cu` change, stop that patch and write the finding as its report.
- **Off path byte- and launch-identical to `2bb052f`.** Unset means today's behaviour exactly.
- **Everything inside `Drafter.body` and `round.stretch` runs under CUDA graph capture.** No host sync, no pageable
  transfer, no host-dependent branching inside captured code. A forked stream must be joined before the capture ends
  (see `round.zig` `joinPrefetch` and `leaveCapture`).
- **Both ranks behave alike** unless the brief says a knob is rank-local.

## Conventions (same as rounds 0265–0279)

- Switch module: follow `zig/src/families/dsv41/idx_fused.zig` and `taps_packed.zig`: a pure module with `env_name`,
  `parse` / `fromText` (bad text prints `<env_name>: expected ...` to stderr and refuses startup), host reference
  functions and `std.testing` tests.
- Engagement banner, mandatory: one `std.log.info("dsv41: <SWITCH>: <what engaged> (first at ...)")` behind an atomic
  `swap(true)`, as 0265's `head_scored_noted` and `taps_packed_noted` do.
- Minimal diff, surrounding idioms, comments only for the why. Known limits get `shortcut: <limit>, <when to upgrade>`.
- Commit subjects (the leading number is load-bearing), one commit each, in this order:
  - `0280 dsv41: TF_DS_PASS_PROFILE (off by default): the drafter pass's phases, steps and bias misses under the profile`
  - `0281 dsv41: TF_DS_MARKOV_CACHE (unset: 256 rows): the drafter's cached Markov bias rows`
  - `0282 dsv41: TF_DS_DRAFT_PREFETCH (off by default): the paced L2 prefetch inside the drafter pass`
  - `0283 dsv41: TF_DS_L2_DELAY_NS / TF_DS_L2_RATE_GBPS / TF_DS_L2_BUDGET_MIB (unset: 3000 / 230 / 16): the L2 prefetch's pacing and budget`

## Local gate (before each commit)

1. `/home/colum/.local/zig-0.17.0/zig build test` from the worktree root: exit 0. Known flakes: `lane_host.test.prefill`
   interleave and one spill-tier timing test; rerun once if that is the only failure.
2. `/home/colum/.local/zig-0.17.0/zig build` (compiles every executable; `zig build test` does not compile
   `lanes_main.zig`).
3. `/home/colum/Workspace/cuda-syntax-stubs/check.sh <worktree>/zig/kernels/cuda`: the same error set as the base
   (nine stub false positives). With no `.cu` change this must be identical.
4. Host tests for each new module: parsing, bounds, and the host-side arithmetic (budget splits, slot tables).
Zig caches are about 320 MB each; delete `.zig-cache` of trees you no longer need. Zig is not on PATH.

## Reports

One per patch at `runs/kernel-round-20261123b/report-<NNNN>.md` in the recipe repo
(`/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`): commit hash, files touched, switch and its values,
launches added or removed at the served shape (one stream, six block rows, six steps; and four streams), gate results,
why the bits cannot move, design decisions, deviations from the brief, and what only hardware can settle. If a brief's
premise turns out wrong, do not commit that patch; write the evidence as its report and go on to the next.

## Staging (after the commits; see BRIEF-staging.md)

Do not commit anything in the recipe repo; write files only under `runs/kernel-round-20261123b/` and `runs/k280/`.
