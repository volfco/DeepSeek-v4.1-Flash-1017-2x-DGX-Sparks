# Kernel round 0277–0279 — common brief (read first)

Worktree: `/home/colum/Workspace/tfref-k<NNNN>` (your brief's number), branch `kernel-<NNNN>`, base `10c9866`
(`kernel-stack-0272` = recipe-0264 `27baa93` + 0272–0275: exactly the image production runs, `bench-fused-next-10c9866`). You are the **sole writer** in this
worktree while you run. Do not touch any other worktree, do not ssh anywhere, do not touch the Sparks or production.

## What this round is

Open `todo.md` kernel items, one brief each (`BRIEF-0277.md` … `BRIEF-0279.md`), each written by a separate agent in its own
worktree on the same base, in parallel. You implement exactly one: the one your prompt names. One git commit. Another agent drives hardware benchmarking — **no GPU work is in scope
here at all** (this workstation has no nvcc and no GPU). The local gate below is the entire verification for you.

## Switch convention (follow `zig/src/families/dsv41/idx_fused.zig` exactly)

- Pure module (no `cuda` import) with `env_name`, `parse` (`unset`/`""`/`"0"` off, `"1"` on, anything else prints
  `<env_name>: expected 1 or 0` to stderr and refuses startup via `fromText` error), host reference functions and
  `std.testing` tests. Name the module per your brief.
- `zig/src/families/dsv41/ops.zig`: a `bool` field + `?cuda.Function` field parsed at `Ops` init
  (`fromText(envSpan(env_name))`), the kernel looked up **only when the switch is on** (off path must be launch-
  identical to today: zero new launches, zero changed bytes).
- Kernel in `zig/kernels/cuda/dsv41_ops.cu` (or `dsv41_torch.cu` when you need its `__device__` helpers — say
  which and why). `extern "C" __global__`, `pdl_enter(pdl)` first statement when the neighbours do, `LOCAL=0
  STACK=0` build flags already apply. No fast math anywhere.
- Call site keeps today's path as the fallback arm when the switch is off or the shape is unsupported.

## Style

Minimal diff that fully solves the brief. Follow the surrounding code's idioms. Comment only the why the code
cannot show. Known limits get `shortcut: <limit>, <when to upgrade>` comments. No new abstraction layers, no
"for later" code. Do not refactor unrelated code (especially do not touch already-qualified kernels — the GPU
gate re-verifies them; duplication of a small arithmetic tail is cheaper than an untestable refactor).

## Local gate (all must pass before your commit)

1. `/home/colum/.local/zig-0.17.0/zig build test` in the worktree — exit 0. Known flake: `lane_host.test.prefill`
   interleave; rerun once if it is the only failure. (Baseline `8d838cc` is clean.)
2. `/home/colum/Workspace/cuda-syntax-stubs/check.sh <worktree>/zig/kernels/cuda` — must print
   `cuda syntax check: ok`. (Catches C++/argument-shape mistakes only; no codegen here.)
3. **CPU-emulation harness** in
   `/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/kernel-round-20261123/h<NNNN>/`:
   extract your new kernel's source from the `.cu` (see `/home/colum/Workspace/p0194/splice.py` +
   `fused_run.cpp` for the extraction trick — `fused_kernel_current.inc`; simple elementwise kernels only need
   a thread-loop emulation, no barrier simulation), execute it on CPU against an **independent** reference
   implementation (written from the spec in your brief, not copied from the kernel), and cover the case matrix
   your brief lists. All cases must match. Keep `build.sh` + sources + `run.log` in that directory.

## Commit

Exactly one commit, subject in the export format (the leading number is load-bearing):
`<NNNN> dsv41: <SWITCH> (off by default): <one-line description>`.
Working tree clean afterwards. Write a short report to
`/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/kernel-round-20261123/report-<NNNN>.md`:
commit hash, files touched, switch name, launch counts before/after at the served shape, gate results, design
decisions and deviations from this brief, known gaps (esp. anything only the hardware gate can settle).

## Parity rules that apply to every patch

- The off path must be byte- and launch-identical to today.
- The on path must reproduce today's outputs **bit for bit** where the brief says so; where bit-identity cannot
  be established locally (GPU-only rounding points), say so explicitly in the report and keep the arithmetic as
  close to the documented reference as the codebase's existing replicators do (see continue.md's
  "Instruction-level findings" register for known Triton-vs-torch divergences).
- Both ranks must behave identically (no rank-conditional code paths unless the brief says so).
- Everything must work under CUDA graph capture (the call sites run inside captured graphs; no host syncs, no
  new pageable transfers inside `body`/graph-observed paths).

## This round's extra rules

- Production runs `TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2` with graphs on, TP2, up to four streams. Size your
  served shape from that (a stream's block is 5 or 6 rows; a verify window up to 7 rows).
- Previous rounds' briefs and reports (`runs/kernel-round-20261122*/BRIEF-02*.md`, `report-02*.md`) and the
  findings register in `continue.md` ("Kernel, instruction and hardware findings register") are worth reading first:
  they record which rounding points are GPU-only and which kit kernels must not be touched.
- Never edit kit cubins or anything under the kit; never change the off path.
- If while reading the code you find the brief's premise is wrong (the saving does not exist at the served shape, or
  bit-identity is impossible), stop, do not commit, and write that finding with evidence as the report instead. A
  well-evidenced "not worth building" is a valid outcome.
- Your process has no GPU. Do not try to build images, run docker, or ssh.
- **Engagement banner (new this round, mandatory):** when the switch is on and the new arm first runs, log once
  `std.log.info("dsv41: <SWITCH>: <what engaged> (first at ...)")` behind an atomic `swap(true)` guard, exactly like
  0265's `head_scored_noted` in `draft.zig` (`~/Workspace/tfref-k0265`). The hardware gate counts that line to prove the
  arm ran. A null result with no banner cannot be told apart from a switch that never fired.
- The `todo.md` line references were written against an older tree; your base is `10c9866`, so line numbers drift.
  Find the code by name.
- Previous rounds: `runs/kernel-round-20261122b/`, `-c/`, `-d/`. Read the one nearest your area.
- The CUDA syntax stub checker reports 9 errors on the base itself (stub false positives). Run it on the base and on your
  tree and require the **same error set** (modulo line numbers), not a clean pass.
- `zig build test` must be run from the worktree root (not `zig/`). `zig build test` does not compile `lanes_main.zig`;
  also run `zig build` to compile every executable.

