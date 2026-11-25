# Kernel pattern sweep — brief (2026-11-23)

**Goal (the operator's words, paraphrased).** Take what the kernel rounds have learned so far: the code patterns that
were changed and that measurably improved performance. Search every other kernel in the engine repository for the
same opportunities and apply the same improvements there, including kernels this recipe's served configuration never
runs (other model families, other backends, unserved dsv41 paths).

## Where things are

- **Engine repository:** git worktrees under `~/Workspace/tfref-*`. Production is `~/Workspace/tfref-kstack4` at
  `2bb052f`. Kernels: `zig/kernels/cuda/**` (about 7,900 lines: `dsv41_ops.cu`, `dsv41_torch.cu`, `dsv41_shard.cu`,
  `sample.cu`, `qmm_group.cu`, `gdn.cu`, `experts*.cu`, `prefill_attention.cu`, `lane_gemv.cu`, `scan_rows.cu`,
  `nemotron_ops.cu`, `torch_ops/*.cu`, `dsv41/rdma_gather.cu`, ...) and `zig/kernels/metal/**` (about 240 files:
  nemotron, flashnext, glm, qwen3_5, kimi, prefill, ops, core, decode). Host call sites: `zig/src/families/*`
  (dsv41, flashnext, glm, kimi_k3, nemotron, qwen3_5) and `zig/src/core`.
- **Your worktree:** create `~/Workspace/tfref-ksweep`, branch `kernel-sweep-0290`, base `2bb052f`
  (`git -C ~/Workspace/tfref-kstack4 worktree add ~/Workspace/tfref-ksweep -b kernel-sweep-0290 2bb052f`). You are its
  sole writer. Write to no other worktree. Patch numbers start at **0290** (0280–0284 belong to another agent working
  in `~/Workspace/tfref-kstack5` right now).
- **Recipe repository** (records only): `/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`. Do not commit
  there. Write only under `runs/kernel-sweep-20261123/`.
- **Off limits, to avoid colliding with the other agent:** the drafter pass (`draft.zig` `pass` / `body`,
  `tri_markov.zig`, `markov_tokens.zig`) and the L2 prefetch (`round.zig` `prefetch*`, `ops.zig` `l2Prefetch`).
- No GPU, no nvcc, no Metal compiler on this workstation. No ssh anywhere. Nothing here touches the serving pair.

## Step 1: the pattern catalogue (write `PATTERNS.md` first)

Build it from the record, not from memory, and keep only patterns with a **measured** gain. Sources:
- `continue.md`: "Kernel, instruction and hardware findings register" (the per-kernel timing table, the SASS findings,
  the bit-exactness rules, "GB10 findings to reuse"), "Lessons from the 0254–0258 gate", and "Audit review and
  re-score of rounds 0259–0279" (which patches were real, which were neutral, and why).
- `docs/fused-kernel-rebenchmark-audit.md` (line-by-line findings, including why the losers lost).
- The patches themselves: `zig/patches/0195`, `0211`, `0213`–`0217`, `0230`, `0231`, `0240`–`0242`, `0250`–`0261`, `0264`
  and their README; engine commits for the rest (`git -C ~/Workspace/tfref-kstack4 log`), and the rejected kernels in
  `~/Workspace/tfref-k0265` … `tfref-k0279`.
- Standalone timings: `runs/fused-kernel-audit-20261120/`, `runs/prune-six-20261120/`, `runs/fused-next-20261121/wp1..wp4/`,
  `runs/suite-fused-next-20261121/kernel-review/`, `runs/fused-followup-20261121/`, `runs/pre-bench-20261121/`.

A starting list to verify and extend (each needs its evidence line in `PATTERNS.md`):

| # | Pattern | Where it was done | Recorded gain |
|---|---|---|---|
| P1 | Stop a radix select at a complete bin instead of running every digit | 0254 | 237 → 101 µs (128 × 32,768, k=512) |
| P2 | Fuse a chain of tiny per-row kernels into one launch at small row counts | 0197, 0199, 0195, 0260 | 3.48 → 1.75 µs; 9.78 → 2.55 µs; 44.8 → 9.8 µs |
| P3 | Reuse a fetched or computed result across layers that share its inputs | 0259 | about 1% of decode wall, the only serving-visible kernel gain |
| P4 | Write in the consumer's layout; drop the gather or copy launches | 0278, 0260 | launches and bytes removed |
| P5 | Aligned, vectorized loads and a static first phase | 0255, 0213 | 1.257 → 1.051 µs |
| P6 | Exact bit-level replacements for float work | 0240 | scale byte from bf16 bits, exact reciprocal, warp integer maximum |
| P7 | Skip provably no-op work (unrotated warps, empty prefixes, launch grids past the last row) | 0242, 0211, 0216, 0264 | per-kernel µs |
| P8 | Spread wide work over the grid; never one block a row over a wide input | 0215 (and the losses of 0193/0194, 0267, 0279) | 12.14 → 10.53 µs; the route lost 18% |

Anti-patterns the record paid for; search for these too, and never introduce them:
one block a row serializing wide work (0193/0194, 0267, 0279); lane stores 64 bytes apart (0195/0206: 316.6 vs
38.5 µs); a scalar CUDA-core dot where a tensor-core dot existed (0267: about +3 ms a greedy pass); a half-warp masked
`__reduce_max_sync` (+0.10 µs); `isfinite` in a hot loop (0207: 13–17% slower); unconditional full passes (0261's
eight radix digits); fusion that loses when the buffers are distinct (0198 at 128 rows); per-call host uploads or
synchronizes that a caller could hoist (0269, the drafter pass's uploads).

## Step 2: the survey (write `SURVEY.md`)

Read **every** kernel file, CUDA and Metal, and every host call site that launches one. For each site where a
catalogue pattern or anti-pattern applies, record: file and lines, the pattern, the concrete change, how often the
kernel runs where that is knowable (per token, per layer, per round, per prompt chunk), whether bit-identity can be
proven locally, and the risk. Known leads to check first, not a complete list: the other radix or top-k selectors
(`torch_ops/topk.cu`, `sample.cu`, `torch_ops/nucleus.cu`, `nemotron_sampler_core.cu`, 0261's
`tf_ds_topk_select_kernel` and the fallback `tf_ds_draw_rows_kernel`, which run the same eight passes, and 0279's
pick); argmax and reduction kernels (`torch_ops/argmax*.cu`); the sharded-KV kernels (`dsv41_shard.cu`); copy and
gather kernels (`torch_ops/movement.cu`, `indexing.cu`) whose callers could write in place; and the Metal families'
equivalents of each.

## Step 3: apply

Apply every surveyed change that meets all of these. Order by pattern strength and call frequency.
1. **Bit-identical outputs.** No change may alter a result bit, an output order or a tie rule. No fast math. A change
   that reorders floating-point accumulation is out.
2. **Proven locally.** Each kernel change gets a CPU-emulation harness under
   `runs/kernel-sweep-20261123/h<NNNN>/` that runs the old kernel source and the new one over an edge-case matrix
   (NaN, ±inf, signed zeros, ties, k at its bounds, row counts 1 / 7 / 16 / 128, widths at alignment boundaries) and
   compares bytes. The extraction technique is in `/home/colum/Workspace/p0194/splice.py` and `fused_run.cpp`; earlier
   harnesses are in `runs/kernel-round-2026112*/h*/`. Keep `build.sh`, sources and `run.log`.
3. **Off by default** behind a switch, following the family's own convention (for dsv41: `idx_fused.zig` /
   `taps_packed.zig`; one-time engagement banner, mandatory). Off must be byte- and launch-identical to `2bb052f`.
4. **Works under CUDA graph capture** where the call site is captured.
5. **Local gate before each commit:** `/home/colum/.local/zig-0.17.0/zig build test` and `zig build` from the worktree
   root (known flakes: `lane_host.test.prefill` interleave, one spill-tier timing test; rerun once), and
   `/home/colum/Workspace/cuda-syntax-stubs/check.sh <worktree>/zig/kernels/cuda` with the **same error set** as the
   base (nine stub false positives).
6. One commit a patch, subject `<NNNN> <family>: <SWITCH> (off by default): <one line>`, numbers from 0290.

**Metal kernels and anything else with no local compile or syntax check:** do not land edits you cannot check. Find
out first whether the repository has a local check for them (`gate.json`, `tools/`, the build's test step). If it
does, use it and apply. If it does not, write each change as a ready-to-apply diff under
`runs/kernel-sweep-20261123/unverified/` with the same harness evidence, and list it in the survey as "not landed: no
local check".

**Honesty rules.** Nothing here is timed: this workstation has no GPU. Never write that a change is faster; write what
work it removes and what was measured for the same pattern elsewhere. Most kernels outside the per-layer forward run
once a round or once a prompt, where the record shows microsecond savings are invisible at serving level; say so per
site, apply anyway where it is exact, and rank the sites by where a hardware gate would be most worth running. If a
pattern does not transfer to a site on inspection, say why and move on. A well-evidenced "does not apply" is a result.

## Deliverables

- `PATTERNS.md`, `SURVEY.md`, one `report-<NNNN>.md` per landed patch (commit, files, switch, work removed, harness
  result, gate lines, what only hardware can settle), `unverified/*.patch` where applicable.
- `REPORT.md`: commits landed, sites surveyed by family and backend, sites not applied and why, the hardware gates each
  landed patch still owes, and everything unverified.
