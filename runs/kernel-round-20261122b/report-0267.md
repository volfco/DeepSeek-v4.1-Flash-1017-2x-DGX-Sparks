# Report 0267 — TF_DS_MARKOV_FUSED (a greedy draft step's Markov bias miss and tile scoring in one kernel)

**Commit:** `6158a6f` on branch `kernel-0267` (worktree `/home/colum/Workspace/tfref-k0267`, base `27baa93` = recipe-0264).
One commit, subject exactly as BRIEF-0267 specifies. Working tree clean.

**Switch:** `TF_DS_MARKOV_FUSED` (off by default; `1` on; unset/`""`/`0` off; anything else refuses startup with
`TF_DS_MARKOV_FUSED: expected 1 or 0`, per the `idx_fused.zig` convention).

## Files touched (9)

| File | Change |
| --- | --- |
| `zig/src/families/dsv41/markov_fused.zig` | **new**, pure module: switch parse, `fuseOk`, the fused arm's `FusedArgs`/`Fused` types, and the host reference (`better`, `biasElem`, `step`) + 6 tests |
| `zig/kernels/cuda/dsv41_ops.cu` | `tf_ds_markov_better` + `tf_ds_markov_fused_kernel` (~95 lines, with the other Markov-loop kernels) |
| `zig/src/families/dsv41/ops.zig` | `markov_fused: bool` + `markov_fused_fn: ?cuda.Function` parsed at `Ops` init (kernel looked up only when on), `Ops.markovFused` launcher |
| `zig/src/families/dsv41/tri_markov.zig` | `Markov.localBestFused` / `Markov.stepsFused` (the fused arm beside `localBestWith`/`stepsTapped`, which are untouched) + 2 tests |
| `zig/src/families/dsv41/draft.zig` | `FusedTap` + the arm choice in `Drafter.body`'s Markov loop (fused for a greedy pass; today's `stepsTapped` is the fallback arm) + a once-only banner |
| `zig/src/families/dsv41/fused_trace.zig` | `markov_fused` / `markov_unfused` arms (0256's dispatch-trace idiom) |
| `zig/src/families/dsv41/markov_tokens.zig` | `has(tok)`: the load-time slot table's membership (the hit rule) + test |
| `zig/src/families/dsv41/model.zig` | the host-side bias-cache hit/miss counter in `Model.pass`, gated on `m.prof` |
| `zig/src/families/dsv41/native.zig` | the counter's `markov_cache` line in `printRoundProfile` (TF_DS_ROUND_PROFILE only) |

Harness: `runs/kernel-round-20261122b/h0267/` (`build.sh`, `splice.py`, `fused_run.cpp`, `fused_kernel_current.inc`,
`run.log`, `stubs/`, `BRANCH-NOTE.md`, `decisions.md`, `todo.md`).

## What the patch does

On a cache miss the kit's `_bias` (markov.py) computes the step's bias row, rounds it to fp16 into the staging rows
and `_score` rereads it: 64,640 staged fp16 values per missed stream, step and rank — 129,280 B written + 129,280 B
reread = **252.5 KiB** (the brief's number checks out exactly at the served 129,280 vocabulary on TP2, where a rank's
row is `n_cols = 64,640`). The fused kernel computes a miss row's bias as `fp16(head[v] . fp16(emb[tok]))` — the same
round to fp16, kept — widens it back, adds the logit and emits each 1,024-column tile's winner (value, id) into
`PV`/`PI` exactly as `_score` writes them. `_finish`, the per-step gather and `_pick` are unchanged. The staging rows
are **not written**: nothing reads them on a greedy step (the only other reader, `TF_DS_SPEC_SAMPLE`'s `SpecTap`,
excludes itself — see the arm rule below). Hit rows read their cached row as before.

**Arm rule (the brief's contract):** the fused arm runs only for greedy steps — `noise == null and mid == null`, i.e.
an all-greedy pass (`TF_DS_T26_KEYED_DRAFT` with every temp ≤ 0, or unset). A pass that may be sampled (keyed noise
with any T > 0, or `TF_DS_SPEC_SAMPLE`) keeps today's `_bias` + `_score` path and therefore its staged bias for the
sampled hook, byte for byte. Anything else the fused launch cannot take (`fuseOk`: rows 1–16, global ids below
`1 << 30`, `n_part = tiles × segs`) also falls back to today's path.

**Deliverable 1, the hit/miss counter.** Bias-cache hit/miss is decided by `slot[tok] >= 0`; the slot table is built
on the host at load and never changes, and the step tokens are read back by the pass anyway (`dr.drafts` + the input
tokens). So the counter is host-side in `Model.pass` (the single funnel of both call sites): for each step j of stream
i, the bias row's token is `out[i, j]` = the pending token (j = 0) then `drafts[i][j-1]`; `markov_tokens.has` counts
hits and misses. It runs **only** while `TF_DS_ROUND_PROFILE` is on (`m.prof`), adds zero device work (graph-safe),
and is printed at the /metrics scrape as
`{"rank": 0, "markov_cache": {"hits": N, "misses": M}}` — cumulative, with the profile's other lines. With the
profile off the code adds two u64 fields and one `if`.

## Launch counts at the served shape

Served: `TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2`, graphs on, TP2 (split), up to four streams.

| | per greedy step | per 6-step drafter pass |
| --- | --- | --- |
| before (off or sampled) | `_bias` + `_score` + `_finish` + `_pick` = 4 | 24 |
| after (on, greedy) | `fused` + `_finish` + `_pick` = 3 | 18 |

(The single-rank path is 3 → 2 a step: no `_pick`.) Launch counts do not depend on the stream count; a pass with
fewer steps scales linearly (5 steps: 20 → 15). `fill`'s launches at load are unchanged. The off path is launch- and
byte-identical: the kernel is looked up only when the switch is on, `stepsTapped` is untouched, and every addition
on the off path is host-side and inert (the profile counter gated on `m.prof`, the trace note gated on
`TF_DS_FUSED_TRACE`).

## Gate results (all three pass)

1. **`zig build test`: exit 0, 27/27 steps.** No new failures against the base. (During development the spill-tier
   timing test flaked once and passed on rerun; the brief's known flake `lane_host.test.prefill` did not appear.)
   New tests: 6 in `markov_fused.zig` (switch parse, `fuseOk` bounds, the tie rule incl. NaN/±inf, the fp16 round's
   ties-to-even, two pipeline cases), 2 `stepsFused` launch-sequence tests in `tri_markov.zig`, 1 in
   `markov_tokens.zig`.
2. **CUDA syntax check: `cuda syntax check: ok`** with the round-local stubs `h0267/stubs/`. The shared
   `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails on the base `27baa93` itself** (24 pre-existing errors:
   `__nv_bfloat16` from int, `__nv_bfloat162`, `float4`, `__clz` — stub drift older than this round, the same finding
   report-0260 documents). The shared stubs were left alone; `h0267/stubs/` is the 0260 round's fixed copy plus
   `__float2half_rn`, and prints ok on both the base and this commit.
3. **CPU-emulation harness `h0267/` (`run.log`, exit 0).** The kernel's exact source (extracted by `splice.py` from
   `dsv41_ops.cu`) runs under 256-threads-a-block emulation (`__syncthreads` a barrier) against an **independent**
   reference written from BRIEF-0267 and the kit's markov.py spec (bias → fp16 round → add → tile argmax with
   torch.argmax's tie rule, then `_finish`'s fold). Compiled `-O2 -ffp-contract=off` (the kernel's `--fmad=false`
   MUL-then-ADD). Results: **170 cases, 729 rows compared, all match** (tile winner value bits and ids, and the final
   greedy draft ids). Cross-checks: fp16 widen vs `_Float16` on all 65,536 half patterns; fp16 round vs `_Float16` on
   200,000 bit patterns (199,208 finite bit-exact, 792 NaN by kind). Case matrix: misses for 1–4 streams (plus 7- and
   16-row bounds), mixed hit/miss rows and all-hit rows (random cached rows: the kernel must read them, not
   recompute), ties (in-tile, across tiles, a whole row tied, ties against the masked tail), NaN/±inf logits (two
   NaNs → the lower index, `+inf` ties, an all `-inf` row), tile widths 1/1023/1024/1025/2176/3000 (masked tail never
   wins), two segments (the non-split geometry, `seg0` offsets, per-segment cache columns), `t_stride` 3, 24
   wild-bit-pattern cases (NaN/inf/subnormal weights, embeddings and cached rows), 120 tame fuzz cases, and one
   served-shape case (64,640 columns × rank 256 × 4 rows). Mutation sensitivity: `better_ties` (ties to the higher
   index) caught with 213 failing checks, `no_round` (the fp16 round skipped) caught with 305, `miss_mask` (slot-0
   rows taken as misses) caught with 301.

## Design decisions (and why)

1. **Keep the `PV`/`PI` protocol and `_finish`/`_pick` unchanged.** The fused kernel replaces exactly `_bias`'s miss
   rows and `_score`; the tile winners land where `_score` puts them (`p = s * s_tiles + t`). Minimal surface, and
   the cross-rank tie behavior (`_pick`: the lower index) is untouched. Consequence: `PV`/`PI` are bit-identical
   buffers for their later reader (`_finish`), which the parity rule requires.
2. **The dot is a sequential fp32 sum in rank order** (`__fadd_rn(acc, __fmul_rn(wk, ek))` with explicit rounding),
   the kit's documented "_bias: fp32 accumulation in rank order" semantics — the serial matrix-vector arithmetic
   `markov_check.py` found equal to the kit's `tl.dot` at these shapes. Not WMMA: the CPU harness can execute the
   arithmetic exactly, and the rounding assumptions stay one line (below).
3. **Kernel shape:** grid `(s_tiles, segs)` as `_score`; 256 threads a block, each thread4 columns of the
   1,024-column tile; the k-outer loop reads each head value once per launch and accumulates 16 unrolled
   (register-resident) row accumulators; hit rows skip the dot entirely; masked columns fold the `(-inf, 1 << 30)`
   identity (exactly `_score`'s masked fill, so the reduce is equivalent); one shared-memory reduce a row under
   `_better` (a total order on distinct ids, so the reduction tree cannot matter).
4. **Counter host-side in `Model.pass`, not in the kernels.** Hit/miss is a pure function of host-known data (the
   static slot table + the tokens the pass already copies back), so counting after the pass is exact, costs nothing
   on device, works under graph capture, and needs no readback. Counted per bias row (step j of stream i reads
   `out[i, j]`): `N × steps` rows a pass, the same rows `_bias` launches.
5. **`fuseOk` validation at the arm choice and again in the launcher** (house pattern: `idx_fused.fuseOk`), so a
   malformed shape falls back instead of launching.
6. **fused_trace arms** (`markov_fused`/`markov_unfused`) so the hardware gate's `TF_DS_FUSED_TRACE=1` run can prove
   which arm executed (0256's idiom), and one once-only `std.log.info` banner at the first fused pass (0260's
   `conf_pack` idiom).
7. **Both ranks run identical code** (the arm choice depends only on the pass and the switch); the counter counts on
   both ranks, and the profile line prints on rank 0 like every other profile line.

## Deviations from BRIEF-0267 / BRIEF-common

- The harness hint "simple elementwise kernels only need a thread-loop emulation, no barrier simulation" does not
  cover this kernel (it has a per-tile argmax reduce); the harness emulates `__syncthreads` with the `p0194`
  barrier pattern.
- Gate 2 as literally written cannot pass (the shared stubs fail on the base); executed and reported as above.
- AGENTS.md's `decisions.md` / `todo.md` live in `h0267/` (the only writable area besides the report) to keep the
  commit minimal and the tree clean; the decision record is mirrored here.
- The brief's counter note ("prefer a host-side count if hit/miss is decided on the host"): the decision is made on
  device but is host-derivable exactly (see 4), so the host-side count is used.

## Rounding points assumed (what only the hardware gate can settle)

1. **The dot's accumulation order.** The fused kernel sums exact fp16×fp16 products in fp32 sequentially in rank
   order. The kit's `tl.dot` accumulates "in rank order, 16 at a time" on tensor cores; the kit's own
   `markov_check.py` evidences that equals the serial matrix-vector shape at these shapes. Any association
   difference below half an fp16 ULP is absorbed by the round to fp16; the residual risk is an element sitting
   exactly on an fp16 rounding boundary where the two orders land on opposite sides (its staged value would differ
   by 1 ULP fp16 and could flip a near-tie argmax). Not observed in the harness; bounded by construction to
   boundary cases.
2. **fp16 rounding is round-to-nearest-even in both arms** (Triton `.to(tl.float16)` / `cvt.rn.f16.f32`), overflow to
   ±inf and sub-2^-24 to the fp16 subnormal/zero ladder; a NaN canonicalises to the quiet `0x7e00` with its sign
   kept (assumed for the GPU conversion).
3. **`score = lg + widen(bias)` is one RN fp32 add in both arms** (`--fmad=false`; Triton without fast math). When
   both operands are NaN, the sign/payload the add returns is instruction-determined and identical in both arms on
   the GPU; the CPU harness compares NaN winning values by kind because C leaves that propagation unspecified across
   separate compilations (13 wild cases showed only NaN-sign differences; all ids matched).
4. **`bf16 → fp16` of the embedding** is exact except at fp16's edges (same conversion in both arms).

## Known gaps (hardware gate / next round)

1. **GPU bit-identity vs the kit cubins** is unproven here (no GPU): the 39-case exact suite and long-reply hashes
   with `TF_DS_MARKOV_FUSED=1` are the gate, per the round's process. The assumed rounding points above are what to
   watch.
2. **Performance is not established locally.** The fused kernel replaces `_bias`'s tensor-core GEMV (always 16 rows
   wide) with CUDA-core dots (only the miss rows), and saves the 252.5 KiB/miss-row staging round trip plus one
   launch. At four all-miss rows that is ~66 MFMA a step on cores — plausibly comparable to `_bias`'s own compute,
   with the staging traffic and launch as the win. If the graphs-on A/B at KEPT=2 shows a loss, drop the switch; the
   counter (deliverable 1) stands alone.
3. **The miss rate at KEPT=2 is unknown** until `markov_cache` is read on hardware — that is exactly what the counter
   is for. The served shape's bias rows are `streams × steps` a pass (e.g. 24 at 4×6); the counter reports hits and
   misses separately.
4. The four-node 2D geometry (`draft2d`'s `lg_w` quarters) flows through the same kernel arguments and arm rule but
   is covered only generically by the harness.
5. Register pressure of the kernel (`acc[16]` + `tokv[16]` per thread) is unmeasured; if nvcc spills, the hardware
   numbers will show it. No nvcc here: the `.cu` passed clang syntax stubs only.
6. Harness-side: the stub's `__float2half_rn` NaN policy (canonical `0x7e00`, sign kept) is assumed equal to the
   GPU conversion; only reachable with NaN weights (real checkpoints don't produce them).

## Process note

Two concurrent instances of this session raced in `h0267/` (the round's `resume-loop.sh` resumed the task while the
original process was still alive; both discovered the collision). The handshake and resolution are recorded in
`h0267/BRANCH-NOTE.md`; the racing instances were stopped before the final harness run, and the run.log, commit and
this report are from a single writer afterwards.
