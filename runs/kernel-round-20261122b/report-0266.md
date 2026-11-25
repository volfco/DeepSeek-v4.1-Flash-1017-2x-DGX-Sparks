# report-0266 — TF_DS_COMPRESS2_PAIR_ONCE

**Commit:** `e7bfa00` (parent `27baa93`, branch `kernel-0266`). **Switch:** `TF_DS_COMPRESS2_PAIR_ONCE` (off by
default; unset/`""`/`"0"` off, `"1"` on, anything else prints `TF_DS_COMPRESS2_PAIR_ONCE: expected 1 or 0` and refuses
startup). Takes effect only on the `TF_DS_FUSED_COMPRESS2_NORM=1` decode arm (the call-site branch is nested inside
`compressArm(...) == .compress2x_norm`, which production sets); at other widths/shapes 0197's kernel runs unchanged.

**Premise verdict: holds.** Rows at positions `2k` and `2k + 1` of one request share the ordered source-slot pair
(`setGlue`'s `gi` rule, `round.zig:949-953`), so one contiguous seven-row window has exactly four distinct pairs at
both starting parities (verified by hand from `setGlue`, pinned by `compress2_pair.zig` tests, and counted by the
harness). Rows from different requests never share: the key carries the request's raw-slot base
(`slot * raw + offset`, `offset < raw`), so keys of different slots are disjoint even under wrap. Member rows' output
bytes are a pure function of the pair's resolved source bytes, so broadcast is byte-identical by construction.

## Files touched

- `zig/src/families/dsv41/compress2_pair.zig` (new, pure): switch parse/`fromText`, `works(on, rows, d)`, the pair-key
  grouping rule's host reference (`pairKey`, `sharePair`, `distinctPairs`) and tests (parse forms, shape gate,
  agreement with `compress2_norm.decodeOk`, `setGlue`'s gi rule incl. the `% 64` wrap, cross-request non-sharing,
  7-row window = 4 pairs at both parities, wrap window pairing).
- `zig/kernels/cuda/dsv41_torch.cu`: purely additive (+117/−0) after 0197's instantiations —
  `tf_ds_compress2_first_pair` (warp-ballot first row of a key), `tf_ds_compress2x_norm_pair_body<D>`,
  `TF_DS_COMPRESS2X_NORM_PAIR_KERNEL`, `tf_ds_compress2x_norm{128,512}_pair_kernel`. **The kernel is in
  `dsv41_torch.cu`, not `dsv41_ops.cu`, because it needs this file's `__device__ helpers** (`tf_ds_compress2_pair`,
  `tf_ds_compress2_partners`, `tf_ds_kit_squares4`/`warp_sum`/`inverse_rms` — the last is load-bearing bit-exact PTX
  per continue.md's register, and duplicating it was not acceptable).
- `zig/src/families/dsv41/exact.zig`: `compress2_pair_once: bool` + two `?cuda.Function` (looked up **only** when the
  switch is on), parsed at `Exact.load` via `compress2_pair.fromText(envSpan(env_name))`; `compress2xNormPair`
  launcher (compress2xNorm's argument list and launch geometry verbatim).
- `zig/src/families/dsv41/round.zig`: `compressorChain`'s decode arm branches on
  `e.exact.compress2_pair_once and compress2_pair.works(true, R, hd)`; 0197's `compress2xNorm` call is the fallback
  arm, byte-identical to today's line.
- `zig/src/families/dsv41/fused_trace.zig`: `compress2x_pair` arm value (the opt-in trace must name the kernel that
  actually ran; its doc promises exact kernel identification).
- `zig/src/families/dsv41/dsv41.zig`: export + `refAll` (the convention every pure switch module follows).

## Launch counts and grid (served shape: TP2, up to 4 streams, a stream's window 5–7 rows, compressed layers)

| | Off (today) | On |
| --- | --- | --- |
| Kernels a ratio-2 chain | 1 (`tf_ds_compress2x_norm{128,512}_kernel`) | 1 (`..._pair_kernel`) |
| Grid (old and new) | `grid.x = R` blocks, `block.x = D` threads (1 block a row, one channel a thread), D ∈ {128, 512} | **unchanged** |
| Pair computations, 7-row window | 7 | 4 (saved 3) |
| Pair computations, 5 / 6-row window | 5 / 6 | 3 / 3 (even start) or 3 / 4 (odd start): saved 2 / 3 or 2 / 2 |
| Pair computations, 4×(5..7)-row round | 20–28 | 12–16 (saved 8–12; all 1,296 size/parity combinations enumerated by the harness) |

Launch count is unchanged; the saving is inside the launch (skipped softmax/expf/norm work). The new grid is
deliberately identical to 0197's: the register's timings (1.75/2.00/2.01 µs at R = 1/7/32 × 512) show blocks already
run concurrently and adding blocks is nearly free, so fewer, fatter blocks (one block a window) would serialise the
pairs and lose. A repeated key's later block stops after its raw-ring scatter (~the scatter + one ballot scan), and
its key's first row stores the shared result straight into every member's output row.

## Design decisions

- **No shared staging, no new barrier.** The brief's ordering rule (a duplicate must not read a shared result before
  it is written) is satisfied trivially: the computing block writes each member's final output row directly, so no row
  ever reads another block's result and no `__syncthreads` was added. Answering the brief's "if the new design adds a
  barrier that plausibly costs more than it saves, say so": it adds none; the only new per-block cost is the key
  scan (`tf_ds_compress2_first_pair`, one ballot scan of `R` rows) and, on computing rows, the member scan for the
  broadcast stores (the total out-store count is unchanged: `R` rows × `D` either way).
- **The computing row's arithmetic is 0197's, instruction for instruction** (same helper calls, same fold order, same
  kit PTX); the only delta is that the once-rounded `yv` is stored per member row instead of recomputed — the same
  bytes0197 would round per row (the value is identical across members). An independent reviewer confirmed the body
  mirrors `tf_ds_compress2x_norm_body` line for line.
- **Pair key = `(gi[2g], gi[2g+1])` exactly** (the resolved source slots). Resolution is a function of the key and the
  round's `rslot` set only; the `row == g` own-row shortcut reads the same `ckv`/`cgate` element the `row >= 0` arm
  reads for another member, and the ring arm is taken only when no current row names the slot (scatters write only
  `rslot[g]` slots, so a same-launch writer can never race a ring read). Ring wrap (`% 64`) and partners from earlier
  rounds produce identical bytes for all members; covered by harness cases.
- **Duplicate `invalid` stores collapse to one** (idempotent `atomicExch`, the flag is a sticky 0/1; `ok0/ok1` are
  key-constant). Final flag byte identical; byte-compared in every harness case.
- **No special "seven-row fast path":** the grouping rule is general (any `R`, any key distribution), and the contract
  requires savings at 5–6-row windows too, where a 7-only path would fall back exactly when savings are needed. The
  harness covers 1–7-row windows per the brief's matrix plus long rounds.
- `shortcut:` none taken in the kernel; the reference's `distinctPairs` fixed 64-key buffer is asserted
  (`compress2_pair.zig`), sized for the round's `max_rows` 48.

## Gate results (all before commit)

1. **`zig build test`** — exit 0 (baseline-clean tree plus my changes; the 4 `TF_DS_COMPRESS2_PAIR_ONCE: expected 1 or
   0` lines in the log are the module test's intentional refusal probes, proving the new tests execute). No flakes hit.
2. **CUDA syntax check** — the brief-named `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails identically at
   base `27baa93`** (pre-existing stub gaps in its `cuda_bf16.h`/`cuda_runtime.h`: no `__nv_bfloat16(int)`,
   `__nv_bfloat162`, `float4`, `__clz`; errors at untouched lines 308/863/969 of `dsv41_torch.cu` and 164/893+ of
   `dsv41_ops.cu`). Base-vs-commit outputs are identical modulo line-number shifts of those same errors — **zero new
   errors from this patch**. The round's stub copy (`runs/kernel-round-20261122/stubs/`, explicitly maintained "so
   the check passes on the base and flags only new mistakes") prints **`cuda syntax check: ok`** on this tree.
   Both outputs are reproducible with the two `check.sh` invocations on the worktree.
3. **CPU-emulation harness** (`runs/kernel-round-20261122b/h0266/`, `build.sh` + `extract.py` + `harness.cpp` +
   `kernels.inc`/`kernels_host.inc` + `run.log`): the p0194 extraction trick — both kernels' exact source pulled from
   the worktree's `.cu`, `__shared__` arrays adapted to a per-block `SimBlock`, the kit's approximate inline PTX
   mapped to shared CPU primitives — run with `std::thread` a CUDA thread and barriers for `__syncthreads`, warp
   shuffles and ballots. Compared on every case: **new kernel vs 0197's extracted kernel** and **new kernel vs an
   independent per-row reference** (written from the brief's contract: per-row resolve → pair → kit-order norm, no
   grouping), byte-for-byte over raw-ring KV, raw-ring score, latent rows **including stride padding**, and the
   invalid flag; plus every row's own raw-ring slot content. **184 cases, 0 failures:** 1–7-row windows at odd/even
   starts, ring wrap (starts 62/63), partners from an earlier round (odd starts; ring pre-seeded with a distinct
   value range), mixed requests (2–4 streams a launch, distinct slots, odd/even/wrapping starts), widths 128 and 512,
   out-of-range partner (zero substitution + invalid flag) at both parities, long rounds (33 and 48 rows at starts
   0/1/62 — the ballot scan's second chunk, duplicate keys past row 32, wrap) and a 4×9-row mixed launch, and 120
   seeded random rounds (1–4 streams, 1–7 rows a stream, random starts). **Negative control** (one broadcast
   condition perturbed in a scratch copy of the `.cu`): 340 failures — the harness detects real divergence
   (`run-negative-control.log`).
4. **Independent code review** (subagent, read-only): verdict **safe to commit**, no blockers. Findings resolved
   before commit: the G > 32 harness gap (should-fix) closed with the long-round cases above; the `works`/`decodeOk`
   drift nit closed with an agreement test in `compress2_pair.zig`; the `distinctPairs` buffer nit closed with an
   assert. Its "third copy of the gi rule in the harness" nit is accepted and recorded below (closing it needs a
   refactor of the off-path `setGlue`, which this round forbids).

## Deviations from the brief

- **Switch fields live on `Exact`, not `Ops`** (bool + `?cuda.Function`, parsed at `Exact.load` with
  `fromText(envSpan(env_name))`, kernels looked up only when on). The brief's convention targets `ops.zig`, but the
  kernel must sit in `dsv41_torch.cu` (its `__device__ helpers; the brief permits this "and say which and why" —
  above), and `Ops.load` only sees the `dsv41_ops.cu` image. `Exact` is the owner of this module's kernel lookups
  (0197's `k_compress2x_norm{128,512}` are its immediate neighbours). Every other convention element (pure module,
  parse forms, lookup-only-when-on, call-site fallback) holds as written.
- **No `decisions.md`/`todo.md`/`continue.md` entries**: those files live in the Sparks repo, which this task forbids
  touching; decisions and follow-ups are recorded here instead. (Recipe commits are source-only.)
- The commit body's case count was amended 170 → 184 after the review's coverage cases (body only; subject is the
  brief's verbatim; still exactly one commit).

## Known gaps (hardware gate to settle)

- **Wall-time impact is unproven and plausibly < 0.1 µs.** 0197 measures 1.75/2.00/2.01 µs at R = 1/7/32 × 512
  (continue.md register), i.e. the kernel is near-constant in row count — blocks already run concurrently and the
  pair arithmetic is mostly off the critical path. If the R = 1→7 marginal (~40 ns/row) is block-serialization, the
  3-of-7 skipped pairs save up to ~0.1 µs of 2.0 µs (≤ 0.004 ms of a 42–87 ms round across 40 layers); if it is
  scheduling noise, the wall saving is ~0 and the patch is work-reduction only. Only the standalone-GB10 A/B can
  settle this; the switch is opt-in for exactly that reason.
- **GPU bit-identity is argued, not measured.** The computing row executes literally the same source and helpers as
  0197 (so GPU rounding cannot diverge), and the CPU harness proves the logic byte-exact with shared primitives; the
  hardware gate should still byte-compare raw rings, latents (incl. stride padding) and the invalid flag as usual.
  `expf`/`MUFU` rounding is shared code on both paths, not emulated here.
- **PDL = 1 arm not exercised in the harness** (the split only reorders the weight load before `pdl_wait`;
  arithmetic path identical by construction).
- **Third copy of the gi/rslot rule** in `harness.cpp`'s case generator (besides `setGlue` and `pairKey`): pinned by
  the Zig tests' literal values and the byte-compare, but a future edit to `setGlue` would not be cross-checked
  automatically. Follow-up when the off path next moves: extract `setGlue`'s formula to `compress2_pair.pairKey`.
- At R = 1 (single-row windows) the pair-once kernel does the key scan0197 skips — marginally more work in the
  shape where nothing repeats; irrelevant at the served 5–7-row windows and harmless elsewhere (opt-in switch).

## For the benchmark driver

Switch arms: `TF_DS_FUSED_COMPRESS2_NORM=1` (both, as production) with `TF_DS_COMPRESS2_PAIR_ONCE=0|1`. Served
shapes: R = 5/6/7 rows one stream and 4×(5..7) mixed, widths 128 and 512, graphs on, PDL as profiled. Expect no
byte differences anywhere (the exact suite and the layer gate should be bit-clean); the only question is time, and
the above ceiling says treat a sub-0.1-µs delta as noise unless repeated.
