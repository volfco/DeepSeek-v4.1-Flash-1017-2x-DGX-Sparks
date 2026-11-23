# report-0278 — TF_DS_TAPS_PACKED

**Commit:** `971c69d4d265164a38b9674a014fcbf772333092` on `kernel-0278` (parent `10c9866`, one commit).
**Switch:** `TF_DS_TAPS_PACKED` — unset/`""`/`"0"` off, `"1"` on, anything else refuses startup with
`TF_DS_TAPS_PACKED: expected 1 or 0` on stderr (`taps_packed.fromText`, `error.BadTapsPacked` at `Ops` init).
**Subject:** `0278 dsv41: TF_DS_TAPS_PACKED (off by default): prompt taps written in the drafter's row-major layout`

## Premise

Holds at the served shape. With drafting on (production `TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2`),
`Model.sliceRun`'s completion arm copies the three tap planes into `dr.at` (row-major `[rows, taps*D]`,
the layout `absorb` and the decode rounds' `rd.taps` use) once per prompt — at the decoder chunk, whose
post-`replayCut` extent is the 128-row replay window (`c.window = sliding_window = 128`) — so exactly
3 `copyRows` launches and 128 × 3 × 5120 × 2 = **3,932,160 B = 3.75 MiB** per prompt per rank (the todo's
figure, reproduced exactly by the harness). The tap layers are `dspark_target_layer_ids = [37, 38, 39]`
(config.zig), all above the replay cut at `layers/2 = 20`, so every tap write already sees the rows the
copies moved; nothing else reads the planar layout (see Reader audit). `hcMean4` already takes an output
stride (`tf_ds_hc_mean4_kernel` writes `out[r * os + d]`), so no new kernel is needed.

## Files touched

- `zig/src/families/dsv41/taps_packed.zig` (new, pure — `idx_fused.zig` pattern): `env_name`, `parse` /
  `fromText` (`error.BadTapsPacked`), host reference `planarElem` / `packedElem`, 2 tests (parse matrix
  incl. refusal, planar+copy ≡ packed layout mapping).
- `zig/src/families/dsv41/dsv41.zig`: export + `refAll(taps_packed)`.
- `zig/src/families/dsv41/ops.zig`: `Ops.taps_packed: bool`, parsed at init
  (`taps_packed.fromText(envSpan(taps_packed.env_name))`). **No `?cuda.Function` field** — no kernel exists
  (the brief's "no new kernel needed if hcMean4's stride suffices").
- `zig/src/families/dsv41/prompt.zig`: `tap` packed arm (`out = ch.taps + j * D * 2`, `os = taps * D` — the
  decode write's shape, round.zig:2129), engagement banner, `tapRows` doc (stays the planar accessor).
- `zig/src/families/dsv41/model.zig`: `sliceRun`'s completion arm — the copy loop guarded
  `if (!e.ops.taps_packed)`; `dr.absorb(..., if (taps_packed) ch.taps else dr.at, ...)`.
- `zig/src/families/dsv41/layer_main.zig` (exact tooling): `runRounds` and `runDrafted` use
  `tsrc = if (taps_packed) ch.taps else dr.at` for the absorb-fixture check and `dr.absorb` (the recorded
  `Drafter.absorb in3` fixture *is* the packed `[n, taps*D]` bytes); `runChunk`'s per-tap check uses a new
  `Run.checkStrided` under the switch (the recorded `post.in.taps.{j}` fixtures are per-tap `[n, D]` rows,
  gathered stride by stride from the packed rows); `Run.check` factored into `point`/`finish` so both check
  paths share the digest/diff tail (output strings byte-identical).
- `tools/decimal.js`, `tools/date.js` — **unrelated to the patch**: a separate session request for two
  standalone helper libraries (with their tests). They ride in this one commit because the brief demands
  exactly one commit with the export subject and a clean tree; they are not referenced by any build and can
  be dropped from the export trivially (`git revert`-free: just omit the two files).

## Off-path parity

Unset ⇒ `ops.taps_packed == false` ⇒ `prompt.tap` executes today's exact `hcMean4` call (same args), the
three copy loops run with today's exact args, `absorb` gets `dr.at`, `checkStrided` is never entered, no
banner, no log lines. Byte- and launch-identical by construction. Both ranks alike (no rank-conditional
code; the switch is read at `Ops` init on every rank).

## Banner (engagement)

`prompt.zig` `taps_packed_noted`, one-shot behind `std.atomic.Value(bool).swap(true, .monotonic)` at the
packed write (0265's `head_scored_noted` pattern):

```
dsv41: TF_DS_TAPS_PACKED: prompt taps written in the drafter's row-major layout (first at tap <j>, <n> rows)
```

Fires exactly once per process on the first tap write with the switch on (both ranks print one line); a
switch-on run always reaches a tap write (the arm is the write itself), so a null result cannot hide.

## Launch counts at the served shape (TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2, graphs on, TP2, ≤4 streams)

| per prompt completion (128-row decoder chunk, per rank) | off (today) | on |
| --- | ---: | ---: |
| completion `copyRows` launches | 3 (`copy16`, 3×128×256 threads) | **0** |
| completion bytes copied (D2D) | 3,932,160 B (3.75 MiB) | **0** |
| `hcMean4` tap writes | 3 | 3 (same launches, new out/stride) |
| `dr.absorb` launches | unchanged | unchanged |
| decode rounds (`rd.taps` writes, eager absorbs) | unchanged | unchanged |

The on path changes launch *arguments* only where launches remain; `absorb`'s `grouped`/`kvNormRope` work
is unchanged (`ch.taps` holds byte-identical rows). Graph capture safe: no host syncs or pageable transfers
added in any captured path (the banner is host-side, at the enqueue site, once).

## CPU-emulation harness (`runs/kernel-round-20261123/h0278/`)

No new kernel exists, so per the brief the harness emulates the write/copy layouts. `build.sh` extracts the
real `tf_ds_hc_mean4_kernel` source from the worktree's `dsv41_torch.cu` (the splice trick) into
`hc_mean4.inc`; `harness.cpp` runs it as a **sequential virtual-thread loop** (no OS threads) and compares,
per completion, the planar-write + `copyRows` path against the packed write (served grid *and* a 1×32 grid
for grid-stride coverage) and an **independent reference** written from the brief's spec (packed rows
`[rows, taps*D]`, `h.to(fp32).mean(1).to(bf16)`, own bf16 conversions). Served dims (cap 2048, taps 3,
D 5120), case matrix: rows 1, 127, 128, 2048, and a partial last chunk (2048 then 517 through the shared
buffers) — **all byte-identical, 0 failures**, including the absorb window `[max(0,n-128), n)` at every
completion and the 3.75 MiB/128-row figure. Negative control (packed base at the planar stride) is detected
as a mismatch, so the comparison sees layout errors. `build.sh`, `harness.cpp`, `hc_mean4.inc`, `run.log`
kept in `h0278/`.

## Gate results

1. `zig build` (all executables incl. `tf-dsv41-layer`) — clean. `zig build test` — exit 0. The first run
   failed once in one test binary without a FAIL line (the brief's known flake pattern,
   `lane_host.test.prefill` interleave; it recurred once more mid-round) — six consecutive clean runs of
   the final tree followed, including the recorded one.
2. `cuda-syntax-stubs/check.sh` — base `10c9866`: 9 errors; this tree: the **same 9 errors** (no `.cu`
   touched; normalized error sets identical). All are the known stub false positives (`float4`, `__clz`,
   `__nv_bfloat162`, `tf_ds_triton_rmsnorm_row`, `hi`).
3. Harness `h0278/` — exit 0, `run.log` kept.

Independent review (subagent): **OK with notes**, no blockers (off-path parity, on-path byte placement vs
the dropped copies and vs `rd.taps`, reader completeness, `checkStrided` soundness, banner, module
conventions all verified). Its two notes are folded in: a `shortcut:` comment at `prompt.tap` (below) and
the `Chunk.taps` field doc now names both layouts.

## Reader audit (every planar-layout reader found by name)

- `prompt.tap` / `prompt.tapRows` — the writer/accessor; both arm-aware (above).
- `model.zig` `sliceRun` completion copies — dropped on, unchanged off.
- `layer_main.zig` `runRounds` ~720, `runDrafted` ~1212 (copies + `Drafter.absorb in3` checks) and
  `runChunk` ~913 (`post.in.taps.{j}` checks) — adapted above.
- `round.zig` probes (`.taps` at `rd.taps`) and decode writes (`tri_basic.tap`) — already packed
  (`[R, taps*D]`), unchanged, **stay correct** (the packed layout is exactly `rd.taps`'s).
- `exact.zig` `hcMean4` — already stride-capable, unchanged.
- No other readers (grep of `ch.taps`/`tapRows`/`.taps` across `zig/src`: only `rd.taps` round-side consumers
  and the `Point` enum remain).

## Design decisions (decision log)

- **Switch as `Ops.taps_packed` bool only** (no `?cuda.Function`): there is no kernel to look up; the
  brief's convention's kernel half does not apply (0275's report notes the same for its kernel-less switch),
  and `Ops` init is where sibling switches (`idx_fused`, `topk_merge`, ...) parse.
- **On path: absorb reads `ch.taps` directly** instead of a packed copy into `dr.at`: the bytes are identical
  (harness-proven), so the copies are deleted work on the on path, and `dr.at` stays exclusively the round
  path's `absorbMany` scratch.
- **`checkStrided` per-row D2H** for the per-tap fixture check rather than a device gather scratch: layer
  tooling syncs anyway; no new device allocation in the tool, and `finish`/`diff` see exactly the recorded
  `[n, D]` tensor.
- **`tapRows` keeps its planar meaning** (it is the off path's accessor); the packed position is computed at
  the one on-path call site, mirroring the decode write's inline math.

## Known gaps (hardware-only or out of scope)

- **GPU rounding:** values are untouched by construction (same kernel, same inputs, per-element independent
  of the out pointer/stride), but only the hardware exact panel can *demonstrate* on-path bit-identity.
- **The 3-launch / 3.75 MiB saving** and its wall-time effect at the served prompt mix need the hardware
  timing gate; the local gate proves the bytes and launches, not the speed.
- **Banner count** proves engagement on hardware (expected: one line per rank at the first prompt tap write;
  `tf-dsv41-layer` with the switch on prints it too).
- **layer_main under the switch** (fixture checks with `TF_DS_TAPS_PACKED=1`) needs the recorded fixtures on
  hardware; locally only the code paths compile and the layout equivalence is proven.
- `shortcut:` (review finding): a tap layer below `layers/2` under `TF_DS_EXACT_RESUME` tail-prepend would
  leave stale tap rows whose bytes differ between layouts — unreachable at the served checkpoint (taps
  [37, 38, 39] > 20); upgrade if tap layers move below the replay cut.
- The commit carries `tools/decimal.js` + `tools/date.js` (session request, unrelated); if the export must
  be patch-pure, drop those two files from it.
