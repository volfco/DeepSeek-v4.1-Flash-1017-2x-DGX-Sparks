# report-0265 — TF_DS_DRAFT_HEAD_SCORED

**Commit:** `c04f020` on `kernel-0265` (parent `27baa93`). **Switch:** `TF_DS_DRAFT_HEAD_SCORED` (off by default; `1`
on, `0`/`""`/unset off, anything else refuses startup with `TF_DS_DRAFT_HEAD_SCORED: expected 1 or 0`).
**Subject:** `0265 dsv41: TF_DS_DRAFT_HEAD_SCORED (off by default): the drafter's vocab head projects only the rows a
pass scores`.

## Premise

Holds. The drafter's final head (`Drafter.body`, `draft.zig:701-703`) projects `R = N * blockRows(steps)` rows while
the Markov loop reads only the first `steps` block rows of each stream of `dr.local` and the confidence head only the
first `steps` of each stream of `dr.xc`. At the served batch (4 streams, 5 block rows, 3 steps) the packed arm
projects 12 rows instead of 20 and avoids 8 × 64,640 = 517,120 FP32 outputs (1.97 MiB) per rank per pass — the
brief's arithmetic, re-checked. The bit-identity argument holds locally (below); the residual GPU-only question is
narrow and recorded under "Known gaps".

## Files touched

- `zig/src/families/dsv41/draft_head_scored.zig` (new, pure): switch parse (`idx_fused.zig` pattern), the row maps'
  host reference (`srcRow` = `stream * n + step`, `packRow` = `stream * steps + step`, `readRow` = `step +
  stream * lrow`), the arm predicate `scored(on, checked, twod, steps, n)`, tests (parse, predicate truth table, the
  72-layout bijectivity/reader matrix).
- `zig/src/families/dsv41/ops.zig`: `draft_head_scored: bool` parsed at `Ops` load via `fromText`. **Deviation from
  the convention's "bool + ?cuda.Function":** no kernel is looked up — 0265 adds no CUDA code at all (see Design).
- `zig/src/families/dsv41/draft.zig`: `Drafter.body`'s head section narrowed (per-stream `_collapse` at row offsets,
  one `_rmsnorm` over `N * steps` rows, one `prompt.grouped` head at `M = N * steps`, packed rows `i * steps + s` in
  the same `dr.xc`/`dr.xn`/`dr.local` buffers); every reader gets the stream row stride `lrow` (`steps` packed,
  `n` full): `stepsTapped`'s `block` arg, `NoiseTap.rs`, `SpecTap.rs` (its `sq` stride unchanged), `confPack`'s `n`
  arg and the else-chain's `toF32` offsets.
- `zig/src/families/dsv41/dsv41.zig`: module export + `refAll`.

## Readers of dr.local / dr.xc / dr.xn and the rows each uses

Rows are (stream `i`, step `s`); full layout row `i * n + s`, packed layout row `i * steps + s`. Every reader indexes
with one stream stride `lrow` (`n` full / `steps` packed), except the probe.

| Reader | Site | Rows it reads | Packed arm |
| --- | --- | --- | --- |
| Markov `_score` (step `s`) | `tri_markov.zig:128-135` (`localBestWith`), loop `:160-165` | local rows `s + i * lrow`, `i < N` (`l_row = block * lw`, `block` = the `lrow` arg at `draft.zig:716`) | `= packRow(i, s)` |
| noise hook (keyed drafts / spec) | `draft.zig:150-162` (`NoiseTap`), kernel `dsv41_ops.cu` `tf_ds_keyed_noise_kernel` | local rows `s + i * rs` (`rs` = `lrow`); noise lands exactly on the rows the step scores | `= packRow(i, s)` |
| speculative-score hook | `draft.zig:170-185` (`SpecTap`), kernel `tf_ds_spec_clean_kernel` | local rows `s + i * rs` (`rs` = `steps` packed, `dr.n` full); `sq` rows `s + i * dr.n` **unchanged** (lanes' spec gather reads `stream * n + step`: `lanes.zig:1637`, `:489-503`) | `= packRow(i, s)` |
| confidence packer (0260) | `draft.zig:725` → `dsv41_ops.cu:1837-1859` (`n` used only as xc row stride, `:1846`); host ref `conf_pack.zig` `pack` | xc rows `i * lrow + s` | `= packRow(i, s)` |
| confidence chain (off variant) | `draft.zig:729` `toF32` | xc rows `i * lrow + 0 .. steps-1` contiguous | `= packRow(i, s)` |
| `dr.xn` | `draft.zig:699/703` head GEMM input only | all packed rows | unchanged |
| Probe `.local` | `draft.zig:705` → `layer_main.zig:637` (`DraftCheck` hashes `numel(shape)` bytes of the recorded `Markov.steps in1` = the whole `[N * n, cols]` buffer) | **every row** | the packed arm is skipped under a probe (fallback), so the probe sees today's full buffer |

No other reader exists (`grep dr.local/dr.xc/dr.xn` over `zig/src` + kernels; `dr.xcf`/`dr.ef`/`dr.embs`/`dr.out` are
untouched by this patch). `h`/`pre` are read by the per-stream collapses at rows `i * n + 0 .. steps-1` and by nothing
after the head section.

## GEMM row-count invariance (the contract's question)

**Invariant per row across the M values involved** (M = 20 today vs 12 packed at the served shape; M < 8 cases too):
`prompt.grouped` routes `e.w.head` to the EXL3 path (`weights.zig:727-729` loads it via `L.linear` with `cols`
non-null, so `plain == 0`; the cuBLAS `plainRows` M-padding path does not apply — cuBLAS rounding *does* follow the
shape there, `prompt.zig:734-737`). For `exl3_linear.glinear`: the kernel template is picked by
`plan(k, planN(layer))` = the layer's shape alone ("a layer keeps one reduction for every row count",
`exl3_linear.zig:15-16`), never by M; the grid (`Σ n/128 * sk`) and block (`wk * 32`) are M-independent; only
`smem = wk * min(M, 8) * 128 * 4` and `rotMany`'s grid vary with M (row tiling/scheduling, not a row's K-reduction
order). The recorded served layers include the head (`5120 × 64640, sk=1, wk=8`). So the dispatch cannot change the
per-row reduction; M stays inside one dispatch "bucket" (the same template) by construction. The one thing this repo
cannot prove is inside the precompiled `linear_grouped` cubin (see Known gaps).

## Launch counts (Drafter.body head section; the stages, Markov and confidence are unchanged)

| | off (today) | on (scored) |
| --- | ---: | ---: |
| `_collapse` | 1 launch (R rows) | N launches (`steps` rows each, per-stream offsets) |
| `_rmsnorm` | 1 (R) | 1 (N × steps) |
| head (`rot_many` + `glinear`) | 2 (M = R) | 2 (M = N × steps) |
| **total** | **4** | **N + 3** |

At the served batch (N = 4, n = 5, steps = 3): 4 → 7 launches, head GEMM rows 20 → 12 (−517,120 FP32 outputs, 1.97
MiB, per rank per pass). At N = 1 shallow (steps = 3, n = 5): 4 → 4 launches, 5 → 3 rows. At `steps == n` (a pass
scoring every row) the full arm runs unchanged.

## Which passes engage (served policy: TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2)

`lanes.zig` `draftFn` (`:1749-1830`): a pass's `steps = max` over its streams of
`passDrafts(asked, base=5, kept=the last round's kept rows, wide_kept=2, room)` (`lanes.zig:2203-2206`): `asked ≤ 5`
runs `asked`; `asked = 6` widens only when the last round kept ≥ 2 rows and `room = round_rows/live - 1 > 5`. Block
rows `n = blockRows(steps) = max(5, steps)` (`draft.zig:558-560`). The round loop's ask per stream is `chooseK`'s
confidence rule, capped at `depth(live, drafts, round_rows)` = **(5, 5, 3, 3)** for 1..4 live streams
(`draft.zig:866-875`, served 16-row rounds). So the packed arm engages **iff steps ≤ 4**, i.e.:

- 3–4 live streams: the ask caps at 3 → steps = 3 < 5 = n → **engages** (12 vs 20 rows at 4 streams).
- 1–2 live streams: engages only when *every* stream in the pass asked ≤ 4 drafts (chooseK's shallow picks, or
  shallow streams near their token cap); any stream asking 5 pushes steps to 5 = n → full arm.
- A widened pass (steps = 6, n = 6) never engages (0 rows to skip).

The benchmark should expect the saving on all 3–4-stream passes and on shallow 1–2-stream passes only.

## Semantics and parity

- **Off path: byte- and launch-identical.** The `else` arm is today's three calls verbatim; `lrow = n = dr.bn` so
  `stepsTapped`/`confPack`/`toF32` get exactly today's arguments; `NoiseTap.rs = n = dr.bn` (its old
  `dr.bn`), `SpecTap.rs = dr.n` explicitly (its old `dr.n`), so no reliance on the spec/DraftBlock exclusion.
- **On path: bit-identical at every row a reader consumes.** The scored rows are produced by the same kit binaries
  (`_collapse`, `_rmsnorm`, `rot_many`, `glinear`) as today: per-row kernels (grid/pointers only differ), and the
  head's dispatch is per-row M-invariant (above). Packed row `i * steps + s` carries exactly the value the full arm
  computes at `i * n + s`; every reader indexes those rows via `lrow`. The CPU harness checks this row-for-row.
- **Both ranks alike:** the arm depends only on (switch, probe, node shape `e.two`, N, steps) — no rank-conditional
  code.
- **Graph capture:** the arm is constant per `PassGraphs.Key = {streams, steps}` (`n = blockRows(steps)` follows from
  the key); all calls are plain launches on `e.s` with fixed dims; no host interaction added. The packed row count is
  a function of the captured shape, as the brief requires.
- **Fallbacks (today's path runs):** switch off; a checked pass (`probe != null` — the fixture-replay checker
  `layer_main.zig` hashes the whole recorded logits buffer, which the packed arm would leave stale past the scored
  rows); a 2D drafter (`e.two != null`: its logits rows are `lg_w` columns, `draft2d.zig`); `steps == n`.

## Design decisions

- **No packing kernel (deviation from the brief's "if you add a packing kernel").** The brief says "pack those rows
  while collapsing the hidden state": the packing is realized as the collapse's own row selection — one kit `_collapse`
  launch a stream at that stream's row offsets, writing the packed `xc` rows (pre-norm values retained for the
  confidence head, as specified). A custom CUDA collapse could not be shown bit-identical to the kit's Triton
  reduction order on this workstation; reusing the kit kernel makes the scored rows bit-identical **by construction**.
  Cost: N collapse launches instead of 1 (+3 launches at N = 4, a few µs of in-graph gap against ~72 µs of expected
  saving).
- **Packed rows land in the same buffers** (`dr.xc`/`dr.xn`/`dr.local`, first `N * steps` rows); no new allocations,
  no scatter-back: all readers take the packed stride instead. This is why every reader call site passes `lrow`.
- **Probe fallback** instead of half-written probe buffers: the checker keeps its whole-buffer view (today's values,
  by construction), production never probes.
- **`shortcut:` note in `draft_head_scored.zig`:** `_score`'s `l_seg` becomes `N * steps * nc` on the packed arm; it
  is inert while `Markov.segs == 1` (always on this port — `tri_markov.zig:71` gives segs = 1 for world 1 and 2) and
  the 2D case falls back; re-derive if a Markov geometry gains `segs > 1`.
- Per AGENTS.md an independent reviewer subagent walked the diff (off-path identity, reader completeness, M-invariance,
  capture, Zig edge cases): verdict "OK with notes, no blockers"; its two code notes (SpecTap's explicit `dr.n`, the
  `l_seg` shortcut comment) are in the commit. Its "harness missing" note was a scoping false positive — the harness
  lives in this round's `h0265/` per the brief, outside the worktree.
- The decision log normally lives in the Sparks repo's `decisions.md`, which this task puts out of scope; this report
  is the record.

## Gate (all against the committed tree)

1. `zig build test`: **exit 0**. Authoritative fully-uncached run (scratch cache dir): `Build Summary: 27/27 steps
   succeeded; 826/826 tests passed`. The very first run of the day hit one flaky test-binary failure (the documented
   `lane_host.test.prefill`-class flake; its name was truncated by my output filter — two flaky test binaries also
   appeared as `failed command:` retries that passed on retry in the uncached run: the `tf-cal` and text-scratch
   binaries, neither touched by this patch); all reruns clean. Final run: 27/27 steps, exit 0.
2. CUDA syntax check: `cuda syntax check: ok`. The shared `/home/colum/Workspace/cuda-syntax-stubs` stubs **fail on
   the unmodified base itself** (7 errors: `float4`, `__nv_bfloat162`, `__clz`, int→bf16 — stubs older than the tree,
   same as report-0260 found). The previous round's fixed stubs are copied to `h0265/stubs/` and used; they print ok
   on both the base and this commit. 0265 adds no `.cu` code, so nothing kernel-side changed.
3. CPU-emulation harness `h0265/` (`build.sh`, `draft_head_scored_run.cpp`, `run.log`, `stubs/`): emulates the
   per-thread row semantics of the packed arm's launches (the per-stream `_collapse` base-pointer arithmetic and
   packed writes, the packed `_rmsnorm`/head rows) against an independent reference written from the brief's row
   spec, plus every reader's stride indexing and value round-trip. **150 layouts, 0 failures** (144 matrix = N 1..4 ×
   n 5..7 × steps 1..n × 2 width regimes, 120 of them shallow; 6 served-width shapes d=5120/lw=64640 incl. the
   brief's 4×5×3 case; 124 shallow total — above the brief's ≥114/≥64 floors). Three injected call-site bugs (pack
   rows by n; reader stride n; collapse offsets by steps) are each caught (45/828/1104 failing checks).

## Known gaps (hardware gate only)

- **Per-row invariance inside the `glinear` cubin** is documented ("a layer keeps one reduction for every row count")
  and structural from the launch args (same template, grid and K plan at M=12 vs 20; only `smem = wk * min(M,8) *
  128 * 4` moves, and both served M values sit at `min = 8`, i.e. an identical launch config), but no GPU here can
  byte-compare. **Hardware gate:** A/B `dr.local` at the scored rows (on vs off) across a drafting run including
  shallow (steps < n) and N=1..4 passes; then the 39-case exact suite and long-reply hashes. A divergence would show
  up as a draft-token difference only (the verify corrects replies), so hash-level parity needs the exact suite.
- Launch count rises 4 → 7 at N = 4 while GEMM rows fall 20 → 12: the net (expected ~0.07 ms/pass at the operator's
  ~9 µs/row, minus a few µs of extra in-graph gaps) is timing-only and unmeasured here. Price it with the eager
  stage timer / a graphs-on A/B (`ph_head`-style window around the pass).
- The probe fallback means a fixture replay (`tf-dsv41-layer`) run **with the switch on** verifies the full arm at
  `.local` by design; it cannot demonstrate the packed arm's rows. Only an on/off A/B of the drafter's buffers can.
- `TF_DS_CONF_PACK` on and off both flow through `lrow` (host-checked row maps; kernel's `n` verified stride-only),
  but the GPU A/B should cover both switch combinations once.
- The saving engages only on passes with steps ≤ 4 (see policy above); at 1–2 live streams with full asks it is
  exactly zero, and the +3 launches do not run (full arm). No regression expected there, by construction.
