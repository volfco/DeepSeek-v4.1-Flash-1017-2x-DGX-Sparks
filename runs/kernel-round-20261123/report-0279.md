# report-0279 — TF_DS_IDX_PICK_WIDE: the indexer's small-row selection past 16,384 blocks

**Commit** (branch `kernel-0279`, base `10c9866`, one commit):

| # | Hash | Subject |
| --- | --- | --- |
| 0279 | `f6e126b94f3532ae5395357a18ed18038609f24b` | `0279 dsv41: TF_DS_IDX_PICK_WIDE (off by default): the indexer's small-row selection up to 65,536 blocks` |

**Switch name:** `TF_DS_IDX_PICK_WIDE` (unset/""/"0" off; "1" on; anything else refuses startup on stderr via
`fromText`). Module `zig/src/families/dsv41/idx_pick_wide.zig` (pure). Working tree clean after the commit.

## Premise: holds — the unfused fallback is reached at served shapes

The code chain (base line numbers; `10c9866`):

- `round.zig:1183` — the indexer's column count is `nb = rd.bucket / ratio` (the candidate source layer's
  compression ratio; the served checkpoint runs ratio 2).
- `round.zig:2013` + `bucketFor` (`round.zig:57`) — `rd.bucket` is the deepest row's position rounded up to the
  next power of two, capped at `pool_cap` (`model.zig:333` = `o.context`; `lanes.zig:3122` shows the 1M-token
  pool), i.e. buckets 1,024 … 1,048,576 tokens.
- `round.zig:1322-1345` (candidate-source decode arm, `li == c.candidate_source`): at `nb > candidate_blocks *
  candidate_block` = 2048 × 8 = 16,384 columns the pool runs, `nbk = ceil(nb / 8)` blocks of 8 columns.
- `fused_trace.idxArm` → `idx_fused.fuseOk` requires `nbk <= 16,384`; past it the decode round falls back to the
  three unfused launches (`block_max` + `topk_indices[_cut]` + `pool_pick`).

At ratio 2 that makes the served map: contexts ≤ 262,144 tokens stay inside the fused bound (nbk ≤ 16,384);
contexts **262,145–524,288 tokens** run nbk = **32,768** (bucket 524,288) and contexts **524,289–1,048,576**
run nbk = **65,536** (bucket 1,048,576) — **every decode round at those lengths pays the three-launch fallback at
the candidate-source layer**. Rows: `R = rows.ids.len` (`round.zig:2003`) is one stream's block (5–6 rows) or a
verify window (≤ 7) at the served config (up to 4 streams, each its own round) — exactly `idx_fused.small_rows`.
(TF_DS_IDX_BUCKET_FINER would give nbk in multiples of 2,048 in the same window; production runs powers of two.
The prompt arm writes the mask and keeps its arms unchanged.)

## What the patch does

- **Kernel** (`zig/kernels/cuda/dsv41_ops.cu`): `tf_ds_idx_pick_wide_kernel`, a variant of
  `tf_ds_idx_pick_kernel` without the on-chip staging: the same `tf_ds_idx_select` reading the row's block maxima
  from global memory (L2-resident: one fp32 maximum a block, ≤ 256 KiB a row at 65,536 blocks). The staging is
  what caps the old path (101,376 B opt-in shared = 25,344 fp32 = the 16,384 bound + slack); the new bound is the
  radix shortcut's: `tf_ds_pool_cut` fixes the inverted column's high bytes at `0xffff`, true for columns ≤ 65,535
  (blocks ≤ 65,536), false at 65,537 (column 65,536 inverts to `0xFFFE…`). `tf_ds_idx_select`'s parameter became
  `const float*` (read-only in every caller; no behaviour change). `tf_ds_idx_pick_kernel` is untouched.
- **Pure module** (`zig/src/families/dsv41/idx_pick_wide.zig`): `env_name`/`parse`/`fromText`
  (`error.BadIdxPickWide`), `max_nbk = 65536`, `wideOk(wide, allow_small, rows, width, bsize, nbk, k)` — decode
  only (block list), rows ≤ `idx_fused.small_rows`, `idx_fused.max_nbk < nbk <= 65536`, k ≤ 2048 — and tests: the
  switch's parse forms, the wideOk window (16,385/32,768/65,536 in; 65,537/16,384/rows 8/prompt/k 2049 out), and
  the radix bound checked on the key's own bytes (0xFFFF high half at column 65,535, 0xFFFE at 65,536).
- **`ops.zig`**: `idx_pick_wide: bool` + `idx_pick_wide_fn: ?cuda.Function`, parsed at `Ops` init
  (`fromText(envSpan(env_name))`), the kernel looked up **only when the switch is on**; `idxPickWide` launcher
  (grid rows, block 1024, no dynamic shared; refuses outside its bounds like `idxPick` does).
- **`fused_trace.zig`**: `Arm.idx_wide`; `idxArm` gained the `wide` switch parameter and consults `wideOk` first
  (the arm stays single-source). The served-matrix tests carry the new rows (16,385/32,768/65,536 → `idx_wide`,
  65,537/prompt/rows 8/wide off → `idx_unfused`, everything ≤ 16,384 unchanged).
- **`round.zig`**: the candidate-source arm's new `.idx_wide` case (block_max unchanged + `idxPickWide`) and the
  engagement banner behind an atomic `swap(true)` guard (0265's pattern):
  `dsv41: TF_DS_IDX_PICK_WIDE: the indexer's small-row selection reads the global maxima past 16,384 blocks
  (first at {d} rows, {d} blocks)`. `prompt.zig` only gained the extra `idxArm` argument (its arm cannot be
  `.idx_wide`).

**Output parity:** `cblk` and its `every` masking are byte-identical to the unfused sequence's — the selection is
the same `tf_ds_idx_select` the qualified small-row form runs, keyed on the same total order (score bits, -0 as
+0, ordered as signed integers, above the inverted column; ties to the lower column) as `tf_ds_topk_indices`, with
the same slot layout (the k picked columns ascending, a column at/after `every` or with a maximum not above -inf
as -1). The CPU harness establishes this against an independent sort reference at the wide shapes (below).

## Launch counts at the served shape (decode, candidate-source layer, one round per stream)

| Shape | Before | After (switch on) |
| --- | --- | --- |
| nbk ≤ 16,384 (contexts ≤ 262,144 tok), rows ≤ 7 | 2 (`block_max` + `idx_pick`) | 2 (unchanged) |
| nbk ≤ 16,384, rows > 7 | 1 (`idx_fused`) | 1 (unchanged) |
| **nbk 16,385–65,536 (contexts 262,145–1,048,576 tok), rows ≤ 7** | **3** (`block_max` + `topk_indices[_cut]` + `pool_pick`) | **2** (`block_max` + `idx_pick_wide`) |
| nbk 16,385–65,536, rows > 7 (not served) | 3 | 3 (unchanged) |
| nbk ≥ 65,537 | 3 | 3 (unchanged; the brief's bound) |

Switch off (or unset): every row of this table is today's behaviour byte for byte (the wide arm cannot be
selected; zero new lookups, zero changed launches). Both ranks take identical paths (no rank-conditional code);
both new launches are plain device launches (graph-capture safe, no host syncs, no new pageable transfers).

## CPU-emulation harness (`runs/kernel-round-20261123/h0279/`)

- `extract.py` takes the kernel's exact source (the wide kernel plus `tf_ds_score_key`, `tf_ds_pool_cut`,
  `tf_ds_tile_prefix`, `tf_ds_idx_select`) verbatim from the worktree's `dsv41_ops.cu` into
  `pick_wide_kernel.inc` (the p0194 splice trick, marker → closing brace; a marker drift raises).
- `pick_wide_run.cpp` executes that source with one cooperative fiber (ucontext) per CUDA thread — blocks
  sequential, one OS thread, ~64 KiB a fiber — with exact `__syncthreads`/`__ballot_sync`/`__shfl_up_sync` at
  their yield points (two-phase warp slots). The reference is a **sort** by the documented total order written
  from BRIEF-0279's spec (not copied from the kernel): k largest keys, columns ascending, `every`/non-finite
  maxima as -1 at their slots.
- Case matrix (all matched, `run.log`): nbk 16,385 / 32,768 / 65,536 × rows 1 / 7 × 6 patterns (random finite,
  equal maxima, signed zeros, all −inf, mixed finite/+inf/NaN islands, NaN maxima) × `every` modes (nbk, nbk/2+1,
  0), k = 2048 — 36 cases + one mixed-pattern 7-row case. Every slot byte-compared.
- **Negative control (65,537):** with the largest maximum at column 65,536 the kernel diverges from the reference
  exactly as the bound says (the reference takes column 65,536, the kernel can never pick it — its inverted id
  leaves `0xFFFE` in the radix prefix's constant bytes), while the host gate refuses the shape.
- `gate.zig` (built by `build.sh` against the **worktree's real** `fused_trace.idxArm` over `idx_pick_wide.wideOk`,
  h0276's import pattern): 16,385/32,768/65,536 × rows 1/7 → `.idx_wide`; **65,537 → `.idx_unfused` (the unfused
  fallback stays)**; switch off / prompt / rows 8 / k 2049 → `.idx_unfused`; 16,384/6,144 → `.idx_small`;
  rows 32 → `.idx_fused`; everything-off → `.idx_unfused`. All as expected (0 mismatches).

## Local gate (all on the tree being committed)

| Gate | Result |
| --- | --- |
| `/home/colum/.local/zig-0.17.0/zig build test` | exit 0 (the very first run failed with no captured test name — the brief's known flake pattern; 5 subsequent runs green: 3 before the review fixes, 2 after) |
| `zig build` (all executables) | exit 0 |
| `cuda-syntax-stubs/check.sh zig/kernels/cuda` | the same 9-error set as base `10c9866` (stub false positives; byte-identical error text modulo line numbers, diffed against `git archive 10c9866`) |
| `h0279/build.sh` | 36 wide cases + mixed case match the reference; negative control diverges as expected; gate.zig 0 mismatches; `run.log` kept |
| independent code review (subagent, pre-commit) | no P0/P1; 4 P2 notes — three fixed before commit (round.zig comment attachment, `dsv41.zig` module registration + `refAll`, the stale "below 8192" radix comment now says 65,536), the fourth (switch independence) is decision 2 above |

## Design decisions (the round's decision log)

1. **A variant kernel, not a mode flag on `tf_ds_idx_pick_kernel`** (brief allows either): the existing kernel
   keeps its staging and its behaviour untouched; the wide variant is 3 lines over the shared `tf_ds_idx_select`.
   Rejected alternative: a `wide` template/mode in one kernel (fewer symbols, but touches the qualified path).
2. **`TF_DS_IDX_PICK_WIDE` is independent of `TF_DS_IDX_FUSED`** — it owns its regime (`max_nbk < nbk ≤ 65,536`)
   and reuses only idx_fused's constants (`small_rows`, `max_k`). With only the wide switch on the arm engages
   exactly where everything else falls back; the brief states no idx_fused requirement.
3. **Decode only** (the brief says "for decode rows"); the prompt arm's mask keeps its arms (wideOk requires
   `allow_small`). `rows > 7` keeps the fallback (the brief's small-row framing).
4. **The arm stays single-source**: `fused_trace.idxArm` gained a `wide` parameter instead of the call site
   composing two helpers (mechanical test churn; one place answers "which kernel runs").
5. **The banner lives at the round.zig call site** (0265's `head_scored_noted` pattern), so it carries the shape
   ("first at R rows, nbk blocks") and fires exactly when the new arm first runs.
6. **No shared-file writes** (`decisions.md`, `todo.md`, `continue.md` live in the shared Sparks dir with agents
   0277/0278 running concurrently; the worktree must end clean). Follow-ups below are for the round driver.
7. **Review fixes** (independent subagent review of the diff): the banner var moved above `attention`'s doc
   comment, `idx_pick_wide` registered in `dsv41.zig` (`pub const` + `refAll`) like its sibling switch modules,
   and `tf_ds_idx_select`'s radix comment corrected ("below 8192" was stale; the constant-high-byte range is
   65,536 columns). All re-gated after the fixes.

## Expected time at 32K/64K blocks (from memory traffic; the GPU gate settles it)

The 0215 note's anchor: at 8,192 blocks the small-row fused pair (2 launches) measured 10.5–11.8 µs vs
12.1–12.4 µs for the three unfused launches (1/7 rows) — the second launch + selection tail ≈ 0.6–1.6 µs. In the
wide regime the maxima count grows 4×/8× and the selection's traffic per row is: pick_wide ≈ 5 passes over nbk
fp32 (4 radix + 1 compaction) = 0.63 MiB (32K) / 1.25 MiB (64K); the unfused pair ≈ 6 radix passes + `pidx`
(16 KiB write + read) + `pool_pick`'s validity re-read ≈ 0.81 / 1.63 MiB. So the saving is ≈ 0.19 / 0.38 MiB of
L2 traffic a row (≈ 1.3 / 2.7 MiB at 7 rows) plus one launch and the `pidx` round trip; all maxima re-reads are
L2-resident (≤ 256 KiB a row). Scaling the 8K anchor by the selection-traffic ratio projects roughly **1–2 µs a
round at 32K blocks and 2–4 µs at 64K** (7 rows), against `a_idx`'s measured 0.8–2.8 ms decode stage — a small
win per round but one launch less on every deep-context round.

**Only the GPU gate can settle:** the actual crossover vs the unfused three launches at 32K/64K blocks (0215
already found the small-row form LOSES at 4,096 blocks — the wide regime is untested territory); occupancy of
one CTA a row at ≤ 7 rows with the much longer tile loop (64 tiles × 2 `tile_prefix` barriers); the L2 hit rate
on the five maxima passes vs the unfused bitonic sort; PDL/graph-replay behaviour of the new launch; and the
hardware end-to-end (the banner's one-line count proves the arm ran).

## Known gaps

- **Bit-identity vs the unfused GPU sequence** is established CPU-side via the shared total order + the sort
  reference (the selection is integer/bit-level; no rounding points), and the unfused chain itself is
  prior-qualified (0254's review). A GPU-side cblk byte-diff at 32K/65K blocks is still a hardware item.
- The **engagement banner** can only be counted on hardware (it fires in the served decode round; the CPU gate
  verifies the arm, not the log line).
- The harness emulates fp32 key/compare semantics on x86-64; there is no arithmetic in the selection, so no
  fast-math/ftz concerns arise.
- The prompt arm at wide nbk keeps the unfused three launches (out of the brief's scope); rows > 7 likewise.

## Follow-ups (for the round driver; shared files untouched)

- `native.zig`'s switch banner does not list `TF_DS_IDX_PICK_WIDE` (nor does it list `TF_DS_IDX_FUSED`; module
  doc comments are the documentation today). One banner line if the driver wants discoverability.
- If hardware shows the wide arm winning, consider the prompt arm's mask mode and rows > 7 (a 2D selection grid),
  and revisit todo.md's "Extend 0195 to 100K+ contexts" (this patch is its first half; hierarchical selection
  beyond 65,536 blocks remains the separate experiment the todo names).

## Files touched

- `zig/kernels/cuda/dsv41_ops.cu` — `tf_ds_idx_select` const parameter + `tf_ds_idx_pick_wide_kernel` (+ comments)
- `zig/src/families/dsv41/idx_pick_wide.zig` — new (pure switch module + tests)
- `zig/src/families/dsv41/dsv41.zig` — module registration (`pub const` + `refAll`, per review)
- `zig/src/families/dsv41/ops.zig` — switch fields, parse/lookup, `idxPickWide` launcher
- `zig/src/families/dsv41/fused_trace.zig` — `Arm.idx_wide`, `idxArm`'s wide gate + tests
- `zig/src/families/dsv41/round.zig` — the `.idx_wide` arm case + engagement banner
- `zig/src/families/dsv41/prompt.zig` — `idxArm` call-site argument only
- Harness (outside the repo): `runs/kernel-round-20261123/h0279/{build.sh, build.zig, extract.py,
  pick_wide_run.cpp, pick_wide_kernel.inc, gate.zig, run.log}`
