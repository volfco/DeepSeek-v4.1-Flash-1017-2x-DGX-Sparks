# report-0262 — TF_DS_POOL_SCORE: NOT DONE (escalated at the brief's point-2 tripwire, no commit)

**Status:** stopped before any code, as BRIEF-0262 point 2 requires ("If the wrapper/kit call cannot express this
... stop and report back"). The worktree stays at `33091fb` (0261; 0263 was also escalated, so a later 0262 commit
would sit directly on 0261). This patch was lowest priority anyway: per `runs/pool-union-20261122/RESULTS.md` the
pooled path runs only on each prompt's ≤128 bounded-replay rows (~0.02% of prefill).

## The disagreement: point 2's single `_index_score_rows` call cannot score per-row compacted keys

The brief's design:
1. Compact each row's retained blocks into a **per-row** key buffer (fixed row stride 16,384).
2. Score them with **the existing `_index_score_rows`** (`tri_index.indexScore`, `keys = false`), `n` = the compact
   width.

The kernel and wrapper cannot express that:

- **One `K` for every row of a launch.** `tri_index.zig:86-93`: `_index_score_rows` takes `K`/`KS` (one codes
  pointer, one scales pointer) and `n`, with per-row `Q`, `Wt` and `VIS` only. It has no per-row key base:
  `tri_index.zig:85` asserts `base == null` on this arm. Only the decode kernel `_index_score` has `KB`/`HAS_BASE`
  (`:76-83`). Scoring row r against its own compacted keys would therefore need one launch per row.
- **A one-row call does not reach `_index_score_rows`.** `tri_index.zig:72`: `rows <= decode_rows (16) or rowwise`
  dispatches to `_index_score` (the decode form). That is a different kernel whose bits are not established equal to
  `_index_score_rows`. RESULTS shows `_index_score_cand` is 60% 1–12 ULP off, and `_index_score` itself was not
  measured. Per-row launches would need a new wrapper path that forces `_index_score_rows` at rows = 1. That is
  outside "invoke the existing `indexScore`", and it is a kit-variant question (the recorded `_index_score_rows`
  specializations in `fixtures/launches.json` include ones with `nt: 1` baked as a constant, and Triton also
  specializes `n % 16`, so a compact `n` may select a different cubin than the dense call: the brief's "Known risk").
- **What was proven bit-exact is a shared list, not per-row lists.**
  `runs/prefill-pool-score-20261122/RESULTS.md` ("Compacted keys through the same kernel") gathered **one list for
  all rows** (2,048 blocks) and got 0 differing bits at three shapes. That supports a **row-block union** design
  (RESULTS' own 5-step recipe: union -> gather -> one `_index_score_rows` over U×8 columns -> each row's own pool as
  −inf in compact space -> top-k mapped back). It does not support the brief's per-row buffer. The union was later
  measured at 57–100% of the width (`pool-union` RESULTS), so it saves little.

`vis` alone would be expressible: the compacted columns ascend, so a row's visible ones form a prefix, and `vis[r]` =
that prefix length works with `SKIP_DEAD`'s tile rule. The blocker is the per-row K, not vis.

## What the brief owner must decide

1. **Union design:** score the row block's union once through the unchanged `indexScore` (rows > 16 keeps
   `_index_score_rows`), then mask per row and map back. Per `pool-union` it saves 0–43%.
2. **Per-row launches through a forced `_index_score_rows` path:** this needs a hardware check that the rows = 1 and
   compact-`n` specializations are the same cubin, or bit-equal to the dense launch.
3. **Drop the item:** ~0.02% of prefill.

## Groundwork that stays valid

- The dense chain to reproduce is in `prompt.zig:1575` (`indexScore(..., n_comp_end, ch.score, null, false, null, false,
  m, ih, id)`), `:1589` (`ops.applyPool(ch.score, n_comp_end, n_comp_end, cb, mrows, ch.pool_ms, m)`) and `:1591`
  (`ops.topkIndices(ch.score, n_comp_end, m, n_comp_end, kk, vr, out)`).
- `_index_score_rows`' recorded prompt specialization: `KEYS=0, PACKED=1, IH=32, ID=128, BN=64, RB=1, TREE=1,
  HAS_TMAX=0`, grid `[rows, nt]`.
- The kit score-key format has **no** top-bit flip (continue.md register; RESULTS idxcand2). A map-topk variant must
  order exactly as `tf_ds_topk_indices_kernel` does over the dense row, ties to the lower **original** column.
