# BRIEF-0262 — Retained-block scoring for deep prompt prefill

**Switch:** `TF_DS_POOL_SCORE` (off by default). **Module:** `zig/src/families/dsv41/pool_score.zig` (pure).
**Commit subject:** `0262 dsv41: TF_DS_POOL_SCORE (off by default): the post-source prompt indexers score only retained blocks`

## What it replaces

`zig/src/families/dsv41/prompt.zig`'s `indexer()` branch `else if (pooled and c.candidate_source < li)` (the
layers after `candidate_source_layer_id = 20` — 24/28/32/36 on the served checkpoint). Today each of those
layers runs the full chain per row block: `tri_index.indexScore` over **all** `n_comp_end` columns (fp32
scores), `ops.applyPool` (mask everything outside the layer-20 pool's retained blocks to -inf), `ops.topkIndices`
over the full width. At 128 × 65,536 that materializes 33.5 MB of scores of which at most a quarter of each
row's columns survive; the 0254 top-k over the full width is a prefill cost.

New: each row scores only its retained eight-key blocks (the pool's per-row mask the candidate source built,
`mrows = ch.pool + r0*ch.pool_ms`, `pk ≤ candidate_blocks = 2048` blocks × `candidate_block = 8` keys = at most
16,384 scored columns a row) and the top-k carries **original column IDs** into `cidx`.

## Chosen design (escalate, do not improvise, if the code disagrees)

1. **Compaction kernel** (new, `dsv41_ops.cu`): per row, walk the retained blocks with `applyPool`/`poolPick`'s
   exact mask rule (read `ops.applyPool`, `ops.poolPick`, `idx_fused.poolPick` and the kernels first —
   including the pinned newest block and **partially visible blocks**), gather the eight-key blocks' fp4
   codes/scales into a compact per-row key buffer, and write a parallel original-`col` array (ascending; pad
   with -1) plus each row's valid count. Fixed row stride at the maximum (16,384) is fine.
2. **Scoring = the existing `_index_score_rows`** (`tri_index.indexScore`, `keys = false`) invoked on the
   compacted keys (`n` = compact width, `vis` covering exactly the valid compact columns). Rationale: the
   scores are then bit-identical to the dense path **by construction** — the same kit kernel computes each
   column's score from the same `(q, k[col], w)` with position-independent arithmetic — which removes the
   fp4 dot-product replication risk entirely. If the wrapper/kit call cannot express this (vis semantics,
   specialization constants, width bounds), **stop and report back**; do NOT reimplement `_index_score`'s
   scoring arithmetic blind. That is the escalation path, not a fallback you pick silently.
3. **Top-k with original ids** (new variant of `tf_ds_topk_indices_kernel` taking a `cols` map, or an
   equivalent): select `kk` picks keyed on (score bits, inverted **original** column) in `idx_fused.scoreKey`'s
   order, and write `cidx` **byte-identically** to the dense `topkIndices` — same output order, same -1/vis
   handling, same behavior for invalid entries and for rows where fewer than `kk` columns are selectable (the
   dense path fills from -inf columns; read the kernel and its references and reproduce exactly what it does,
   ties to the lower original column included).

Fallback (switch off, shape over bounds, pool not active): today's chain, untouched. The candidate source
layer itself (20), the keyed/pruned arm, the split-index arm and the prompt bypass stay exactly as they are.

## Parity contract

`cidx` bytes for every row must equal the dense chain's (`indexScore` full + `applyPool` + `topkIndices`)
across the test matrix: **cutoff ties** (equal scores straddling `kk`), **invalid entries** (all--inf rows,
blocks whose maximum is finite but whose columns are -inf/NaN), **partially visible blocks** (a retained block
whose later columns sit at or past the row's `vis`), rows retaining fewer than `kk` columns, `vis` edges
(0, mid-block, exactly at a block boundary), and multi-row blocks with differing masks. The scoring step is
compared through the same kit kernel on both sides (point 2), so the tests here cover selection + compaction
equivalence given identical scores.

## Host reference + tests (`pool_score.zig`)

- `retainedPicks(scores, mask, vis, kk)` (compaction + id-carrying selection semantics) vs `densePicks(scores,
  mask, vis, kk)` (an independent transcription of `applyPool` + `topkIndices` semantics read from the
  kernels) — byte-equal `cidx` over the matrix above.
- The mask rule the compaction uses vs `applyPool`'s: a column is scored iff its dense score survives
  `applyPool` (and `vis`); assert set equality incl. partially visible blocks.

## CPU-emulation harness (`runs/kernel-round-20261122/h0262/`)

Extract and execute the compaction and map-topk kernels' sources on CPU against independent references. For
the scoring step (a kit cubin, not locally executable), verify the compaction's gathered key bytes equal the
dense buffer's bytes at the original columns (per-column `(q, k, w)` inputs identical), and run the selection
comparison with a shared stub score function so the pick parity is still exercised end to end.

## Known risk to record in the report

The only hardware-gateable property is the end-to-end byte compare against the dense chain (it is in the
overnight plan). If the kit `_index_score_rows` cubin's specialization differs from what the dense path
launches (constants like KEYS/HAS_CAND, width handling), point 2's "by construction" claim fails and the
patch must not ship — the report should say exactly which invocation the dense path uses (`tri_index.zig`'s
`indexScore` + `fixtures/launches.json` conformance) and what you invoked.

## Out of scope

Scratch-sharing/arena reductions (todo.md's separate items), the mask-only arm, the contiguous-width timing
work (benchmark side), the `0254` selection internals beyond the variant you add, and any change to the
candidate source layer.
