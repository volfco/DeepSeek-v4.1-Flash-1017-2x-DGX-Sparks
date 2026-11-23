# BRIEF-0279 — TF_DS_IDX_PICK_WIDE: the indexer's small-row selection past 16,384 blocks (long-context decode)

**Switch:** `TF_DS_IDX_PICK_WIDE` (off by default). **Module:** `zig/src/families/dsv41/idx_pick_wide.zig`.
**Commit subject:** `0279 dsv41: TF_DS_IDX_PICK_WIDE (off by default): the indexer's small-row selection up to 65,536 blocks`

- Context: `TF_DS_IDX_FUSED` (on in production; `idx_fused.zig`) covers nbk ≤ `max_nbk` = 16,384 blocks (131,072 columns,
  a 262,144-token bucket). Past it every decode round falls back to the unfused three launches
  (`tf_ds_block_max_kernel`, `tf_ds_topk_indices_kernel`, `tf_ds_pool_pick_kernel`). Production serves up to 1M tokens.
  The bound is `tf_ds_idx_pick_kernel`'s on-chip staging of the maxima (101,376 B opt-in shared = 25,344 fp32); the radix
  shortcut holds to 65,536 blocks (`idx_fused.zig` doc comment), and `todo.md` ("Extend 0195 to 100K+ contexts") says
  65,537 must keep the unfused fallback.
- With the switch on, for decode rows ≤ `idx_fused.small_rows` and `max_nbk < nbk ≤ 65,536`: the small-row form's first
  launch (block_max over the 2D grid) is unchanged; its selection reads the maxima from global memory (L2-resident:
  ≤ 256 KiB a row) instead of staging all of them in shared memory (a variant of `tf_ds_idx_pick_kernel`, or the same
  kernel with a global-maxima mode — do not change its existing behaviour). The output block list (`cblk`) and `every`
  must be **byte-identical** to the unfused sequence's, with the same total order on `(score bits, inverted column)`
  including ties.
- Show from the code which call sites hit nbk > 16,384 at the served config (decode, rows 1–7 a stream, up to 4 streams)
  and at what context lengths. If the unfused fallback is not actually reached at served shapes, stop and write that as
  the report.
- Harness: CPU emulation of the new selection vs an independent reference (sort by the total order) at nbk 16,385,
  32,768, 65,536, plus 65,537 (must refuse → fallback), rows 1/7, adversarial ties (equal maxima, signed zeros, -inf), k =
  candidate_blocks (2048).
- Report expected time: the existing small-row note measured 10.5–11.8 µs at 8,192 blocks fused vs 12.1–12.4 unfused;
  estimate per-row cost at 32K/64K blocks from the memory traffic, and say what only the GPU gate can settle.
