# WP1: the 0195 indexer (`tf_ds_idx_fused_kernel`, `idx_fused.zig`, the candidate-source arms in `round.zig`/`prompt.zig`)

Evidence folder: `runs/fused-next-20261121/wp1/`. Engine commits numbered from **0213**. Use the WP0 harness
(`tools/fused_bench/`, `wp0/BASELINE.md`) for every parity check and timing; extend it where an item needs a case.
This kernel is the one decode-relevant, work-bound fused kernel (9.8 µs at 4,096 blocks), so it is where the
microbenchmarks can pay: be thorough.

Items, in this order (todo.md text is the spec; read each item there in full):

1. **Fix and re-land the key fold (old 0206, parked in the recipe repo's `zig/patches-parked/`).** At the served block
   size the fused pool launch can also write `_score_keys`' outputs (every column's key and each 64-key tile's max) and
   skip that kit launch. It changed 4 of 39 long-prompt replies on the pair because it wrote `idx_fused.zig:scoreKey`'s
   encoding (top bit XORed) where the kit's `topk_select_pruned` reads `_score_keys`' encoding. Measure the kit's bytes
   (`_score_keys` cubin) as the reference, pin the width window in which the fold engages (why does `len_100000` still
   match?), write the kit's encoding, and prove byte identity of keys and tile maxima against the kit cubin at every
   width in the window including partial tiles. Then decide with timings whether the fold pays: it costs key writes
   (316.6 µs vs 38.5 at R=128 width 32,768 in the gate); price the kit launch it removes (WP0's number). Re-land the high-word
   cache (old 0208) only if the fold lands and the cache measures faster.
2. **Compile-time block size + vectorized loads** (todo "Specialize 0195's block size" and "Specialize and vectorize
   aligned served-shape loads"): template on `bsize == 8`, two aligned `float4` loads per block with guarded scalar
   tails, unchanged comparison/NaN order; SASS must show LDG.E.128 and a fully unrolled body.
3. **Warp-aggregated histogram** (`__match_any_sync` + leader `atomicAdd`): measure random, tied and flat inputs and
   256/512/1024-thread CTAs; keep only if it wins on the served distribution.
4. **Mask-only prompt path skips compaction** and **cutoff ties resolved during decode compaction** (todo
   "Additional 0195+ performance findings", Prompt indexing): exact mask/`cblk` parity including newest block, ties,
   NaN, -inf.
5. **Small-row parallelism** (todo research item): two-phase or row-per-warp-group form for R = 1..7 with 4,096 and
   8,192 blocks; the merge must reproduce the total order on (score bits, inverted column) exactly.
6. **Bypass prompt scoring when every compressed key fits** and **score only the retained blocks in later prompt
   indexers** (Prompt indexing items): these remove whole launches; check 511/512/513 columns and partially visible
   blocks.
7. **100K+ contexts** (research item "Extend 0195 to 100K+ contexts"): larger opt-in shared allocation, parity at
   8,192/8,193 and 65,536/65,537 blocks, unfused fallback beyond.

For each item: parity, interleaved timings with the launch floor, SASS diff, bound, verdict, commit or "closed, no
change". Finish with `wp1/RESULTS.md` summarizing all seven and the kernel's final position against its roofline.
