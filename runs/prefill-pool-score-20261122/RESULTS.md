# Candidate 3, November 22: prefill scores 4–8× more columns than it keeps; the kit's pool-only scorer is fast but not bit-equal

**Result.** At the four prompt indexer layers after the candidate source (24/28/32/36), prefill scores every compressed column with the kit's `_index_score_rows` and then masks all but the pool's 2,048 blocks (16,384 columns) to −inf. Scoring is linear in width, so at 64K and 128K deep rows 75% and 87.5% of that work is discarded. The kit already has a pool-only scorer (`_index_score_cand`, the decode form). It runs in constant time at any width and gets columns and visibility right. **Its score bits differ from `_index_score_rows` on about 60% of columns (1 to 12+ ULP)**, so substituting it changes selections near ties and would not be bit-identical to production. A bit-exact path has to keep `_index_score_rows`' arithmetic. Options are below.

## Kit prompt scorer cost (GB10, standalone, `timing_valid`, idle head)

`_index_score_rows`, variant `739ce7` (KEYS=0, fp32 scores; the one prompt blocks of more than 16 rows launch), 128 rows:

| Width (compressed columns) | Median | Per row per layer |
|---:|---:|---:|
| 16,384 | 468.0 µs | 3.66 µs |
| 32,768 | 936.9 µs | 7.32 µs |
| 65,536 | 1,910.2 µs | 14.92 µs |
| 131,072 | 3,825.7 µs | 29.89 µs |

Across the four pooled layers, a 128K-deep prompt row spends about 120 µs scoring, of which about 105 µs is discarded by the pool mask. At roughly 0.5–1 ms of prefill per token this is about 10% of long-context prefill (an estimate from standalone timings, not a serving measurement). [idxscore.json](idxscore.json)

## Pool-only scorer vs prompt scorer (parity probe)

16 rows, a causal visibility per row (`width − 64·r − 3`), 2,043 random sorted pool blocks per row plus 5 empty slots. Key format: the kit's key is the ordered score bits (−0 as +0, negatives `^ 0x7FFFFFFF`) above the inverted column, with **no** top-bit flip. idx_fused.zig's `scoreKey` flips it, as noted for 0206.

| Width | Compared columns | Score bits differ | Past-vis mismatches | Column half wrong | `_index_score_rows` (16 rows, full) | `_index_score_cand` (pool) |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 261,504 | 154,860 (59%) | 0 | 0 | 118.6 µs | 66.3 µs |
| 65,536 | 261,504 | 156,651 (60%) | 0 | 0 | 234.4 µs | 68.2 µs |
| 131,072 | 261,504 | 156,466 (60%) | 0 | 0 | 466.9 µs | 68.4 µs |

The bit-distance histogram is dominated by 1 ULP (about 84K), then 2 (35K), 4 (11K) and 8 (5K), with a tail past 12. The two kernels sum the heads/dimensions in different orders (`_index_score_rows` uses `TREE=1`). Empty slots come out as key `0x8000000000000000`. [idxcand2.json](idxcand2.json) (the earlier [idxcand.json](idxcand.json) used the flipped key format and is superseded).

## Options for a bit-exact pool-only prompt scorer

1. **Compact the keys, keep the kernel.** Each column's score in `_index_score_rows` depends only on that column's key, the row's query and the weights, so scoring a compacted key list should reproduce the bits. The pool differs per row while K is shared per launch, so this means one 1-row launch per row (grid 1 × 256 at 16,384 columns), plus gathering 16,384 × 68 B of keys per row. That is about 1.1 MB per row and roughly 0.7 ms of copies per 128-row block at 200 GB/s. Gathering per pool block shared by many rows would cut that. **Check first:** that a compacted call reproduces the bits (a standalone probe, as above).
2. **A repo kernel that reproduces `_index_score_rows`' TREE reduction.** It would read keys through the row's block list, like the decode fetch. This is bit-exact only if the Triton reduction order is replicated (the 0197 RMSNorm repair shows it can be done, with work). Read the kit's Triton source for `_index_score_rows` first.
3. **Accept the decode scorer's arithmetic in prefill.** This is not bit-identical: it changes long-prompt selections near ties and needs the quality gates, not the exact panel. It is the fastest to build (the kit kernels plus 0195's block list for prompt rows) and is the operator's call.

Either exact option then needs `topkIndices` over the compact scores mapped back through the sorted block list. Ties stay on the lower column because the list ascends. Add a fallback when a row has fewer than `kk` finite pool scores, where the full-width path would pick −inf columns.
