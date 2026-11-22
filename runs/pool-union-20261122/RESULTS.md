# Candidate 3 closed, November 22: the pooled prompt indexer only runs on the ≤128 bounded-replay rows

**Result.** The `TF_DS_POOL_UNION_STAT` diagnostic (branch `pool-union-stat` `34cc705`, image `bench-fused-next-34cc705`, one restart, production restored and validated) fired **only twice per cold prompt, on chunks of 104 + 24, 91 + 37 and 93 + 35 rows. Each pair is exactly 128 rows.** The pooled (unkeyed) indexer path, which candidate 3 would shrink, runs only for each prompt's bounded-replay tail (128 rows, the decoder-bound rows). The bulk of every prompt goes through the keyed path (`prompt_keys`: `indexScore` with keys + `topkSelectPruned`), where no candidate pool exists. Priced on that scope, candidate 3 saves about 128 rows × 4 layers × ≤30 µs ≈ 15–20 ms per cold 160K prompt, **about 0.02% of its 80.7 s prefill**. Not worth building. [run.sh](run.sh) · [union.log](union.log)

The earlier "~10% of long prefill" estimate in [prefill-pool-score](../prefill-pool-score-20261122/RESULTS.md) assumed every prompt row took the pooled path. It does not.

## Union of pool blocks across replay rows (candidate source, layer 20)

| Prompt tokens | Chunk rows | Width (blocks) | Per-row pool | Union per 128-row block | Union per 16-row block |
|---:|---:|---:|---:|---:|---:|
| 24,600 | 104 + 24 | 3,072 | 2,048 | 3,072 / 3,067 (100% / 99.7%) | 98.3% / 93.4% |
| 49,176 | 104 + 24 | 6,144 | 2,048 | 96.0% / 87.0% | 70.6% / 72.9% |
| 98,341 | 91 + 37 | 12,288 | 2,048 | 81.7% / 72.0% | 51.6% / 47.1% |
| 163,875 | 93 + 35 | 20,480 | 2,048 | 68.7% / 56.4% | 38.4% / 33.4% |

Adjacent rows' pools overlap far less than expected. At 160K a 128-row block's union is 57–69% of the width, about 7 times one row's pool, so even on its own scope a union-gather scorer would save only about a third. Per-16-row unions run 33–98% of the width.

## Cold prefill on production's image (the diagnostic's syncs included)

| Prompt tokens | Seconds | Tokens/s |
|---:|---:|---:|
| 24,600 | 12.4 | 1,984 |
| 49,176 | 23.8 | 2,066 |
| 98,341 | 48.2 | 2,040 |
| 163,875 | 80.7 | 2,031 |

## Where prefill indexing cost actually is

The keyed path scores every visible compressed column for every row at each index layer (2/8/14 at ratio 2, then 20/24/28/32/36), with pruned top-k on the keys. It has no pool. That is the real prefill indexer cost and it is unpriced. The next prefill question is its share of a chunk (the eager phase timer does not cover prompts; the served path prints no `prefill_phases_ms`).
