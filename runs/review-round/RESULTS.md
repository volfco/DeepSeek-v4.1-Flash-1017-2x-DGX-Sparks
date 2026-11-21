# November 17–18 inference benchmarks (UTC)

All serving tests in this round used the same TensorFold image, `tensorfold-dsv41:exl3-tp2-234f9080b326`, on two DGX Sparks. The clients recorded rank image IDs, relevant switches, streamed first and last tokens, actual prompt and cached tokens, completion counts, and engine done lines joined by request ID. A serving cell required an idle start, exactly its own completions, an idle finish and healthy ranks. The served trials recorded request timings and memory snapshots. They did not log GPU clocks continuously; use the return blocks to check drift. These are synthetic tasks, so the fraction of production traffic affected remains unmeasured.

## Sampled warm turns

Production → `best` → production on the pinned image. Each block used the same natural history and three coding tasks at two depths, each with a fixed seed and 256 output tokens, thinking on, temperature 1, top_p .95 and top_k 20. Each depth also had a simultaneous four-request burst. Independent one-digit salts prevented cross-arm KV reuse and changed the input text. Compare the block pairs for magnitude and drift. A salt can change sampled output and speculative acceptance even when task, seed, length and cache cut match.

The candidate passed the 39-case greedy exact comparison against combo3's all-off reference and the eight saved validate cases. Its startup smoke passed on the first attempt. Returning to the original settings produced the previously observed first-smoke failure; the retry succeeded. The orchestrator compared both ranks' image, command and full environment at restoration, persisting only a digest and the public test switches. [Exact comparison](best-b.exact-compare.log) · [Trial log](round.log)

| Warm workload | Production A | Best | Production return |
| --- | ---: | ---: | ---: |
| 4K solo TTFT, seconds | 1.273 | 1.203 | 1.276 |
| 100K solo TTFT, seconds | 1.290 | 1.234 | 1.300 |
| 100K solo output tok/s | 69.43 | 64.78 | 67.78 |
| 100K solo tokens/round | 2.59 | 2.35 | 2.51 |
| 4K burst, generated tokens per wall second | 52.38 | 54.85 | 52.62 |
| 100K burst, generated tokens per wall second | 51.45 | 50.81 | 50.91 |

Median paired warm TTFT improved 5.5% at 4K and 4.6% at 100K. The warm 100K turns resumed at 98,304 of 99,788–99,789 tokens; prefill improved 5.1%. Encoding took 90/93/93 ms at 100K. The history and task were concatenated into a single changing user message, so this workload gives little opportunity to reuse the large tokenizer piece. All six shallow/deep bursts lacked a common four-stream decode interval. This trial measures burst completion under admission delay. [Paired results](sampled.csv) · [Production](production-a.jsonl) · [Best](best-b.jsonl) · [Return](production-c.jsonl)

**Same-input replay.** Production then replayed the candidate's salt, corpus, tasks and seeds. All 14 warm replies had the same output token SHA, prompt length, cached length and generated length as the candidate. At 100K the three solo requests also had the same tokens/round. Median paired TTFT was 4.9% lower on best, and sampled output rate was 1.1% higher. At 4K the paired differences were −5.8% TTFT and −1.7% output rate. The apparent 100K decode loss in the fresh-salt bracket disappears under identical output. Rank the configuration on the reproducible prefill/TTFT benefit. The small decode differences change direction with depth. One matched replay per task cannot resolve them. [Same-input production replay](production-replay.jsonl) · [Candidate](best-b.jsonl)

## Admission under simultaneous arrivals

On restored production, 16 four-request bursts crossed shared/independent warm 4K prefixes, greedy/sampled generation, and 256/512 output budgets. Each cell ran twice in reversed order. All 64 replies filled their budget; each prefill took 1.26–1.29 s. No KV no-target refusals occurred. Both parts retained at least 952,320 free rows throughout the sampled trace.

| Output limit | Bursts | Highest live lane count | Common decode interval | Longest request queue per burst | Burst throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 8 | 1 in all eight | 0 s | 14.00–14.64 s | 52.2–55.1 tok/s |
| 512 | 8 | 4 in all eight | 13.21–14.45 s | 3.82–3.84 s | 97.7–104.6 tok/s |

Prefix sharing and sampling did not change this split. After a prefill of at least one second, the lane host gives a running request with at most 256 tokens left up to five seconds to finish before admitting the next request. Repeated prefills can repeatedly trigger the delay. The different amounts of output limit a direct speed comparison between these two budgets. The behavior points to admission scheduling as a throughput and TTFT target for short queued replies. [Request and KV traces](admission.jsonl) · [Near-end check](../../zig/patches/0005-prefill-interleave.patch) · [Finish-window constants](../../zig/patches/0014-core-cache-tier.patch)

### One-token boundary

A second trial held the same four sampled prompts, seeds and cached positions fixed and ran 256/257/258/258/257/256 in order. Every reply generated its entire requested output and all prefills exceeded one second. Every cell had zero new KV placement refusals.

| Requested tokens per reply | Highest live lane count | Longest queue in each burst | Wall seconds per burst | Generated tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1, 1 | 14.70 / 15.08 s | 19.79 / 19.81 | 51.74 / 51.68 |
| 257 | 1, 1 | 15.02 / 14.72 s | 19.74 / 19.82 | 52.08 / 51.87 |
| 258 | 4, 4 | 3.81 / 3.80 s | 12.92 / 12.86 | 79.85 / 80.25 |

One extra requested token (257 → 258) reduced burst completion time ~35% and raised aggregate throughput ~54%. The first token produced by prefill leaves 256 versus 257 tokens of budget. This crosses the source's near-end check. Evaluate a change to the admission grace on the same fixed budgets, then test mixed arrivals and measure the active stream's inter-token stalls as well as queued users' TTFT. Changing clients' output limits changes their generation contract. [Threshold traces](threshold.jsonl)

### Shared 300K history

Four sampled bursts used the same ~300,576-token history, tasks and seeds, with 299,008 tokens cached in all 16 requests. The budgets ran in order 256/1,024/1,024/256. Each burst began with 428,032 kept rows across the parts, compared with ~1.04M before the earlier high-occupancy repeats. Restore time was zero for every request. No no-target refusals occurred.

| Output limit | Peak live lanes | Common decode | Longest queue | Generated tokens/wall second |
| --- | ---: | ---: | ---: | ---: |
| 256, two bursts | 1, 1 | 0 s, 0 s | 16.22 s, 15.83 s | 47.39, 47.60 |
| 1,024, two bursts | 4, 4 | 30.63 s, 30.75 s | 4.33 s, 4.31 s | 103.55, 104.17 |

The more crowded part reached 43,008 free rows during four-way decode. This run establishes that four 300K streams can decode together at this occupancy. The earlier same-depth 1,024-token replicates with more kept rows had no common interval and 54 new no-target refusals. Those runs differed in traffic history and sampling. The controlled replay below uses this exact fixture. [Deep traces](deep-admission.jsonl) · [Earlier depth study](../depth/RESULTS.md)

### Controlled KV occupancy

On one production process, two 1,024-token bursts started with 319,488 kept rows on part 0 and none on part 1. Three independently primed 300K histories raised the total to 1,216,512. The same four requests then ran twice more. Every replay resumed at 299,008, generated 1,024 tokens and reproduced the same per-seed output token SHA. Restore time was zero throughout.

| Initial kept rows, part 0 / part 1 | Longest queue | New no-target refusals | Common four-way decode | Burst throughput |
| --- | ---: | ---: | ---: | ---: |
| 319,488 / 0, first control | 4.45 s | 0 | 30.36 s | 104.24 tok/s |
| 319,488 / 0, second control | 4.31 s | 0 | 30.54 s | 105.14 tok/s |
| 618,496 / 598,016 | 24.87 s | 21 | 2.77 s | 86.88 tok/s |
| 299,008 / 299,008, after reclamation | 14.61 s | 10 | 18.39 s | 94.47 tok/s |

This reproduces the long queue by adding independent kept states while holding actual input, output and cache hit fixed. The first pressured burst marked three states stale. As a result, the fourth burst began at intermediate occupancy and is a recovery observation. Only one burst began at the highest occupancy. Further work should trace each refusal, reclaim and retry by part, reset the target occupancy before each replicate, and price any earlier reclamation against the warmed sessions it evicts. [Request and placement traces](occupancy.jsonl)

## Token cache on a 250K conversation

This trial changed only `TENSORFOLD_TOKEN_CACHE_MB` between production off, 32 MiB, and the return to off. A fixed long user message and assistant reply preceded a changing final user message. Each arm ran three seeded requests at each of 100, 500 and 2,000 appended tokens, rotating their order. The benchmark checked serialized request hashes, prompt lengths, and an exact 249,856-token KV cut. Cold priming and disk-restored priming are recorded separately. They had different cache cuts and are excluded from the warm comparison. [Off](cache-off-a.jsonl) · [On](cache-on-b.jsonl)

| Appended tokens | Encode ms, off/on/off | TTFT ms, off/on/off | Median paired TTFT reduction vs bracket |
| ---: | ---: | ---: | ---: |
| 100 | 236 / 4 / 224 | 826 / 595 / 816 | 226 ms (27.5%) |
| 500 | 241 / 4 / 223 | 1,061 / 829 / 1,042 | 229 ms (21.6%) |
| 2,000 | 244 / 7 / 225 | 2,125 / 1,900 / 2,119 | 206 ms (9.8%) |

All nine warm output token hashes matched between all three arms. Both the candidate and the restored server passed validate 8/8. The ~228 ms encoding reduction is repeatable for this immutable long message. On the single changing 100K user message in the depth trial, encoding was nearly unchanged. Use the immutable-history case to size the deployment benefit for multi-turn chats. [Return block](cache-off-c.jsonl) · [Candidate validate](cache-on-b.validate.log) · [Restored validate](cache-off-c.validate.log) · [Rank restoration](cache-round.log)

## Applied-cap T15

The comparison uses the spill-free batch driver, graphs enabled and profiling disabled. Both ranks keep `TF_DS_T13_EXPERT_PRUNE=1 TF_DS_T13_EXPERT_CAP=12,1` throughout. Only `TF_DS_T15_PLACE_BOUND` changes. At eight fixed rows the pinned code limits the logical launch from 56 slots to 21 places. This is calculated from the verified flags and shape; the uninstrumented driver exposes no GPU launch counter. Compare each output SHA against the same capped control. A T15 result alone cannot price the quality of introducing the cap.

The first control was rejected. `TF_DS_FIX_K=0` still allowed 237 suffix-copy draft tokens (151 accepted), yielding 408 shared rounds. The corrected trial adds `--serial 1` to disable drafting per stream while retaining `--parallel 8`. The pilot's two original containers restarted. Docker reordered the mount list on inspection; comparing every mount without regard to order reproduced both original signatures, and validation passed 8/8. [Excluded pilot](t15-r8/a1.rank0.log) · [Pilot restore validation](t15-pilot-restore.validate.log)

The corrected schedule used T15 off/on/on/off on four fresh loads. Each load ran one 32-token warm-up wave followed by three 128-token waves. All four loads completed exactly 412 shared decode rounds with zero drafts. All 32 per-request token hashes matched across loads, and both rank signatures matched across arms after excluding the T15 switch.

| Median seconds per timed wave | Off A1 | On B1 | On B2 | Off A2 |
| --- | ---: | ---: | ---: | ---: |
| First request admission through completion | 7.67 | 7.67 | 7.65 | 7.66 |
| Last prefill through end of decode (127 rounds) | 4.18 | 4.18 | 4.17 | 4.19 |

Bounding 56 logical slots to 21 reduced the paired median decode window by 0.0% and 0.5%. Individual waves extended to 5.48 s on B1 and 4.63 s on A2. Changes of this size are below the 3% decode timing threshold. Deprioritize T15 for this shape. The cost of the cap itself, its quality on independent continuations and any effect on speculative served shapes require separate comparisons. [Arm settings and raw timings](t15-r8-fixed.jsonl) · [Batch driver](t15.py)

The offline run took both GPUs. At 00:14:44 UTC on November 18, both original serving containers were running again with their pre-run image, command, environment, host configuration and mounts. At 00:15:01 all eight saved validation replies matched, and all eight concurrent replies matched their solo controls. The validation suite skipped video because PyAV was unavailable. [Final production validation](t15-r8-fixed/restore.validate.log)

## Reproduction

Run the clients from the recipe root on the head. They read the existing `runs/combo4/corpus` and join done lines locally. Use one fresh single-digit salt per arm of the sampled depth trial. For the piece-cache trial reuse the same warm-chat fixture across restarts. The [client](sample.py) records raw JSONL and the [comparator](compare.py) checks pairing before summarizing. The [restart driver](round.py) takes a fresh `REVIEW_RUN_DIR` for repeat runs, pins the existing image and restores production settings.
