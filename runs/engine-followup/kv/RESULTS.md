# Admission under retained KV pressure at 300K

On the pinned production image, the last of four warm requests waited about 25 seconds at peak retained occupancy. Each burst contained the same four follow-up tasks. Three independent process resets reproduced the result. In a separate pair of bursts, placing the shared prefix on the other KV partition and concentrating three independent 300K reservations on part 0 produced the same delay. Both low-occupancy controls had a maximum queue below 4.4 seconds.

Each client requested 1,024 tokens with thinking enabled, temperature 1, top_p 0.95, top_k 20 and a fixed seed. Every request returned 1,024 tokens with a 299,008-token KV hit and zero restore time. Within each comparison, all four input digests and output token hashes matched. The part-1 workload uses a different marker, so it has its own matched low control. Each burst was the only load observed on the pair. [Three peak trials](run.log) · [Part-1 trial](partition.log)

| Shared prefix | Kept rows before burst, part 0 / part 1 | Longest queue (s) | Common four-way decode (s) | Aggregate output (tok/s) | New no-target refusals | States marked stale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Part 0, low | 319,488 / 0 | 4.369 | 30.50 | 104.42 | 0 | 0 |
| Part 0, peak 1 | 618,496 / 598,016 | 24.745 | 2.56 | 87.96 | 21 | 3 |
| Part 0, peak 2 | 598,016 / 598,016 | 24.851 | 4.09 | 87.98 | 21 | 1 |
| Part 0, peak 3 | 598,016 / 598,016 | 24.812 | 2.96 | 87.65 | 21 | 1 |
| Part 1, low | 20,480 / 299,008 | 4.251 | 26.33 | 107.70 | 0 | 0 |
| Part 1, 3:1 split | 917,504 / 299,008 | 24.949 | 1.90 | 92.37 | 20 | 2 |

After each peak, all three independent sessions returned for a 16-token turn. The client had measured the same request as a memory hit before the peak. Five of twelve revisits read their 299,008-token state from disk. Their engine restore times were 0.161–0.167 s and their paired client TTFT rose by 0.143–0.170 s. The other seven revisits were served from memory. **Every session retained its full KV hit.** The earlier study's three stales therefore do not imply three re-prefills. Revisit order rotated across the three part-0 replicates. A later probe sees any placement changes made by the earlier probes. [Per-burst transitions and paired revisits](summary.json) · [Asymmetric placement and revisits](partition-summary.json)

For example, in the first peak, the sampled stale counter grew by two around 15 seconds after the burst started, and by one around 27 seconds. The recorded part-0 kept rows and then part-1 kept rows fell over those intervals. The engine exports refusals and stales as global cumulative counters. It exports free, live, kept, twin and held rows individually by part. Sub-second ordering and the exact requesting lane for each refusal cannot be recovered from the roughly one-second health samples. Supplement these traces with planner instrumentation for a policy comparison. The incumbent's 10 s held-prefix patience is a code constant; its switch for protecting shared states changes victim selection. The benchmark build below changes the sharded held-prefix deadline and measures returning sessions under both policies. [Fixture and health sampling](pressure.py) · [Partition fixture](partition.py) · [Offline analyzer](summarize.py)

Two gaps between benchmark campaigns contained zero recorded request completions, across 32m26s and 67m19s. Those intervals provide no arrival distribution for extrapolating the throughput result to real traffic. Only aggregate counts and the interval endpoints were retained. [Archived interval](archive-window.json) · [Current-container interval](summary.json)

Both campaigns restored the original image and settings on both ranks. The final health check found zero connected or queued clients. The restored server passed all eight saved reply comparisons. [Independent-reset validation](restore.validate.log) · [Partition trial validation](partition-restore.validate.log)

## Timing the held-prefix deadline

On an isolated build of the pinned engine, reducing the sharded planner’s held-prefix deadline from 10 seconds to 500 ms let four warm clients overlap for 27.86 seconds and improved aggregate output by 18.6% against the mean of the two controls. The return to 10 seconds reproduced the delay. All three arms began with 598,016 kept rows per part. All 12 responses generated 1,024 tokens, reused 299,008 cached tokens and matched their corresponding input and output hashes. [Raw trial](policy01.log) · [Arm and revisit measurements](policy-summary.json)

| Held-prefix deadline | Longest queue (s) | Last first token (s) | Four-way overlap (s) | Burst wall (s) | Output (tok/s) | No-target refusals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 s, first control | 25.010 | 26.759 | 3.421 | 46.976 | 87.19 | 21 |
| 500 ms | 6.320 | 8.050 | 27.860 | 39.519 | 103.65 | 3 |
| 10 s, return control | 24.876 | 26.601 | 3.323 | 46.786 | 87.55 | 21 |

Placement instrumentation on part 0 recorded two late successful resumes per burst. They occurred 10.22–10.40 seconds after the first wait with the original deadline, and 1.016–1.034 seconds after the first wait with 500 ms. Retry cadence determines when the planner acts after expiry. Each arm marked one state stale. In all three arms the same one of the three independent returning histories restored its full prefix from disk (0.162–0.173 seconds), the other two hit memory and every revisit returned the original tokens. [Instrumented source change](held-wait.patch) · [Revisit details](policy-summary.json)

The trial used the same benchmark binary on both GPUs in every arm. The startup banner and both container environments acknowledged the requested deadline. The production image, flags and original spill directory were restored on both ranks, with all eight saved replies passing on each arm and after restoration. [Policy driver](policy.py) · [Build digest](head.binary) · [Restored validation](policy01/production.validate.log)

A second comparison changed the placement and reversed the test order. The shared target occupied part 1, with 897,024 independent kept rows on part 0 and 299,008 on part 1 before each burst. The 500 ms arm again cut the longest queue by 18.6 seconds, reduced the burst wall time by 15.1% and raised aggregate throughput by 17.8%. All four input hashes, token counts, hit lengths and output hashes matched between arms. The same one of three returning sessions used spill in both arms, at 0.172 s and 0.164 s engine restore time respectively. Every returning session preserved its full hit and response hash. [Reverse-order trial](policy-partition01.log) · [Paired revisit, timing and planner counters](policy-partition-summary.json)

| Held-prefix deadline | Longest queue (s) | Last first token (s) | Four-way overlap (s) | Burst wall (s) | Output (tok/s) | No-target refusals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 ms, part 1 | 6.367 | 8.124 | 26.977 | 38.880 | 105.35 | 2 |
| 10 s, part 1 | 24.997 | 26.732 | 2.826 | 45.803 | 89.43 | 20 |

The 500 ms build passed all 39 exact cases with the flag armed, both arms passed the eight saved replies and the original two ranks passed after restoration. These runs establish a candidate for deep shared-prefix workloads with spill available. A rollout decision still needs representative arrival and revisit data, especially for sessions whose displaced prefix would require a cold prefill. [Armed exact comparison](policy-partition01/earlier.exact-compare.log) · [Armed validation](policy-partition01/earlier.validate.log) · [Restored validation](policy-partition01/production.validate.log)
