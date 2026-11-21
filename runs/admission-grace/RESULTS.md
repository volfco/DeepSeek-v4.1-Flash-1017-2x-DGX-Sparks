# Admission grace at fixed output limits

The earlier 257/258-token boundary changed both scheduling and the reply budget. This experiment changes the scheduler on top of the incumbent eight-switch `best` configuration. All arms use the same requests, output limits, client and benchmark binary. The unmodified pinned engine uses a five-second grace after a prefill taking at least one second when an active stream has at most 256 tokens left. [Boundary evidence](../review-round/RESULTS.md#one-token-boundary) · [Benchmark-only engine patch](finish-grace.patch)

## Conditions

| Workload | Arrivals and requested output |
| --- | --- |
| Solo control | One seeded 256-token reply per loaded configuration |
| Simultaneous | Four seeded 256-token replies behind a common start barrier |
| Short first reply | A 256-token first request. Three 256-token followers arrive 0.20, 0.55 and 0.90 seconds after its first streamed text. |
| Long first reply | A 640-token first request. The same three 256-token followers arrive at the same offsets. |

Each block primes a shared natural 4K context, then runs the three group workloads forward and in reverse order. The 2,048-token cache cut stays fixed. All four requests use thinking, temperature 1, top_p 0.95 and top_k 20. The client joins prompt, output and queue counts to the server’s completion by request ID. Every measured reply must finish at its requested length. The largest client stream gap measures time between nonempty SSE text or reasoning events. One event can include multiple tokens. [Client](../review-round/sample.py) · [Paired analysis](summarize.py)

## 4K results

Each entry is the median of two bursts at the candidate setting or four across the two five-second control blocks. Follower TTFT is the slowest follower’s time to first text, measured from its own arrival. For simultaneous arrivals the first stream to emit is counted as the lead. All 24 burst outputs per candidate and all 24 in the return control matched their corresponding first-control token hashes. No KV placement refusal occurred in any burst. [Per-burst measurements](4k.csv)

| Workload | Grace | Aggregate tok/s | Batch wall, s | Slowest follower TTFT, s | Lead completes, s | Lead’s longest gap, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simultaneous | 5 s | 52.1 | 19.64 | 15.96 | 5.03 | 0.05 |
| Simultaneous | 500 ms | 71.8 | 14.34 | 6.83 | 13.27 | 1.59 |
| Simultaneous | 0 | 75.5 | 13.67 | 4.91 | 13.22 | 3.79 |
| Short first reply | 5 s | 52.7 | 19.41 | 13.51 | 5.02 | 0.05 |
| Short first reply | 500 ms | 76.4 | 13.40 | 4.28 | 12.28 | 1.27 |
| Short first reply | 0 | 80.5 | 12.73 | 2.89 | 12.70 | 3.64 |
| Long first reply | 5 s | 70.4 | 20.01 | 12.90 | 18.92 | 1.27 |
| Long first reply | 500 ms | 75.7 | 18.62 | 3.95 | 18.62 | 1.29 |
| Long first reply | 0 | 76.1 | 18.50 | 2.90 | 18.50 | 3.66 |

With a short first reply, 500 ms raised throughput 45% and cut the slowest follower’s TTFT 68%. The active user’s completion moved from 5.02 to 12.28 s, so mean request completion actually rose from 10.86 to 11.24 s. For a 640-token lead, 500 ms reduced the slowest follower’s TTFT from 12.90 to 3.95 s while the lead’s longest gap stayed near 1.3 s. Zero grace delivered the highest throughput in these runs and produced the longest active-stream stalls.

All three grace settings matched the 39-case greedy reference. All four blocks matched the saved validation replies. GPU clocks sampled during the request workload were 2366–2392 MHz on the head and 2359–2385 MHz on the worker. [5 s exact](run01/control-a.exact.log) · [500 ms exact](run01/half-second.exact.log) · [Zero exact](run01/zero.exact.log) · [Return validation](run01/control-b.validate.log)

## 100K warm turns

The follow-up compared 5,000 / 500 / 5,000 ms on the same benchmark image. All four staggered requests asked for 256 tokens and resumed at exactly 98,304 cached tokens. The short tails were 386–387 actual uncached tokens and the long tails 1,994–1,995. The driver ran both tails forward and backward. Every measured burst started at 98,304 kept rows on one part and zero on the other, with zero restore time and zero new KV placement refusals. All 16 candidate output hashes and all 16 return-control hashes matched the first control. [Per-burst measurements](100k.csv) · [Request traces](depth01/half-second.jsonl)

| Uncached tail | Grace | Aggregate tok/s | Batch wall, s | Slowest follower TTFT, s | Lead completes, s | Lead’s longest gap, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 386–387 | 5 s | 94.5 | 10.85 | 1.09 | 10.42 | 1.90 |
| 386–387 | 500 ms | 87.3 | 11.90 | 1.13 | 11.16 | 1.91 |
| 1,994–1,995 | 5 s | 51.0 | 20.08 | 14.10 | 5.09 | 0.07 |
| 1,994–1,995 | 500 ms | 71.7 | 14.30 | 4.73 | 12.52 | 1.47 |

For the long tail, 500 ms raised throughput 41% and reduced the slowest follower’s TTFT 66%. The short first reply took 7.43 s longer to finish. Its longest gap increased by 1.40 s. Mean time to complete a request rose from 11.19 to 11.97 s. This reproduces the 4K trade-off with a much deeper cached history.

The short-tail burst prefills took 0.563–0.590 s, compared with 1.336–1.598 s for the long tail. Maximum queue time remained 0.45–0.50 s across all arms. The candidate’s two throughput values were 76.8 and 97.8 tok/s, versus 90.6–98.6 in the four control bursts. One slow repeat accounts for most of the median difference. The queue and streaming measurements give no evidence of a grace benefit below the trigger. The three blocks all passed validation. [First control](depth01/control-a.validate.log) · [500 ms](depth01/half-second.validate.log) · [Return control](depth01/control-b.validate.log)

## Configuration decision

Retain the general-purpose incumbent on the pinned image. For an installation that gives waiting users’ TTFT and batch completion more weight than uninterrupted short replies, the 500 ms benchmark variant is the best compromise tested. For batch throughput alone, zero was fastest in the 4K trial. This experiment covered four requests, two repeats per candidate and low KV occupancy. Count how often real turns have simultaneous arrivals, a near-finished stream and a long prefill before extrapolating the gain. A remaining-service-time policy can be compared to the five-second and 500 ms controls on these same fixtures, with active-stream completion and longest gaps as explicit outcomes. [Incumbent and challengers](../../best-runtime.md)

## Runtime and reproduction

The experiment starts from the actual deployed image `tensorfold-dsv41:exl3-tp2-234f9080b326` and its embedded patches. The build reconstructs source commit `fbbc97132cd004237d9887b8d3185896efaedc5c` from a Git archive, applies the parent image's patch series and adds one opt-in environment setting, `TF_DS_FINISH_GRACE_MS`. Both Sparks received an identical native executable with SHA-256 `a1127c07ad30c1124520dfbdb05f721862d9fa251806ab5f1fcf16d1a06200b5`. The worker's lanes executable comes from the production image. A separate benchmark spill directory and cache identity protect the production cache. [Build recipe](build.Dockerfile) · [Runtime image](runtime.Dockerfile) · [Per-rank image and binary proof](image-proof.json)

The driver checks the model, commands, environment and mounts against the live recipe on both ranks. It checks for the active grace banner on the head and samples both GPUs' clocks during the workloads. The block order is 5,000 / 0 / 500 / 5,000 ms. For a rerun from the recipe root on the head, pass a fresh directory, for example `python3 -I runs/admission-grace/campaign.py runs/admission-grace/run02`. For the matched 100K follow-up add `--depth-grace-ms 500` with another fresh directory. The driver restores and validates the original serving recipe. [Driver](campaign.py)

## Final serving state

The two campaigns completed 161 workload requests, including 36 four-request bursts. After each campaign the production image, commands, environment and mount signatures matched the initial state on both ranks. On the final restart the first smoke request failed and its retry succeeded, the known production startup behavior. All eight saved replies and all eight concurrent-versus-solo checks matched. Video was skipped because PyAV was unavailable. At 01:27 UTC on November 18 the production pair was healthy, with zero active or waiting connections. [4K campaign log](run01.log) · [100K campaign log](depth01.log) · [Final validation](depth01/production.validate.log) · [Health](final-health.json)
