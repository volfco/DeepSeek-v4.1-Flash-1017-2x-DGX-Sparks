# Follow-up campaign on the two DGX Sparks

The controls use the pinned serving image and the eight-switch warm-serving incumbent. The KV-pressure replication starts on the deployed recipe so it can reproduce the previous placement result. The scheduler build adds an experiment-only remaining-time estimate. All load-changing campaigns restore the deployed recipe.

The largest measured engine gain came from lowering held-prefix patience to 500 ms under 300K KV pressure. Retain the eight-switch warm-serving incumbent, current uncapped T13 pruning, L2 prefetch and fetch-mode KV decode for general serving. The 500 ms KV policy and adaptive finish grace were tested on separate benchmark builds. Both ranks were restored to their original production image, settings and spill mounts. All eight saved replies matched after the final restore at 06:18 UTC. The 06:30 UTC health check saw no waiting or connected clients, stalls or pair failures. Video validation was skipped because PyAV was unavailable. [Configuration record](../../best-runtime.md) · [Final identities](owner-timing02.log) · [Validation](owner-timing02/production.validate.log) · [Health](final-health.json)

## 300K placement replication

The reference burst delivered 104.42 tok/s. All three independently reset peak states produced 21 new no-target refusals, a maximum queue between 24.75 and 24.85 s, and 87.65–87.98 tok/s. Each of the four requests resumed at the same 299,008-token cut and retained its 1,024-token output. Four of nine later visits to the independent sessions needed disk restoration, adding 154–170 ms of time to first token versus their warm references. The reclaimed sessions here had a surviving disk snapshot. [Replication and per-request traces](kv/summary.json)

A separate 3:1 split put the shared prefix on part 1 and 917,504 retained rows on part 0. With four matching requests, the maximum queue rose from 4.25 to 24.95 s, throughput fell from 107.70 to 92.37 tok/s and one of three revisits paid 143 ms extra TTFT. Including both placements, five of twelve revisits restored from disk and all twelve retained the full cache cut. [Partition evidence](kv/RESULTS.md)

## Shorter held-prefix patience

With 598,016 retained rows on each partition, the 500 ms policy reduced maximum queue time from 25.010 / 24.876 s in the opening / return 10 s controls to **6.320 s**. Throughput rose from 87.19 / 87.55 to **103.65 tok/s**. The policy allowed all four requests to decode together for 27.86 s, up from 3.42 / 3.32 s. No-target refusals fell from 21 to 3. The last two resumes followed waits of about 1.0 s versus 10.2–10.4 s in the controls. The planner retries periodically, so a 500 ms deadline can be observed after about one second. All three runs had identical payloads, cache cuts, lengths and output token hashes. [Controlled policy traces](kv/policy01.log) · [Paired queue, restore and throughput](kv/policy-summary.json)

All three previously independent sessions retained the 299,008-token cut when revisited. The same one came from disk in each arm. Its added client TTFT was 85 / 172 ms in the control arms and 180 ms with the shorter deadline, each versus its own memory-hit reference. This prices the measured future visit when a valid disk copy exists. A missing snapshot or a different revisit order could change that cost. [Displaced-session probes](kv/policy-summary.json)

An independent confirmation reversed the arm order and held exactly 897,024 / 299,008 rows across partitions, with the shared prefix on part 1. Against the 10 s control, the 500 ms policy again reduced maximum queue (24.997 → 6.367 s) and increased throughput (89.43 → 105.35 tok/s). The common four-way decode interval grew from 2.83 to 26.98 s. No-target refusals fell from 20 to 2. Each arm evicted one state, and the same one of three later visits restored its full prefix from disk. Every burst input, cache cut, output length and output hash matched across arms. The armed challenger matched all 39 greedy references and all eight saved validation replies. This is the preferred tested policy for the measured 300K pressure case. [Reverse-order traces](kv/policy-partition01.log) · [Armed exact](kv/policy-partition01/earlier.exact-compare.log) · [Armed validation](kv/policy-partition01/earlier.validate.log)

## Traffic coverage

Two gaps between campaigns on November 18, totaling 99m45s, contained zero completions. On the older November 13 image, 652 completed requests after the startup smoke had usable timing fields. Of those, 536 had more than one second of prefill. An estimated 389 arrived while another request was decoding, including 121 in the first five seconds after a long prefill, across 108 distinct leads. The estimate excludes arrivals within 0.2 s of the interval boundaries. With margins of zero and 0.5 s, the latter count was 129 and 109. Arrival was reconstructed from completion time, decode duration and TTFT, using the prefill-complete event as a proxy for first output. Aborted requests are outside the sample. The archive lacks the leads’ requested token limits, per-part pressure and reusable-message sizes. These counts establish historical overlap. They cannot price the current admission policy or estimate tokenizer-cache savings across current traffic. [Aggregate counts](historical-traffic.csv) · [Reproduce](traffic.py) · [Recent windows](kv/RESULTS.md)

## Adaptive admission

Protecting a stream estimated to finish within one second avoided interrupting the 80-token lead at both depths. Compared with fixed 500 ms, this let it finish 2.85 s earlier at 4K and 3.03 s earlier at 100K, while costing the slowest follower 0.54 s and 0.52 s. The longest observed lead gap in these runs fell from 1.28/1.46 s to 0.05/0.07 s (means of each lead's maximum). Against the five-second setting, the adaptive rule delivered the same lead completion and cut the slowest follower's TTFT by approximately 6.1 s at either depth. For a 256-token lead after long prefill it behaved like fixed 500 ms, admitting followers sooner and stretching the lead's completion.

Each triplet is **five seconds / adaptive / 500 ms**. The five-second result averages four bursts across start and return controls. Each other result averages two identically seeded bursts. Each burst had four requests.

| Warm workload | Lead completion (s) | Slowest follower TTFT (s) | Aggregate output (tok/s) |
| --- | ---: | ---: | ---: |
| 4K, 80-token lead | 2.25 / 2.25 / 5.10 | 10.96 / 4.83 / 4.29 | 50.07 / 62.09 / 65.80 |
| 100K, ~2K uncached, 80-token lead | 2.42 / 2.42 / 5.45 | 11.39 / 5.26 / 4.74 | 48.82 / 62.20 / 64.67 |
| 4K, 256-token lead | 5.03 / 12.21 / 12.10 | 13.54 / 4.27 / 4.28 | 52.64 / 76.81 / 77.27 |
| 100K, ~2K uncached, 256-token lead | 5.30 / 12.33 / 12.32 | 14.13 / 4.70 / 4.69 | 50.42 / 72.55 / 72.75 |
| 100K, ~386 uncached, 256-token lead | 10.82 / 10.51 / 10.47 | 1.10 / 1.09 / 1.10 | 91.18 / 93.76 / 94.28 |

The adaptive branch logged four extensions, one for each tested 80-token lead. The 56 timed bursts had equal request payloads, cache cuts and generated lengths. All 168 outputs outside the opening control matched their corresponding token hashes. Each distinct setting matched the saved 39-case greedy reference and passed validation. The four extensions used a budget-based remaining-time estimate. A stream that stops early with a large unused budget need not receive the same protection. The general-serving choice still depends on how often these arrivals occur and how much first-user completion matters. [Per-burst measurements](adaptive.csv) · [Extension evidence](adaptive01/extensions.log) · [Adaptive exact](adaptive01/adaptive-exact.compare.log) · [GPU clock samples](adaptive-clocks.csv)

Both ranks ran the same native executable, SHA-256 `b6f0b73919a12dac96fc97e8f0d22881a9d4ad3f0d59ac5250e208227c00ae63`. The control used the same build with adaptive grace disabled. After the return control the runner restored the original recipe and spill mount on both ranks, and all eight saved production replies matched. [Runtime identities](adaptive01.log) · [Production validation](adaptive01/production.validate.log)

## Sampled decode at depth and serving prefetch

Keep the recipe's `PREFETCH=1` on both ranks. With prefetch off, physical rounds took longer in every matched task/seed cell than the average of the two on controls. The median paired off penalty was 5.6% shallow and 6.3% deep for one stream, and 2.5% and 2.2% for four. All 32 sampled outputs with prefetch off matched the opening control, as did all 32 in the return control. Prefetch off also matched all 39 greedy references. The head environment and worker command were verified for each arm. [Per-window data](decode.csv) · [Armed exact comparison](runtime01/prefetch-off-exact.compare.log) · [Rank settings](runtime01.log)

The shallow prompts were 2,434–2,435 tokens and the deep prompts 299,395–299,396, with 2,048 and 299,008 cached respectively. Four code tasks each used two seeds and a limit of 512 generated tokens. Four-stream figures use windows strictly inside simultaneous decode, with no waiters. Each return-control common window lasted about 13 s at width four. The table shows medians within each prefetch-on control as opening / return. Committed tokens and physical rounds come from differences within the same active interval. The accepted fraction covers all drafted tokens in the completed requests.

| Active streams | Actual depth | Physical ms/round | Committed tokens/stream/round | Aggregate active tok/s | Draft tokens accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.4K | 37.03 / 37.64 | 2.61 / 2.64 | 71.83 / 70.95 | 59.4% / 59.4% |
| 1 | 299K | 38.93 / 38.41 | 2.71 / 2.64 | 69.71 / 69.60 | 60.5% / 60.5% |
| 4 | 2.4K | 72.53 / 73.24 | 2.42 / 2.39 | 133.63 / 130.69 | 68.3% / 67.3% |
| 4 | 299K | 79.06 / 79.49 | 2.58 / 2.60 | 130.31 / 130.67 | 68.6% / 68.6% |

On prefetch-on controls, matching by task and seed gave a median increase of 6.1% per physical round at depth for one stream (range −3.4% to +14.2%). Both four-way seed sets were slower per physical round at depth (+7.4%, +10.1%). More committed tokens per deep four-way round largely offset the additional time. The earlier production trace had different prompts and cache states. This experiment separates depth from acceptance for the four sampled coding tasks. [Workload](../review-round/sample.py) · [Clock samples](runtime-clocks.csv)

## Eager phase attribution

On the same four-stream fixtures, the largest increase was in the phase that compresses and indexes attention rows (`a_idx`), **1.21 → 5.08 ms per round**. The MoE phase rose 41.94 → 45.18 ms as rows verified per round rose 12.19 → 13.28. KV row exchange (`sh_rows`) rose 1.30 → 2.50 ms. Attention gather fell 3.56 → 1.51 ms. Summed measured GPU phases rose 66.60 → 73.13 ms. At width one the index phase rose 0.97 → 2.48 ms and the MoE phase 16.52 → 17.31 ms, at 3.65 → 3.85 rows per round. Each figure averages one value per task/seed window. The sampled row mix changes with depth. A fixed-row comparison around the context bucket boundaries would separate that effect from depth.

Both ranks had graphs disabled, the head had `TF_DS_ROUND_PROFILE=2`, and the synchronized eager run used the same prompts, seeds, and limits as the graph-on trial. The server emitted 1,136 cumulative snapshots. The analysis differences their phase totals within common active intervals. The raw GPU-clock round counter on this image had a constant `0xaaaaaaaaaaaaaaaa` origin offset. The analyzer uses the zero-based completed-round count for the denominator and requires the clock increments to agree at both endpoints. This retained 1,128 frames and excluded eight mid-update scrapes. The monotonic-to-wall offset drifted 1.2 µs. Use the graph-on table for serving speed. [Per-window phase times](profile.csv) · [Configuration](profile01/profile.json) · [Snapshots](profile01/frames.jsonl) · [Validation after restore](profile01/production.validate.log)

## Fetch versus owner decode

Keep fetch mode on the pinned image. The owner mode passed all 39 exact cases, with 583 owner rounds observed and zero fetch rounds during that gate. Its first 300K timing sweep followed the exact suite in the same process. Residual KV states produced 11 and 10 placement refusals and only 1.95 and 1.43 s of four-way overlap. Those samples failed the two-second interior-window requirement and are excluded from the clean comparison. The runner now separates the correctness and timing processes and uses separate private spill directories. [Armed exact](owner01/owner-exact.compare.log) · [Original trace](owner01/owner-decode.jsonl) · [Runner](driver.py)

The fresh owner and fetch runs began every deep burst at 299,008 / 0 retained rows, had zero new placement refusals, and matched all 32 sampled outputs across modes. Each deep burst had over 12.7 s of fully active measurement. The table reports medians of eight single-stream or two four-stream windows per cell. The small differences provide no basis for changing the general recipe. [Matched timing windows](owner.csv) · [Clean owner run](owner-timing02/owner-decode.jsonl) · [Fresh fetch control](owner01/fetch-return-decode.jsonl)

| Active streams and depth | Fetch ms/round | Owner ms/round | Fetch active tok/s | Owner active tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 at 2.4K | 37.24 | 37.50 | 70.13 | 71.50 |
| 4 at 2.4K | 71.30 | 71.55 | 134.40 | 135.12 |
| 1 at 299K | 38.39 | 39.06 | 68.81 | 69.67 |
| 4 at 299K | 79.68 | 80.34 | 130.48 | 129.14 |

## Concurrent pruning tasks

All four arms passed all 96 sampled responses, 16 SQL and 16 tool answers per width with the same eight questions per family and two seeds. Every width-four and width-eight batch contained overlapping decode for all of its requests. The task suite reached its ceiling. The cap still made deterministic output depend on accompanying requests. All eight validation prompts changed between solo and concurrent service with cap 12,1. With prune off, those eight comparisons matched within the arm, and two of eight matched the production references. The text, thinking, tool, image and long-prompt functional checks passed in both arms. [Task measurements](task-results.csv) · [Cap validation](runtime01/cap12.validate.log) · [Prune-off validation](runtime01/prune-off.validate.log)

| Requested width | Current prune, opening / return (tok/s) | Prune off (tok/s) | Cap 12,1 (tok/s) |
| ---: | ---: | ---: | ---: |
| 1 | 60.67 / 59.94 | 51.93 | 59.07 |
| 4 | 74.12 / 73.92 | 71.17 | 86.07 |
| 8 | 74.72 / 74.74 | 72.19 | 88.25 |

These rates count generated tokens over the sum of batch wall times. Sampled trajectories can change under a lossy rule. At width eight all four capped batches also yielded usable all-active measurement intervals, totaling 40 physical rounds, versus 46/47 rounds in the controls. The cap averaged 53.0 ms/round there, versus 85.1/82.2 ms, at 1.95 versus 1.96/1.91 committed tokens per stream per round. The pinned served metrics expose lane and round counts. They do not export the per-layer count of selected experts for these requests. [Per-width results](task-results.csv)

The teacher-forced set contained 3,362 target tokens across two excerpts per source, ten prose files and ten code files. Each excerpt contributes one mean token NLL and each source contributes the average of its two paired differences. The 95% interval uses the sample standard deviation across 20 source means and a Student t multiplier with 19 degrees of freedom. Current pruning increased NLL by 0.00643 versus prune off (−0.00463 to +0.01749). Cap 12,1 increased NLL by 0.02665 versus current (+0.00593 to +0.04736), exceeding the predeclared upper limit of 0.020. Retain the current pruning rule for general serving. For workloads sensitive to these differences, test more demanding tasks with simultaneous real requests. Teacher forcing runs one request at a time and cannot measure cross-request routing effects. [Frozen excerpts](reference-text.jsonl) · [Tokenized set](runtime-score01/items.jsonl) · [Scoring method](analyze.py)

The first score invocation used the scoring CLI's default port 8888 and returned 404 on all items. Those records were excluded. The recovery targeted the live service at port 8000 and reused the byte-identical frozen token-ID file. SQL/tool timing and behavior were measured separately against the correct service in the original arms. [Original run](runtime01.log) · [Corrected current score](runtime-score01/current-a.score.json)

The return control reproduced each of the 40 original current-arm NLLs. Mean per-item NLL was 1.943510 with prune off, 1.949938 in both current controls, and 1.976584 with the cap. [Paired scores and intervals](quality.csv)

## 500K immutable chat history

Keep the 32 MiB tokenizer cache for long-lived chat serving. In the fresh-start on/off/on trial, the median paired client TTFT saving was **445, 450 and 508 ms** for 100, 500 and 2,000-token appends, respectively. The corresponding median server encoding savings were **436, 436 and 437 ms**. All nine matched triplets used byte-identical inputs and the same 499,712-token KV cut, generated identical tokens and incurred zero warm restore time. The three priming requests all restored the same cut in 0.257–0.259 s. No priming time enters the comparison. [Matched trials](cache.csv)

Each size had three paired measurements. The TTFT savings ranged from 437–562, 436–505 and 477–789 ms respectively. The single approximately 0.79 s saving also includes prefill variability. In the original mixed-workload sequence, cached arms followed a deep decode sweep while the cache-off arm ran right after startup. That sequence gave larger TTFT estimates of 571, 590 and 784 ms, despite almost the same 438 ms encoding saving. Prefer the repeated comparison with equal process history. [Original sequence](cache-first.csv) · [Cache-off repeated workload](runtime-score01/cache-off-cache.jsonl)

## Comparisons fixed before timing

- **Admission.** After a long prefill, the adaptive challenger initially gives a near-finished stream 500 ms. It extends that opportunity within the existing five-second ceiling when the remaining token budget, divided by the stream's observed wall-time generation rate, is at most one second. Compare against fixed 500 ms and five seconds on identical inputs, including a new 80-token lead. Record the active user's completion and longest streaming gap as well as the slowest follower's time to first text. [Patch](adaptive.patch)
- **Deep decode and prefetch.** Compare the same four coding tasks, two sets of seeds and identical 512-token budgets at shallow and 300K cached histories. For one and four active streams, difference physical fetch/owner round counts and emitted tokens strictly within the common decode interval. Timestamp samples come from a single metrics snapshot. Use prefetch on/off/on on the pinned image and verify the head environment and worker command. [Workload](../review-round/sample.py)
- **Pruning.** Score 40 frozen repository continuations selected independently of the serving model, drawn from 20 sources, alongside SQL generation and tool selection at 1/4/8 requested concurrency. SQL answers run on four undisclosed database fixtures per question, including an empty table. Tool calls must choose the specified operation and supply the exact typed arguments. Evaluate current pruning, prune off and cap 12,1 with separate spill directories. A conservative screen for *tightening* pruning is an upper 95% paired NLL increase below 0.02 nats/token versus the incumbent and no observed decrease in either task family. These are screening criteria chosen for this trial. The task count cannot certify rare quality failures. [References](references.py) · [Task evaluator](tasks.py)
- **500K cache.** Isolate the 32 MiB tokenizer cache on/off/on at matched 100/500/2,000-token appends. Require equal serialized requests, actual cache cuts, generated lengths and token hashes for paired latency attribution. Exclude priming. [Driver](driver.py)

