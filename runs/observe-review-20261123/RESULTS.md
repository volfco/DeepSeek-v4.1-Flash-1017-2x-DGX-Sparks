# OpenObserve review: benchmarks against live data, and the dashboards (2026-11-23)

Read from `https://openobserve1.lab.volf.co` (organization `default`) from 06:37 UTC on; the engine data analysed ends at
07:10 UTC. Nothing was run on the Sparks. The only writes were to two dashboards, after exporting them. Times are UTC.

## Findings

1. **Current production (image `2bb052f`, since 05:16) has 1.9 hours of data and 399 requests that no run log accounts
   for.** That is too thin to confirm the one-stream benchmarks: five requests ran alone. Where the comparison is like
   for like it agrees: a cold 30,206-token prefill ran at 2,104 tok/s against 2,105–2,118 benchmarked at 32K, and the
   one short-prompt decode round was 3.9% above the benchmark's line. A cold 15,434-token prefill ran at 2,051 tok/s,
   3–5% under the README's 16K figure.
2. **The four deploy epochs before it (Nov 22 06:01 to Nov 23 05:16) carried no client traffic at all.** Every request in
   them belongs to a suite, a deploy check, a validate, the cache replay or a short harness run. The last large body of
   live traffic is from the fused image of Nov 21–22 (2,738 requests) and earlier builds.
3. **Live decode is slower than the headline tok/s; the most likely cause is the workload, not a regression.** One
   stream alone on the Nov 21–22 image ran at a median 78 tok/s (n = 280) against 105–135 benchmarked on code-like
   prompts. Tokens a round account for most of it (3.1 live against 4.2–6.1); the round itself is 8–12% longer than the
   benchmark's at the same drafted rows, and that excess grows with context depth. Benchmarks use short prompts; live
   prompts average 155,000 tokens. The four current-production requests at depth sit within 5 points of the earlier
   build's excess at the same depth (three below it, one 0.4 above). Content cannot be read from the metrics, so this is not proven.
4. **Production serves 8 streams at once, and at that width live agrees with the one benchmark there is.** The
   overnight suites and the sampled probe stop at 4 streams, but the Nov 21–22 suites time an 8-stream cell: 136–138.5
   tok/s in total and about 87 ms a round. Live, over 45 saturated minutes today: 91.7 ms a round and 164 tok/s by wall
   clock (about 21 tok/s a stream), on a burst whose drafts were accepted 92% of the time. **What no benchmark covers is
   the queue behind 8 streams:** up to 35 requests waited, mean time to first token was 422 s, and 113 of 153 requests ended when the client
   disconnected (21 of the 22 that can be timed one by one ended 600 s after arrival).
5. **A long prompt pass slows every stream that is decoding.** While cold passes of 166,000–192,000 tokens ran, decode
   rounds of the other streams stretched from about 92 ms to 155–312 ms, and the passes themselves ran at 1,670–1,690
   tok/s instead of 2,040–2,100. No benchmark in the current set mixes the two.
6. **Resumed turns prefill at about 1,620 tok/s, not 2,100.** Live multi-turn traffic matches the cache replay's figure
   (about 1.1 s to first token a turn), which is 22% below the cold-prefill headline.
7. **Two things are missing from the data.** `tensorfold:requests_total{key,status}` is scraped but never stored, so
   nothing in OpenObserve tells one client from another. The engine's own log does not reach OpenObserve, so there are
   no per-request `done` lines and no round profile.

## 1. What data exists

| Data | Streams | From | Labels that matter | Interval | First sample | Notes |
|---|---:|---|---|---|---|---|
| Engine, `tensorfold:*` | 200 with samples (+10 empty histogram parents) | head `gx10-eb0e`, collector scraping `localhost:8000` | `le` (10 histograms), `rank` (48 families), `part` (6), `pool`, `stream` | 30 s | 168 on Nov 10 04:55; 13 on Oct 31; the rest Nov 11–17 as switches were added | 33,456 scrapes held of 37,709 possible since Nov 10; the gaps are the engine down. 811 engine lifetimes (restarts) in that time |
| Collector's own (`up`, `scrape_*`) | 5 per job | both Sparks, jobs `tensorfold`, `dcgm-gpu-metrics`, `doca-telemetry` | `service_name`, `host_name` | 30 s | Oct 31 | the worker's `tensorfold` target is 0 in every sample since Nov 11 |
| Host, `system_*` | 28 | both Sparks (also `lab-head`, `ai-node`) | `state`, `device`, `direction`, `mountpoint`, `cpu` | 30 s | Oct 31 16:18 | hostmetrics receiver |
| GPU, `dcgm_fi_dev_*` | 12 | both Sparks, one GPU each | `gpu`, `uuid`, `modelname` | 30 s | Oct 31 17:28 | no frame-buffer or memory-clock fields on this hardware |
| ConnectX, DOCA telemetry | 2,643 | both Sparks, 5,274 samples a scrape each | `device_name`, `hca`, `port` | 30 s | Oct 31 17:32 | 321 device-level counters; 2,322 are per queue or index |
| Logs `default` | 1 | both Sparks: syslog, auth.log, kern.log | `log_file_name` | - | Oct 31 | head 819,675 rows, worker 864,075 in 23 days |
| Logs `journald` | 1 | both Sparks | `body_syslog_identifier`, `body__systemd_unit` | - | Oct 31 | no engine container output |
| Retired names | 729 | `tensorfold_*` (13, to Nov 10), `tensorfold_health:*` (77, Oct 31 only), `host` = `spark0` / `spark1` (639, Oct 31 only) | | | | not on any panel |
| Traces | 0 | | | | | |

The worker is `gx10-235f`. The scrape interval is 30 s, not the 60 s the notes say.

**The `done` lines do not reach OpenObserve.** Both log streams were searched for `tok_per_s`, `prefill_s`, `decode_s`,
`ttft_s` and `accepted` in every text field: 3 to 9 matches each in 23 days, all command lines of ssh sessions. The
substitute used here: the engine updates its finished-request counters together when a request ends and reads them under
one lock at scrape time (`server/metrics.zig`). So the difference between two scrapes is the exact total over the
requests that finished in between, and where one request finished it is that request's own record. The 4,335
unattributed requests since Nov 18 22:21 fall in 1,661 such records, 529 of them single requests.

**Three lists.** The source at `2bb052f` can emit 182 families (202 series). OpenObserve holds 200 of them. The
*Tensorfold* dashboard showed 149 before this review and shows 200 now. Per-series detail is in `COVERAGE.md`.

## 2. Benchmark windows and deploy epochs

- **Windows** come from the UTC stamps inside the logs of 92 run directories (`scripts/windows.py`,
  `data/windows.json`). Deploy checks are taken as the 6 minutes before each `finished.txt`.
- The **validate** that follows every restore is recognized by its signature at the head of an engine lifetime (23
  requests, 22,9xx prompt tokens).
- The **cache replay** on production has no stamps; it is placed at 00:58–01:04 on Nov 23 from file modification times
  and matches the 74 requests of that lifetime exactly.
- Requests in an engine lifetime shorter than 15 minutes are set aside as harness work (1,693 since Nov 18), since a
  client does not restart the engine.
- **Everything else is "unattributed, presumed client traffic".** Nothing in OpenObserve names a client, so that is
  inferred, not measured.

| Epoch | From (UTC) | Hours | Engine up, h | Lifetimes | Benchmark, check and harness requests | Unattributed requests | Prompt tokens (cached) | Generated | Most running / waiting |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| E0 older images | Nov 10 04:55 | 209.4 | 183.3 | 539 | 35,543 | 15,856 | 2,598 M (97.7%) | 11.87 M | 8 / 9 |
| E1 `ea522d35` + `DRAFT_BLOCK=6` | Nov 18 22:21 | 57.3 | 54.7 | 83 | 4,123 | 1,198 | 187 M (96.7%) | 0.99 M | 8 / 7 |
| E2 fused image | Nov 21 07:38 | 22.4 | 21.5 | 16 | 1,662 | 2,738 | 425 M (97.8%) | 3.00 M | 8 / 5 |
| E3 fused-next to 0264 | Nov 22 06:01 | 7.8 | 5.7 | 64 | 3,522 | **0** | | | 6 / 6 |
| E4 `DRAFT_BLOCK_KEPT=2` | Nov 22 13:49 | 7.7 | 5.7 | 59 | 4,484 | **0** | | | 8 / 7 |
| E5 image `10c9866` | Nov 22 21:29 | 4.9 | 4.0 | 25 | 1,060 | **0** | | | 5 / 3 |
| E6 `NCCL_MAX_CTAS=4` | Nov 23 02:23 | 2.9 | 2.0 | 27 | 1,132 | **0** | | | 6 / 5 |
| E7 image `2bb052f` (current) | Nov 23 05:16 | 1.9 | 1.9 | 1 | 24 | 399 | 30.8 M (94.8%) | 0.70 M | 8 / 35 |

E7's 399 requests fall in three stretches:

| Stretch | Requests | Mean prompt (cached) | Generated | Tokens a round | Accepted / drafted | Streams |
|---|---:|---|---:|---:|---:|---|
| 05:26–05:31 | 7 | 7,002 (0%) | 8,372 | 3.72 | 75.6% | 1 |
| 05:31–06:22, the burst | 178 | 27,945 (83%) | 464,771 | 2.01 | 92.6% | up to 8, up to 35 waiting; 115 client disconnects |
| 06:22–07:10 | 214 | 120,667 (97%) | 227,037 | 2.96 | 75.5% | up to 5 |

The burst is dozens of requests with the same 3,718-token prompt sent together, each generating until 600 s. No file in
the repository records it. The last stretch looks like the long-context traffic of E1 and E2.

## 3. Benchmarked figures against live

Live figures are for unattributed requests only. "Alone" means a single request with no other stream in any 30 s gauge
sample during it (concurrency is inferred from those samples). Spread is p10–p90. E7 is current production; E2 is the
largest recent sample but an earlier build, so read its percentages as workload evidence, not as a test of today's
image.

**Definitions.** *ms a round* = decode seconds / (rounds − requests); *tokens a round* = generated / rounds; *prefill
tok/s* = (prompt − cached) tokens / prompt-pass seconds. The benchmark line for one stream is fitted to the 32 sampled
one-stream rows of `k269` ctl1 and ctl2: **ms a round = 24.73 + 3.524 × drafted rows a round**, residual 0.72 ms (1.9%).
Given their total drafted rows, the same line lands within 2% of the benchmark's own bursts (2 streams: 57.6–57.8 ms a
round by wall against 58.8 predicted; 4 streams: 67.1–67.3 against 66.8; `data/compare.txt`). Cells with five or fewer
live requests list the observations themselves, not percentiles.

| # | Benchmarked figure (source) | Live counterpart | Live n | Verdict |
|---|---|---|---|---|
| 1 | One-stream decode round, short prompts: 42.6 ms at 5.5 drafted rows, 30.2 ms at 1.5 (`k265` / `k269` probe) | E7 alone, prompt under 8K: 38.8 ms, **+3.9%** against the line at the same drafted rows | 1 | **Matches** within the line's spread; one request |
| 2 | Same line, at depth (the probe has no deep-context rows) | E7 alone: +4.0% and +5.1% (8–64K), +9.9% (64–200K), +8.0% (≥ 200K). E2 alone: **+7.5%** (8–64K; +7.0 to +8.4), **+9.5%** (64–200K; +8.4 to +11.3), **+12.3%** (≥ 200K; +10.4 to +14.1). E1: +8.9 / +9.4 / +13.6% | E7: 4. E2: 11 / 155 / 114. E1: 8 / 19 / 43 | **Deviates**, +8 to +12% at depth. See D1 |
| 3 | Tokens a round, one stream: 4.2–6.1 code-like, 1.8 prose | E7 alone: 2.89, 3.50, 3.73, 4.60, 4.64. E2 alone median 3.1 (2.8–4.2) | 5; 280 | **Cannot be compared** on content; live sits between the two classes |
| 4 | Accepted / drafted: 73–87% code-like, 51–54% prose | E7 alone: 60, 73, 76, 78, 84%. E2 alone 64% (61–76%) | 5; 280 | **Cannot be compared** on content; inside the benchmarked band |
| 5 | One-stream decode tok/s: 105–135 sampled code-like, 59–60 prose (README, `k265` ctl1) | E7 alone: 74.6, 91.5, 96.2, 102.9, 110.1. E2 alone median 77.8 (72–96) | 5; 280 | **Deviates** from the code-like rows (−40%); inside the prose-to-code band. See D2 |
| 6 | 2 streams: 168.4 tok/s total, 57.7 ms a round at 4.8 drafted rows a stream | E2: 58.5 ms a round (53.5–64.3) at 2.9 drafted rows a stream; about 101 tok/s total | 19 records, 28 requests | **Deviates**: +30% a round at equal drafted rows. See D3 |
| 7 | 4 streams. Sampled Tetris burst: 219.8 tok/s total, 66.2 ms a round, 3.7 tokens a stream a round (`k265` / `k269`). Suite decode cell `dec4`: 116.0–119.3 tok/s total (`suite-fused`, `suite-fused-next2`) | E7 at 4 streams: 84.2 ms a round (5 records: 78.9 to 86.1), 2.56 tokens a stream a round, about 123 tok/s total. At 5 streams: 102.7 ms, about 124 tok/s | 5 records, 8 requests; 11 records, 29 requests | **Matches** the `dec4` cell's total (+3 to +6%). **Deviates** from the Tetris burst: −44% total, +27% a round. See D3 |
| 8 | 8 streams. Suite decode cell `dec8`: 136.0–138.5 tok/s total; "a standard dec8 round took about 87 ms" (`suite-fused`, `suite-fused-next2`). The overnight suites and the sampled probe stop at 4 | E7: 91.7 ms a round (89.7–94.4), 1.92 tokens and 0.99 drafted rows a stream a round, 92% accepted. **164.2 tok/s by wall clock** over 45 minutes | 40 records, 95 requests; 153 requests by wall | **Matches** on the round (+5%). Total is 19% above the cell, on content that accepted 92% of drafts. See D4 |
| 9 | Cold prefill. README (prompt tokens / prompt-pass seconds, early November): 2,155 at 16K, 2,154 at 32K, 2,077 at 128K. `k278` (prompt tokens / TTFT, current image): 2,105–2,118 at 32K, 2,040–2,059 at 96K | E7 alone, nothing cached: 30,206 tokens at **2,104** tok/s by prompt pass, 2,102 by TTFT; 15,434 tokens at **2,051** and 2,049. E1 alone, cold: median 2,028 (1,973–2,338) on 50K–256K prompts | 2; 11 | **Matches** `k278` at 32K (−0.1 to −0.8% by TTFT). The 15K request is 2.8% (by TTFT) to 4.8% under the README's 16K figure; `k278` has no cell there. One request each |
| 10 | Cold prefill with other streams decoding: none | E7 with 8 streams live: 1,668 / 1,678 / 1,686 tok/s on 166K–192K passes (`data/burst.txt`) | 3 | **No benchmark.** 19% below the pass alone. See D5 |
| 11 | A resumed 8K turn: about 1.1 s to first token; 1,796 uncached tokens at about 1,680 tok/s (`cache-20261122`) | E2 alone: TTFT 1.28 s (0.88–1.48) with 256–2,047 uncached tokens; 2.29 s (1.75–3.56) with 2,048–8,191; prefill 1,619 tok/s (1,541–1,707) on 4K–8K tails of 90K–300K contexts. E1: 1.08 s. E7 alone: two resumed turns, 0.92 and 1.34 s | 90; 194; 61; 32; 2 | **Matches** the replay |
| 12 | TTFT, short prompt, one stream: 133 ms prose, 213 ms code (README, sparkDash) | E7 alone: 0.18 s (37-token prompt). E1: 0.10 s | 1; 1 | **Cannot be compared**: one request each, and the benchmark is client-timed on an early-November build |
| 13 | Restore from the disk tier: 299,008 tokens with TTFT 1.50 s; 598,016 with 2.71 s (`k278`) | E7: one restore, 180,224 tokens, 0.4 s waited. E2: 18 restores averaging 15,360 tokens, 0.11 s a wait | 1; 18 | **Consistent**; no live restore of the benchmark's size |
| 14 | Decode at depth: 74.2 tok/s at 300K, 73.2 at 600K, greedy on a repeating text (`k278`) | E7 alone at ≥ 200K: 74.6 tok/s. E2: median 77.8 (71.3–89.2) | 1; 114 | **Cannot be compared**: greedy on repeating text against sampled live content; the agreement is a coincidence of mix |
| 15 | Pair health: exact and stable in every suite | E1, E2, E7: `pair_failed` 0, `service_refused` 0, preemptions 0, stale cache misses 0 | all requests | **Matches** |

### Deviations and their likely causes

**D1. A decode round costs 8–12% more at depth (rows 1–2).**
- *Evidence.* The excess rises with prompt depth inside one build (E2: +7.5 → +9.5 → +12.3%; E1: +8.9 → +9.4 →
  +13.6%), and the one E7 request with a short prompt shows +3.9%. Seconds from a request's last token to its end are
  0.00, so a tail after decode is not the cause.
- *Build.* E1 and E2 ran earlier images. The recorded gains since are small (`KV_FETCH_REUSE` 0.3–1.6% by wall,
  `KEPT=2` 0.3%), so at most about 2 points of the excess is the build.
- *Regression check.* E7 against E2 at the same depth, four requests: +4.0 and +5.1 against a median +7.5 (8–64K);
  +9.9 against +9.5 (64–200K); +8.0 against +12.3 (≥ 200K). Two bins are lower, one is 0.4 points higher. That is no
  sign of a regression, and too few requests to rule out a small one.
- *To confirm.* Run the sampled probe with 64K and 200K contexts and record `ms_per_round` and `drafted_per_round`.

**D2. One-stream tok/s is 40% under the code-like rows (row 5).** Tokens a round explain it: 3.1 live against 5.4 on
the benchmark's Tetris row (−43%), partly offset by the shorter round that fewer drafted rows give (39–41 ms against
42.6). Accepted share is 64% live against 73–87% on code-like prompts. That is content, and thinking or sampling
settings cannot be read from the metrics.

**D3. Against the sampled bursts, rounds with 2–4 streams are about 30% longer (rows 6–7).** Every live record at these
stream counts had a stream deeper than 16K tokens, and several overlapped prompt passes; the Tetris bursts use short
prompts with nothing else running, and accept more tokens a round (3.7 a stream against 2.6 live). Total tok/s at 4–5
streams agrees with the suites' `dec4` cell instead (123–124 against 116–119). The 30 s gauges cannot separate depth
from an overlapping prompt pass, and the E7 sample is 8 requests at 4 streams. To confirm: run the 4-stream probe on
deep contexts, then again with a cold prefill started alongside.

**D4. 8 streams (row 8).** The round matches the `dec8` cell within 5% (91.7 against about 87 ms), with a deep stream
present in 39 of the 40 live records. Each stream got about one drafted row a round (0.99) and 1.92 tokens a round.
- *4 against 8 streams.* The benchmark's own cells give 116–119 tok/s at 4 and 136–138.5 at 8, on the same prompts.
  Live gives about 123 at 4–5 streams and 164–169 at 8, but on different content (71–73% of drafts accepted at 4–5
  streams, 92% in the burst at 8). Neither live pair nor live-against-Tetris is like for like, so this record does not
  say what a lower stream cap would do.
- *What is not benchmarked.* Admission under overload: 35 waiting, 422 s mean time to first token, clients leaving at
  600 s. `continue.md` lists "more than 4 concurrent streams" as open with "no suite has gone past 4"; the `dec8` cells
  of Nov 21–22 are an exception worth noting there.

**D5. Prefill and decode interfere (row 10).** Three cold passes of 166K–192K tokens ran at 1,668–1,686 tok/s with 8
streams live, and requests finishing in the 150 s before each pass ended show 155–312 ms a round against about 92 ms in
steady state. To confirm: a suite cell that starts a cold 128K prefill during a 4-stream decode.

### Benchmarked figures with no live counterpart

- **The cycle model** (forward 17.9 ms + 3.6 ms a row, one drafter pass 4.15 ms): `forward_ms`, `pass_ms` and
  `rows_a_round` exist only in the round-profile log lines. Rows 1–2 are the nearest proxy.
- **Client-timed figures**: the sparkDash tables, streamed tok/s (129.6–129.9 at 1 stream, about 215 at 4) and
  chunk-gap percentiles. The server's counters start at arrival and count tokens at request end.
- **Greedy against sampled, thinking on or off, prose against code**: not in the metrics.
- **Per-request queue, restore, render and encode seconds**: in the `done` line only; TTFT lumps them together.
- **TTFT percentiles under 0.25 s**: the histogram's edges are 0.1, 0.25, 0.5 s. Means from `_sum` / `_count` were used.

## 4. Dashboards

Both changed in place, by `PUT` with the hash the read returned (twice each: the second write added units, the `Spark`
selector and the 4- and 8-stream reference lines). Every original tab and panel is unchanged (89 of 89 and 21 of 21
compare equal); new tabs were appended. The saved dashboards read back identical to what was sent (0 differences).

| Dashboard | Panels before | Panels after | Tabs | Queries returning data over 24 h |
|---|---:|---:|---|---|
| Tensorfold (`7522360476127199234`) | 89 | 161 | 6 → 14 | before 89 of 89 panels; after 155 of 157 (4 text panels) |
| DGX Sparks (`7522324382660689920`) | 21 | 119 | 1 → 6 | before 21 of 21; after 117 of 117 (2 text panels) |

**Tensorfold, new tabs:**
- *Benchmark vs live*: decode tok/s a stream, total tok/s by wall, ms a round, tokens a round, drafted rows a round,
  accepted / drafted, prefill tok/s on uncached tokens, TTFT and its prompt-pass part, TTFT per 1,000 uncached tokens,
  prompt depth, streams running and waiting, tokens each live stream holds. Benchmark figures are drawn as labelled
  reference lines.
- *Throughput & latency (means)*: the `_sum` / `_count` series as means, rates and quantiles.
- *Cache causes & resume depth*, *Engram row cache*, *KV shard by rank* (with a `Rank` selector), *vLLM-name copies*.
- *Ranks, scrape & restarts*: scrape target up by host, samples a scrape, restarts in the last hour, pair state.
- *Any series*: a selector listing all 200 series and the 90 retired names.

**DGX Sparks, new tabs:** every host series (28 panels), every DCGM series (13), every device-level ConnectX counter
(321 counters on 48 panels), the collector's scrape health, and a box that shows any series by name. A `Spark` selector
(both, head, worker) filters the new tabs; the original tab is as it was. Counters are per-second rates with units set
by series (bytes/s, packets/s, seconds, °C, W, MHz, percent); 20 panels of counts and states carry plain numbers.

**Two panels are empty, as expected:** *Energy (TF_DS_ENERGY)*, because `joules_total` exists only with that switch on;
and *Share of has() calls answered without the lock*, because the three `engram_rowcache_*` counters have read 0 since
Nov 17.

**Left untouched** (exported to `dashboards-before/` only): *Host Metrics* (25 of 25 panels return data), *GPU
Monitoring - NVIDIA* (18 of 24; six ask for DCGM fields this hardware does not export), *Docker Container Metrics* (0 of
7; no such streams exist).

**Restore** (the undo for each dashboard; it PUTs the exported definition back with the live hash):

```sh
cd /home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/observe-review-20261123
. scripts/o2env.sh
python3 -I scripts/dash_api.py restore dashboards-before/7522360476127199234.json   # Tensorfold
python3 -I scripts/dash_api.py restore dashboards-before/7522324382660689920.json   # DGX Sparks
```

This command was run on a scratch copy of *Tensorfold* after changing it: 0 differences from the backup and the
original hash came back. It has not been run on the two real dashboards. The scratch copy was deleted afterwards.

## 5. Metrics the engine exposes that are missing

- **`tensorfold:requests_total{key,status}`**: the collector scrapes 373 samples, OpenObserve stores 372. Where the one
  is dropped is not established.
- **Rank 1**: only the disk tier and `kv_shard_rank_*` carry a `rank` label; everything else is rank 0's view. The
  worker's collector also scrapes a `tensorfold` target that does not exist and has failed every 30 s since Nov 11.
- **`joules_total`**: only with `TF_DS_ENERGY`. **`process_footprint_bytes`**: macOS only.
- **Log-only**: the `done` line (queue, restore, render and encode seconds, thinking, finish reason) and the round
  profile. `COVERAGE.md` gives the cheapest route for each.

## 6. Not verified

- That unattributed traffic is client traffic. The E7 burst in particular could be a load test nobody logged.
- Concurrency during a request, and whether a prompt pass overlapped it: both come from 30 s gauge samples.
- The causes given for D1, D3, D4 and D5. They fit the data; none was tested by an experiment.
- That E1 and E2 differ from the benchmark's build by only about 2%: taken from `continue.md`, not measured here.
- The prompts and sampling settings of the suites' `dec4` / `dec8` cells: their totals were read from the two
  `RESULTS.md` files, the cell definition was not traced.
- Where `requests_total` is dropped, and the collector's configuration on either Spark (the nodes were off limits).
- That device memory and the Engram histogram are rank 0's only: read from the source, not checked against rank 1.
- How the new panels look in a browser. Each query was run through the API and the saved definitions read back equal,
  but no page was rendered: reference-line legends, units and the two selectors (`Rank`, `Series`) and the *Any series
  name* box were not seen working.
- The restore command on the two real dashboards (proven on a scratch copy only).
- The cache replay's window (from file modification times) and the 15-minute rule for harness lifetimes.
- Benchmark windows before Nov 18 were applied but E0 was not analysed beyond the inventory row.
- Benchmarks that left no stamped log would be counted as unattributed. `runs/fused-next-20261121` (files written Nov
  21 07:41–14:36) and `runs/pre-bench-20261121` (18:58–20:44) overlap E2's production lifetimes; their contents look
  like offline work, which was not confirmed.

## 7. Files

| Path | What |
|---|---|
| `COVERAGE.md` | every series, its panel, and what is missing |
| `dashboards-before/` | the five dashboards as exported before any change (the undo), and the dashboard list |
| `dashboards-after/<id>.sent.json`, `<id>.json` | what was sent, and what OpenObserve returned afterwards |
| `scripts/o2env.sh`, `o2.py` | credentials from the operator's helper at run time into the environment; the client |
| `scripts/inventory.py`, `source_metrics.py`, `pull.py`, `table.py` | stream inventory, the source's series list, raw samples, one row a scrape |
| `scripts/windows.py`, `analyze.py`, `compare.py`, `burst.py` | benchmark windows and epochs, interval records, the comparison, the saturated stretch |
| `scripts/build_dashboards.py`, `verify_dashboard.py`, `dash_api.py`, `coverage.py` | build, verify, write and restore the dashboards; write `COVERAGE.md` |
| `scripts/leakcheck.sh` | scans this directory for the password (taken from the environment); clean at the end |
| `data/` | listings, `epochs.txt`, `compare.txt`, `burst.txt`, panel verification before and after. `scrapes.json` (37 MB) and `records.json` (13 MB) are git-ignored; the 225 MB of raw samples were deleted and `pull.py 2026-11-10T00:00:00Z` fetches them again |

One file was written outside this directory by mistake early on, `/tmp/o2_streams_probe.json` (a stream listing), and
was deleted. Nothing was committed. The dashboard exports carry the owner's login email as the API returns it; the
password and its encoded form appear in no file here (`scripts/leakcheck.sh`).
