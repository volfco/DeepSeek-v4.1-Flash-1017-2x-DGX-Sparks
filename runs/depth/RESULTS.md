# Depth: decode per round at fixed context depth, and the warm turn (2026-11-17)

Production pair as it served: image `tensorfold-dsv41:exl3-tp2-234f9080b326`, started 17:28:16Z, `.env` settings
(KV sharding, spill tier, graphs on, cap 128). **No restart, no downtime.** All cells were client requests on the
idle-gated, clean-checked client (`tools/bench_div.py` `guarded`/`multi`). All 75 cells were clean on their first
attempt (`depth.log`: no `clean=False`, no retries).
Identity before and after is in `prod.rank*.id` and `end.rank*.id` (image, command, `TF_*`/`TENSORFOLD_*`; secrets
redacted). The two are unchanged, rechecked at 18:11Z and again at 18:23Z, both times with `/health` ok and the
pair not broken.

Drivers: `depth.py` (ladder, bookends, warm turns), `four.py` (the 100K 4-stream cell; see "Deviations"),
`lowrows.py` (low-rows short cells), `fourwin.py` (an attempted windowed 4-stream check; see "4 streams").
Memory sampler: `mem.sh` -> `mem.head`/`mem.worker`, 1 s. Clocks: `clocks.log`.
Numbers: `analyze.py` -> `tables.md`, `fit1.py` -> `fit1.md`, `warmfit.py` -> `warmfit.md`. Engine logs:
`run.rank{0,1}.log`. Settings: temperature 0 (greedy; production samples at 1.0), thinking on, 1,024 decoded
tokens per request. Every reply reached `max_tokens` while still thinking. ms/round = `decode_s`/`rounds` from each
request's `done` line, and rows/round = 1 + drafted/rounds. GPU clocks were under the operator's 2400 MHz cap
(2359-2392 MHz), with max temperature 83 C on the head and 75 C on the worker.

## Verdict

**At 1 stream, decode costs slightly more per round with depth, about 0.9 ms per 100K tokens, close to linear.**
The matched-row 125K and 140K observations are close; the 250K and 270K observations differ by ~0.35 ms. Sparse rungs with different prompts cannot resolve a threshold effect or set an upper bound on one. Against a short-context line measured the same way (n=10, 2.6-5.2 rows,
`ms = 19.64 + 4.36 x rows`, rms 0.22 ms), the 1-stream cells at the same rows are higher by:
- +0.9 to +1.5 ms (2-4%) at 100K;
- +1.1 to +1.9 ms (3-5%) at 125-200K;
- +2.0 to +2.1 ms (5-6%) at 250K;
- +2.4 to +2.7 ms (7-7.5%) at 270-300K.

The short line has no cells between 3.0 and 4.3 rows, where most depth cells sit, so it is interpolated there.
Four depth cells at 4.1-4.9 rows (250K r0, 300K r0, both 100K bookends) need no interpolation and give the same
excess: +2.0, +2.6, +1.2 and +0.9 ms. A joint fit over all 26 1-stream cells gives `ms = 20.02 + 4.29 x rows + 0.89 x (prompt/100K)`, rms 0.26 ms.

The deepest greedy solo cells here took 38–40 ms/round. The historical 59–88 ms production rounds used different prompts, sampling and cache state. Isolating the earlier slowdown requires matched traffic. The solo 10% gate was unmet. Four-stream steady decode remains unresolved.

The rank-0 profile model (`11.4 + 4.93 x rows + 4`) is 3-5% low at the short reference itself (4.3-5.2 rows) and
about 6% low at 3.9 rows. The served path's fixed part is about 4 ms larger and its slope smaller. Measured against
that model, the depth cells read +10-15%; about half of that is the model's misfit, not depth.

## Depth ladder, 1 stream (measured: `depth.json`, `tables.md`, `fit1.md`)

| depth (prompt) | cached | ms/round | rows/round | model ms | short line at these rows | tok/round | tok/s | mem low H/W GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 0 | 40.5 / 38.3 | 4.67 / 4.29 | 38.4 / 36.6 | (reference) | 3.98 / 3.57 | 98.8 / 92.6 | 6.87 / 7.05 |
| ~50 (low-rows short, 4 reps) | 0 | 31.1-32.4 | 2.61-3.02 | 28.3-30.3 | (reference) | - | - | 5.73 / 6.72 |
| 101,292 | 0 / 100,352 | 37.1 / 37.6 | 3.67 / 3.82 | 33.5 / 34.2 | 35.6 / 36.3 | 2.66 / 2.83 | 71.7 / 75.1 | 6.46 / 7.38 |
| 125,132 | 100,352 / 124,928 | 38.1 / 36.4 | 3.99 / 3.46 | 35.1 / 32.5 | 37.0 / 34.7 | 2.81 / 2.63 | 73.3 / 72.3 | 6.26 / 7.08 |
| 140,114 | 124,928 / 139,264 | 37.0 / 37.4 | 3.60 / 3.64 | 33.1 / 33.3 | 35.3 / 35.5 | 2.65 / 2.68 | 71.5 / 71.7 | 6.20 / 7.03 |
| 199,878 | 139,264 / 198,656 | 38.2 / 36.6 | 3.81 / 3.46 | 34.2 / 32.4 | 36.3 / 34.7 | 2.88 / 2.40 | 75.2 / 65.5 | 6.17 / 6.94 |
| 250,213 | 198,656 / 249,856 | 41.3 / 37.1 | 4.50 / 3.51 | 37.6 / 32.7 | 39.3 / 35.0 | 3.88 / 2.50 | 94.0 / 67.5 | 6.11 / 6.89 |
| 269,906 | 249,856 / 268,288 | 37.4 / 37.8 | 3.50 / 3.62 | 32.7 / 33.2 | 34.9 / 35.4 | 2.57 / 2.63 | 68.4 / 69.6 | 5.83 / 6.86 |
| 300,575 | 268,288 / 299,008 | 40.3 / 38.3 | 4.14 / 3.65 | 35.8 / 33.4 | 37.7 / 35.6 | 3.41 / 2.80 | 84.9 / 72.9 | 5.78 / 6.88 |

The two values in a cell are the two repetitions. Rep 0 at each rung paid the increment's prefill; its decode window
is after TTFT. Reps 0 and 1 agree within 1-2 ms at matched rows. In every request, `queue_s` was 0.000-0.003 and
`restore_s` was 0.000.

- **Bookends (drift check):** after 300K, `end-d100000` read 39.2 / 41.8 ms at 4.22 / 4.87 rows, which is +1.2 /
  +0.9 ms over the short line (the first 100K reading was +1.5 / +1.3). `end-d2000` read 42.3 / 39.8 ms at 5.22 / 4.68
  rows. `final-d2000`, 3 minutes later with the same prompts, reproduced the same tokens at 42.3 / 40.2 ms. No drift
  over the run.
- **Tokens per second** fall from 93-113 at 2K to 65-85 at depth. That is mostly acceptance (tokens/round 2.4-2.9 at
  depth against 3.6-4.8 at 2K), not round cost. Whether acceptance itself falls with depth is unresolved here: the
  100K bookends drafted 3.4-4.4 tokens/round on the same text with a different task line.
- **Onset:** there is no rung between 2K and 100K, so where the first ~1 ms appears is not placed.

## 4 streams (measured; n=1 per depth, inferred prediction)

| depth | ms/round med (range) | summed rows | `11.4 + 4.93 x summed rows` (inferred) | aggregate tok/s | mem low H/W |
| --- | --- | --- | --- | --- | --- |
| 2,000 | 77.0 (65.9-88.9) | 14.42 | 82.5 | 131.8 | 6.65 / 6.86 |
| 101,292 | 70.6 (69.3-70.9) | 12.07 | 70.9 | 122.5 | 5.73 / 6.76 |
| 125,132 | 73.0 (70.9-77.2) | 13.05 | 75.7 | 133.0 | 6.15 / 7.05 |
| 140,114 | 72.7 (70.9-78.7) | 12.87 | 74.8 | 123.2 | 6.10 / 6.98 |
| 199,878 | 75.5 (66.2-82.0) | 12.99 | 75.4 | 122.0 | 6.03 / 6.92 |
| 250,213 | 73.7 (72.4-75.9) | 13.40 | 77.5 | 131.0 | 5.90 / 6.94 |
| 269,906 | 77.7 (68.4-83.0) | 12.54 | 73.2 | 118.0 | 5.63 / 6.87 |
| 300,575 | 77.8 (69.6-88.9) | 13.26 | 76.8 | 117.8 | 5.60 / 6.89 |
| bookend 101,293 | 73.3 (70.7-79.1) | 12.38 | 72.4 | 123.5 | 5.75 / 6.84 |
| bookend 2,001 | 70.6 (68.6-76.3) | 14.03 | 80.6 | 130.3 | 5.82 / 6.84 |

Each stream's `decode_s` includes rounds shared with the other streams' staggered prefills (`queue_s` up to 4.4 s).
- Adjusting these whole-request measurements by the row model suggests a possible 5–15% difference. Staggered prefill, partial overlap and a changing number of active requests prevent a steady four-stream estimate.
- The window check saw cumulative round counters update on completion. For all three later 300K bursts the last stream's decode started after the first had finished. No common four-stream interval exists there. Use streamed first/last-token intervals to establish overlap and round events for phase timing.
- Seven of the twelve 300K requests queued 11.715–27.350 s, with zero restore time. The other five queued at most 1.5 s. Head memory stayed above 5.60 GiB. The earlier 17:58Z burst queued at most 4.4 s. The ladder left about 1.04M kept rows in a 2,101,248-row pool split into two equal parts. Four independent 300K streams require about 1.2M live rows.
- `refused_no_target` rose from 1 to 55 and `twins_made` from 47 to 60 over the later bursts. Three stale-placement messages fell inside rep 1 and none inside reps 2–3. Idle `twin_rows=0` cannot establish sharing during decode. Compare per-part occupancy, near-finish admission delays and shared/independent prefixes with time-aligned request intervals. These waits establish an admission problem for this workload; the fraction due to each mechanism remains open. The earlier 17:58Z 300K ladder burst had ~23.7 s of common decode; the later zero-overlap bursts occurred at the same nominal depth and reply budget. [Burst data](fourwin.json)

A later controlled replay held the same four 300K inputs, outputs and 299,008-token cut fixed. Increasing kept rows from 319,488 to 1,216,512 raised the longest queue from 4.31–4.45 s to 24.87 s, with 21 new no-target refusals. This establishes a placement-pressure effect for the reproduced workload. The precise retry/reclamation delay still needs attribution. [Controlled replay](../review-round/RESULTS.md#controlled-kv-occupancy)

## Warm turn (measured: `tables.md` "warm turns", `warmfit.md`)

The append branches from the same cached history each time. `cached` is always a 2,048 multiple, so uncached =
history slack past the grid + the append + the question.

| history | append | uncached | prefill_s | ttft_s | tok/s on uncached |
| --- | --- | --- | --- | --- | --- |
| 8K | 100 | 2,132 | 1.511 | 1.521 | 1,411 |
| 8K | 500 | 2,546-2,559 | 1.673-1.688 | 1.683-1.698 | 1,508-1,530 |
| 8K | 2,000 | 4,002-4,042 | 2.330-2.347 | 2.341-2.358 | 1,718-1,722 |
| 100K | 100 | 1,053-1,103 | 1.003-1.021 | 1.097-1.119 | 1,050-1,080 |
| 100K | 500 | 1,483-1,484 | 1.194-1.198 | 1.290-1.293 | 1,239-1,242 |
| 100K | 2,000 | 2,993-3,080 | 2.103-2.129 | 2.199-2.226 | 1,423-1,447 |
| 200K | 100 | 1,313-1,319 | 1.222-1.227 | 1.408-1.418 | 1,070-1,079 |
| 200K | 500 | 1,699-1,756 | 1.425-1.460 | 1.612-1.647 | 1,192-1,203 |
| 200K | 2,000 | 3,255-3,353 | 2.432-2.497 | 2.617-2.682 | 1,338-1,343 |

`queue_s` was 0.000-0.002 and `restore_s` 0.000 in every row. The 8K 100-append rep 0 is left out: its history prime
was not kept, so it ran cold (8,289 tokens, 4.596 s, 1,804 tok/s).

Fits `prefill_s = a + b x uncached`, using the warm rows plus the history prime as a zero-append point:

| history | a (fixed, s) | b (ms/token) | marginal tok/s | `ttft_s - prefill_s` | a 2,200-token re-prefill: prefill / ttft |
| --- | --- | --- | --- | --- | --- |
| 8K | 0.558 | 0.442 | 2,261 | 0.010 s | 1.53 / 1.54 s |
| 100K | 0.373 | 0.572 | 1,747 | 0.096 s | 1.63 / 1.73 s |
| 200K | 0.380 | 0.629 | 1,589 | 0.186 s | 1.76 / 1.95 s |

- **Slack against overhead.** A 100-token append onto 100K costs 1.0 s of prefill. Of that, about 0.37 s is the
  fixed part, about 0.55 s is the ~950 tokens of grid slack and question re-run, and well under 0.1 s is the append
  itself. A small-append turn is mostly slack plus fixed overhead.
- **Cost against depth.** The cost of a warm turn grows with depth in two ways:
  - The per-token cost of the re-prefilled tokens rises, about 0.44 -> 0.57 -> 0.63 ms at 8K -> 100K -> 200K, with
    each new token attending to a longer history.
  - The TTFT residual includes work before prefill. Across six warm appends per depth, median encode_s was 0.0935/0.182 s at 100K/200K, against TTFT minus prefill of 0.096/0.1865 s. Most of this residual is input encoding.

  The 8K intercept is not comparable (uncached never went below 2.1K there). The fixed part at 100K and 200K is the
  same, about 0.375 s, but that is an extrapolation, since uncached never went below about 890.
- The fit predicts 1.63-1.76 s of prefill (1.73-1.95 s TTFT) for production's typical 2.2K re-prefill at
  100-200K. That matches the 1.7-1.8 s in `runs/overnight/break-1600Z.rank0.log`.
- The cold rate is 1,948 tok/s at 100K (`prime-100000`). Incremental prime rates fall with depth: 1,777 at 125K,
  1,750 at 200K, 1,573 at 300K.

## Exact resume (0121): quality difference (from existing artifacts, not rerun)

Source: `runs/matrix/step0-cold-shape/step0-{A..F}.json`. These are teacher-forced `tools/score.py` runs of the
control and five `TF_DS_DEBUG_SPLIT` off-grid cuts. decisions.md, "0121 exact resume: LOSSY", says the split
reproduces the exact-resume reply sha. Per item, mean NLL per token against control A:

| arm | t3k (2,840 prompt) | t10k (9,481) | t70k (66,412) |
| --- | --- | --- | --- |
| B split 8228 | 0 | -0.148 | +0.104 |
| C split 9480 | 0 | -0.031 | +0.008 |
| D split 9440 | 0 | -0.059 | +0.029 |
| E split 9380 | 0 | -0.070 | +0.055 |
| F split 9300 | 0 | -0.135 | -0.023 |

- Shifts are -0.15 to +0.10 nats/token, of both signs. The mean over the ten affected item-arm pairs is about
  -0.03. The largest moves are not small in perplexity terms: -0.148 on t10k is a 14% perplexity change, and +0.104
  on t70k is +11%. The argmax match moved by at most 1 token of 17. t3k is identical in every arm because its prompt
  is shorter than every split.
- Behaviourally, decisions.md records that turns 6-8 of 8 replied differently from cold, deterministically over 3/3
  passes.
- **Limits.** Only 17 scored tokens per item, and only per-item means are stored, so the paired SE of the
  difference cannot be computed. Each shift is well inside the per-item `nll_se` of about 1.7, which is the SE of a
  17-token mean. These are `DEBUG_SPLIT` arms, not `TF_DS_EXACT_RESUME=1` itself.
- **Reading:** the observed perturbations have mixed signs. With 17 tokens
  per item it cannot show whether there is a loss, and per-item shifts reach 10-14% in perplexity. Use held-out long prompts, actual cold/grid/off-grid resume positions and paired continuation scores with uncertainty. The short existing prompts need longer contexts to exercise these cuts.

## Item 1: production traffic on a healthy config

None found in the logs. Since 2026-11-15, every chat POST in the head's `runs/**/*.log` came from 127.0.0.1 or from
100.126.133.124. That address is the operator workstation's tailnet IP: 50 requests per arm, prompts of 30.7K or
less, all in `runs/series*`, so it is campaign traffic. The serving container's log (since 17:28Z) had only
127.0.0.1. The only external client, 203.0.113.142, appears in `runs/overnight/break-1600Z.rank0.log` (the row-cache
day). OpenObserve was **not searched**: there are no credentials in this session, and `tools/obs_arm_metrics.py`
reads Prometheus series, not request lines.

## Other observations

- **CUDA graph cap reached on both ranks:** "128 kept (the cap), N destroyed for newer ones". N passed 1 at 17:44:45
  (the first 4-stream cell), 16 at 17:46:56 (first 100K decode), 32 at 17:48:33 (125K), 64 at 17:49:25 (140K),
  128 at 17:56:21 (270K) and 256 at 18:00:31 (the 2K 4-stream bookend). The log prints at powers of two, so the
  final count is 256-511. New depths and 4-stream shapes push out graphs (`run.rank0.log`, `run.rank1.log`). The
  overnight campaign never reached this regime (0 destroyed, 64 kept). Second reps at a depth match first reps, so it
  is not shown to drive the per-round figures, but the `TF_DS_GRAPH_CAP` question (to-benchmark row E) now has a
  serving trace where the cap binds.
- **Memory:**
  - Head `MemAvailable`: 7.27 GiB at 17:37Z, then a low of 5.60 at 300K x 4 streams. Idle readings afterwards:
    5.83 (18:11Z), 5.73 (18:17Z, six minutes after the last request) and 6.05 GiB (18:23Z).
  - Worker: 8.0 GiB at 17:37Z, then a low of 6.74. Idle afterwards: 6.75-6.78 GiB.
  - The 4-stream extra over 1 stream was at most about 0.25 GiB at any depth.
  - **This does not fit the recorded figures.** The brief had the head losing 2.3 GiB from load to 261K on one
    conversation. Here it lost about 1.2-1.7 GiB to 300K, and the engine's own `memory.active` rose only 0.16 GB.
- **What the run left in production's cache** (`/health`, 17:37Z -> 18:23Z):
  - prompt-cache entries: 1 -> 25 (held 6.7 -> 167 MB);
  - spill stored: 25 / 10 MB -> 557 / 688 MB on rank 0 / 1, 25 points;
  - `kv_shard` kept rows: 20,480 / 0 -> 299,008 / 538,624.

  These are synthetic `[depth-1117]` histories. They age out under the normal eviction rules; nothing was
  restarted to reclaim them.

## Deviations

- **Primes not kept:** the per-rung prime (`max_tokens=1`, thinking off) was not kept by the prompt cache, so each
  rung's rep 0 paid the increment's prefill, resuming from the previous rung. The decode timings are unaffected.
- **100K 4-stream gate bug:** `depth.py` scaled the 2K rung's 4-stream memory extra by the depth ratio (50x) and
  skipped the 100K 4-stream cell. Rungs from 125K on compared against the previous rung and ran. `four.py` added the
  100K cell after the warm turns, behind a 1-stream refresh.

## Next arms (not run)

- The served round profile (`TF_DS_ROUND_PROFILE=2`, recipe `GRAPHS=0`; 0123 is in the serving binary) at 2K and
  300K for attribution. It synchronizes each round and emits cumulative counters with served step_ms=-1. Difference snapshots and use separate graphs-on wall timing.
- If the measured exchange cost warrants it, test `TF_DS_KV_SHARD_DECODE=owner`.
  - Engine source of a later checkout (`tfref-2ndpass-srv`): `featureString` covers
    spill/spill_min/kv_shard/kv_carry/twins/engram_lanes/image_cache/streams/round_rows/pool/drafts/part, plus
    edit_mark, keep_protect, exact_resume, spec_sample and spill_shared_min when on.
  - The Features struct there also carries `owner_version`. Whether `owner` changes the fingerprint on this image
    is **unconfirmed**; check the serving build's source before any run against `/spill`.
