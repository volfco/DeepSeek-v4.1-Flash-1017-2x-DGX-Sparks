# OpenObserve review — brief (2026-11-23)

**Goal (the operator's words, paraphrased).** Review the OpenObserve data collected from the Spark nodes. Using it:
1. make sure production's performance as benchmarked is reflected in the live engine data, with no significant
   deviation between the benchmarked numbers and real workloads;
2. improve the dashboard so that it includes every useful metric, presents all the data available, and leaves no
   metric out.

## Access

- The instance is the operator's own: `https://openobserve1.lab.volf.co`, organization `default`. Deployment
  reference: `~/Workspace/homelab/kubernetes2/openobserve.yaml`. A working query helper with the API shape is
  `~/Workspace/homelab/kubernetes2/.o2q.py`; the recipe's own cross-check tool is `tools/obs_arm_metrics.py`
  (Prometheus-compatible API; it takes `OBS_USER` / `OBS_PASS` from the environment).
- **Credentials:** read them from those two files at run time and pass them through environment variables. Never
  write them into any file you create, any script, any log, the results or your report. Do not edit anything under
  `~/Workspace/homelab`.
- **Reads:** anything in the instance (streams, metrics, logs, dashboards, settings).
- **Writes allowed:** dashboards only, and only after saving the current definition (see part 2). No deleting or
  altering streams, data, users, alerts, pipelines, retention or settings.
- **The Spark nodes themselves are out of bounds:** no ssh, no request to the engine's port 8000 (a `/metrics` scrape
  there makes the engine log its profile lines, and production is live). Everything comes from OpenObserve, the saved
  runs and the engine source.
- If the instance is unreachable or the credentials are refused, stop and report exactly what failed.

## What the data is

The engine's Prometheus endpoint is scraped every 60 s into OpenObserve (`decisions.md`, "Benchmark evidence tools").
Its series are named `tensorfold:*` and are defined in `~/Workspace/tfref-kstack4/zig/src/server/metrics.zig` (about
42 names appear in the source; production is commit `2bb052f`). There may also be host, GPU and log streams from the
two nodes (head `gx10-eb0e`; find the worker's name in the data). **Start with an inventory:** every stream that
carries Spark-node data, its type, its labels, its first and last sample, its scrape interval, and for the engine
series which rank or host each comes from. Compare three lists: what the engine source exposes, what OpenObserve
holds, and what the dashboard shows. Note whether the engine's per-request `done` log lines (rounds, accepted/drafted,
`prefill_s`, `ttft_s`, `decode_s`, `tok_per_s`, prompt and cached tokens) reach OpenObserve as logs: if they do, they
are the best live data there is.

## Part 1: benchmarked against live

**The benchmarked numbers** (recipe repository, `/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`):
- `README.md` lines 29–62 (decode tok/s by workload and stream count, TTFT, prefill tok/s by depth, the sampled-decode
  table); `best-runtime.md`; `continue.md` "Status summary" and the adoption table;
- the latest production probes and deploy checks: `runs/k278/RESULTS.md`, `runs/deploy-2bb052f-20261123/`,
  `runs/overnight-20261122/k269/RESULTS.md` (client-timed probe: streamed x1 129.6–129.9 tok/s, x4 about 215),
  `runs/overnight-20261122/k265/ctl1.probe.json` (per-workload rounds, tokens a round, accepted/drafted);
- the cycle model in `continue.md`, "Audit review and re-score of rounds 0259–0279": one-stream cycle = forward +
  one drafter pass (38.3 + 4.15 ms), forward ≈ 17.9 ms + 3.6 ms a row, accepted/drafted 73–87% on code-like sampled
  rows and about 54% on prose, cold prefill about 2,040–2,120 tok/s.

**Things that will mislead you unless handled:**
- **Benchmark suites ran on the same engine for most of November 22–23 UTC.** Each arm restarts the engine (counters
  reset) and drives synthetic load. Their windows are in the suites' logs (`runs/*/run.log`, `runs/overnight-20261122/*/`
  arm timestamps, `continue.md`). Separate benchmark windows from real client traffic before comparing anything, and
  say how much real traffic there is in each period.
- **Production changed several times.** Segment by deploy epoch (`continue.md` adoption table: November 22 13:49 UTC
  `KEPT=2`; 21:25 image `10c9866`; November 23 02:22 `NCCL_MAX_CTAS=4`; 05:20 image `2bb052f`), and further back by
  the deploy records under `runs/deploy-*`.
- **Benchmarks are not the live workload.** Benchmarks use fixed prompts, thinking off, set stream counts and depths;
  suite arms also run with `TF_DS_ROUND_PROFILE=1`. Live requests differ in depth, cache hits, thinking, concurrency
  and content. Compare like with like: condition on what the data lets you condition on (streams active, prompt
  depth, cached share, sampled or greedy), and prefer rates that do not depend on the mix (decode seconds a round,
  tokens a round, accepted/drafted, prefill tokens a second on uncached tokens, TTFT against uncached tokens).
- **A 60 s scrape of counters** gives interval rates, not per-request values; reset-aware differencing is required
  (`tools/obs_arm_metrics.py` shows how).

**Deliver:** a table of each benchmarked figure against its live counterpart with the live sample size and spread; a
verdict per figure (matches, deviates, cannot be compared and why); for each deviation above the benchmark's own noise
the most likely cause with evidence (workload mix, concurrency, depth, cache misses, a regression) and how to confirm
it; and which benchmarked figures have **no** live counterpart because the metric does not exist.

## Part 2: the dashboard

1. Export every existing dashboard that shows Spark or engine data to
   `runs/observe-review-20261123/dashboards-before/<id>.json` first. This is the undo.
2. Improve it. Every `tensorfold:*` series and every other Spark-node series in the instance must appear on a panel,
   grouped so a reader can find things: throughput and latency (requests, tokens, decode tok/s, TTFT, queueing),
   prefill and the prompt cache (cached share, spill tier, restores), decode internals (rounds, rows a round,
   accepted/drafted, drafter), Engram, the pair's health (link, broken/failed, restarts), and host and GPU (memory,
   temperature, clocks, bus) if those streams exist. Derived panels belong where they answer part 1 directly: live
   decode tok/s and tokens a round against the benchmarked reference lines, accepted/drafted, prefill tok/s on
   uncached tokens, TTFT against prompt depth. Counters as reset-aware rates, histograms as quantiles, units and
   legends set, variables for rank or host where a series has both.
3. Verify before and after: run each panel's query through the API and confirm it returns data over the last 24 h (or
   say that the series is empty and why). Then re-read the saved dashboard and confirm it matches what you sent.
4. Prefer updating the existing dashboard in place (the backup makes that reversible). If the dashboard API or schema
   leaves you unsure that an in-place update is safe, create a new dashboard beside it with a clear title and leave
   the original untouched. Say which you did, and give the restore command.
5. Write `runs/observe-review-20261123/COVERAGE.md`: every series, the panel it is on, and for anything the engine
   exposes that OpenObserve does not hold (or holds from one rank only), what would need to change in the scrape.
   Useful numbers that exist only as log lines and not as metrics (the round profile's `forward_ms`, `pass_ms`,
   `rows_a_round`; per-request `done` fields) belong on that list too, with the cheapest way to get each into
   OpenObserve.

## Deliverables

Under `runs/observe-review-20261123/`: `RESULTS.md` (inventory, the benchmark-against-live table and verdicts, the
dashboard changes, what remains unverified), `COVERAGE.md`, `dashboards-before/`, `dashboards-after/`, and the query
scripts (credentials from the environment only). Do not commit. Live logs may contain request text: do not copy
prompts or replies into any file.
