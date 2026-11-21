# Overnight Engram / engine A/B campaign: results

Status: **DONE 2026-11-13 21:39Z. Production restored and verified** (see "Final restored state").
- **Scope:** a single pass per candidate, as the user asked at 18:37Z. There were no 3x alternations, so nothing is
  labelled POSITIVE.
- **Clock lock:** the user applied it on both Sparks (`gpuClkMin 0, gpuClkMax 2400`). The agent only read clocks
  back, and **left the lock in place** for the user to reset (`nvidia-smi -rgc`).
- **Live traffic:** none arrived during the campaign (see "Protocol as run").

## Verdicts (single pass, one block per candidate with a base block on each side)

- The deltas are the candidate's median against the left and right base blocks.
- The band is the larger of the per-rep spread of all three blocks and the base-to-base gap.
- Decode figures are paired per rep index: every block decodes the same prompt at the same index, greedily.
- "SIGNAL +" or "SIGNAL -" means the candidate beat, or lost to, **both** adjacent bases by more than the band on a
  primary metric, in one alternation.
- **Bold** cells are outside the band.

| Cand. | Block (bases) | Prefill 8K / 32K / 100K | Decode 1 / 4 / 8 streams (paired) | Other | Mem low H/W GiB, cand (bases) | Armed proof | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A `ENGRAM_LEAD=1` | b03 (b02/b04) | +2.9/+1.8% (band 2.8) / **+4.3/+3.4%** (1.5) / **+3.6/+2.4%** (1.2) | -0.4/-0.0, -0.4/-0.3, -0.0/-0.4% (band <= 0.8) | p125 8K +6.2/+5.4% | 5.62/7.54 (5.99,5.88 / 7.18,7.59) | env both ranks; `engram-lead: on` both | **SIGNAL +** at >= 32K prefill; decode neutral |
| B `ENGRAM_SCALES=map` | b05 (b04/b06) | **+6.1/+8.8%** (3.6) / **+4.7/+5.6%** (1.9) / **+3.5/+4.2%** (0.7) | -0.1/-0.3, -2.4/-1.9, +0.1/+0.6% (within) | code 8K **+4.6/+4.6%**; **cold cost: first request after restart (2K warm-up) 257 tok/s vs base 1399/1438 (-82%); second (8K) 1119 vs 1907/1891 (-41%)**; first 32K/100K cold 2074/1982 (no cost) | 6.81/7.86 (5.88,5.86 / 7.59,7.92) | env both ranks | **SIGNAL +** in steady state (after one warm-up per size); large cold cost on the first two requests after a restart |
| C `ROWCACHE_MB=256` | b07 (b06/b08) | +0.9/-1.2% / -1.2/-2.0% / -16.3/-16.6% (band 26: one slow rep) | **-4.4/-3.8, -9.1/-9.3, -17.5/-17.1%** (band 1.1 / 3.6 / 7.2) | **code 8K -39%, p125 8K -61%** | 5.41/7.25 (5.86,6.37 / 7.92,7.92) | env both; `Engram row cache: 256 MiB` both | **SIGNAL - / RULED OUT**: a progressive regression that starts after ~370-470K prefill tokens since the restart (see "Row-cache onset") |
| C `ROWCACHE_MB=64` | b23 (b22/b24) | not measured | not measured | **rank pair broke 9 s after the cold 100K warm-up** (rank 0: `spill tier's agreement: rank 1: ReplyTimeout`, exit 75; rank 1 logged nothing after load) | - | env both; `Engram row cache: 64 MiB` both | **UNMEASURED** (pair broke); not re-run (RC256 already negative; minimal pass) |
| D production as deployed (`LEAD=1 ROWCACHE_MB=256 STRICT=1`) vs clean base | b09 (b08/b10) | **+3.0/+2.9%** (0.8) / **+2.5/+2.9%** (0.9) / **+1.2/+1.0%** (0.8) | **-4.3/-4.2, -10.5/-10.4, -17.6/-17.6%** (band 1.5 / 3.3 / 6.2) | code 8K -22%, p125 8K -55%; TTFT 8K 4.3/2.9% better | 5.32/7.05 (6.37,5.62 / 7.92,7.44) | env both; lead on, rc 256 | **SIGNAL -** once past the onset: same progressive regression as C256 (onset right after ~470K prefill tokens); before it, LEAD's prefill gain with no decode cost (i01: same flags, decode at base speed after 143K tokens) |
| D `LEAD=1 + SCALES=map` | c01 (b24/c02) | **+6.9/+7.5%** (3.9) / **+6.0/+6.1%** (0.3) / **+4.4/+5.1%** (0.7) | -0.1/-0.2, +0.8/+0.7, **+1.0/+0.8%** (band 0.4) | code 8K +2.5/+8.1% (within); cold: first request (2K) 262 tok/s vs base 1403/~1400, second (8K) 1341 vs 1904 | 6.79/8.02 (6.26,6.36 / 7.80,7.85) | env both; lead on both | **SIGNAL +**, largest prefill gain of the pass (A and B stack); decode neutral to +1% |
| E `GRAPH_CAP=512` (100K-context decode cells on) | b11 (b10/b12) | -0.3/-0.5 / -0.1/-0.3 / -0.4/-0.1% | +0.2/+0.2, +0.1/+0.1, +0.2/+0.5%; 100K-ctx lc1 -7.2/-3.7% (band 14.7), lc4 -0.1/-1.4% | **0 graphs destroyed in every block (64 kept, under the 128 cap)** | 5.76/7.43 | rank0 env + rank1 `--graph-cap 512` | **INCONCLUSIVE**: no cost, but the bench never reached the thrash regime |
| E `GRAPH_CAP=0` | b13 (b12/b14) | -0.5/+0.2 / -0.3/-0.0 / -0.4/+0.4% | +0.1/+0.2, -0.0/+0.0, +0.5/+0.4%; lc4 -6.7/-1.7% (band 5.1, n=1) | as above | 5.52/7.47 | as above | **INCONCLUSIVE** (same reason) |
| F `ENGRAM_AIO=0` | b15 (b14/b16) | **+4.0/+3.6%** (0.6) / **+1.8/+2.1%** (1.4) / **+3.4/+2.8%** (0.9) | -1.3/-1.4, -1.0/-0.7, +0.3/+0.2% (within) | TTFT 8K 2.3% better | 6.30/7.63 | rank0 env + rank1 `--aio 0` | **SIGNAL +** on prefill (the prior said neutral) |
| G `ENGRAM_FAST=1` | b17 (b16/b18) | **-2.3/-2.1%** (0.9) / +0.3/+0.1 / -0.4/-0.8% | within | TTFT 8K 6% worse | 5.93/7.84 | env both | **RULED OUT** (slightly negative at 8K, neutral elsewhere) |
| G `ENGRAM_DEDUPE=1` (LEAD=0, RC=0) | b19 (b18/b20) | +0.5/+2.9 / +0.3/+0.8 / -0.1/+0.1% | within | | 6.29/7.43 | env both | **RULED OUT** (neutral) |
| H `T25_DRAFT_CAL=0.05` | b21 (b20/b22; thinking-on code cells in all three) | +1.8/+1.4 / -0.1/-1.0 / -0.6/-0.7% (within) | -2.1/-2.3, -7.5/-7.4, **-11.9/-11.9%** (band 3.6) | thinking-on code decode: t1 -0.0/-0.1%, t4 -0.3/-0.0% (within) | 6.11/7.67 | env both | **SIGNAL - / RULED OUT**: -12% at 8-stream thinking-off decode, neutral with thinking on |
| I profile | i01 | - | - | see "Measurements"; production flags; decode at base speed (dec1 80.4, dec8 136.1), because i01 stayed under the row-cache onset | 5.43/7.38 | rank1 `--profile 1` in Cmd | measured (attribution only) |
| J host overhead | b04 | - | - | gap below the 0.1 s done-line resolution | - | - | measured; not separable |
| K `SPILL_MIN_TOKENS` | b02 -> b03 | - | - | 2048: resumed 4096 of 6103 after a restart | - | - | measured at 2048; **8192 not run** (spill fingerprint) |
| L KV shard decode | - | - | - | - | - | - | **NOT RUN** (dropped from the single pass) |

## What this means (inferred from the measurements above, not measured as such)

- **Row cache: the strongest finding, and conditional.** With `ROWCACHE_MB=256` (b07 alone, b09 the deployed combo),
  every request ran at base speed up to roughly 370-470K cumulative uncached prefill tokens after the restart. After
  that, every later request was slower, and it kept getting slower:
  - b07: the second 100K rep at 1421 vs 1846 tok/s; code 8K 1125-1168 tok/s; p125 909 -> 814 -> 746; decode -4% at
    1 stream, -9% at 4, -17% at 8.
  - b09: the onset came right after 470K, then code 1405-1516 and p125 1040 -> 852, with the same decode losses.
  - i01 had the same production flags but only 143K tokens before its decode cells, and it decoded at base speed
    (dec8 136.1 vs base 135-138).
  - So this is a state the cache reaches, not a constant cost.
  - **Likely, not measured:** production processes far more than 500K uncached tokens between restarts, so the live
    pair is probably in the degraded state most of the time. On this harness that state costs ~17% of 8-stream
    decode.
  - Removing the row cache (or, more cheaply, a LEAD+STRICT-only restart) is **the operator's call**. Production was
    restored *with* the row cache on, as rule 7 requires.
- **The strongest single-pass win is LEAD + SCALES=map:** +6-7% at 8K/32K and +4-5% at 100K, with decode neutral.
  - Cost: the first two requests after each restart (the mapped scales fault in: the first request ran -82% and the
    second -41%).
  - It also leaves ~0.5 GiB *more* head MemAvailable.
  - **F (AIO=0)** is also positive alone. Do not combine it with LEAD or map untested: LEAD needs the AIO ring, and
    the earlier matrix measured AIO=0+map at -44%.
- **These are single-pass signals.** Each needs 2 more ABAB alternations, plus `tools/validate.py --compare`, before
  it can be called POSITIVE. Earlier matrix records already read `ident` for lead, map and aio0.
- **The production client has sent nothing since 15:58Z.** The pair broke at 16:00Z, and the client did not
  reconnect during the campaign: 0 foreign POSTs in all saved block logs, against 1,629 of the campaign's own. The
  pair is restored, but the operator's agents are not back.

## Row-cache onset (per-request timeline; cum = uncached prefill tokens since the restart)

| Block | flags | at cum ~370K | first slow request | after |
| --- | --- | --- | --- | --- |
| b07 | RC256 | 100K rep 1: 1846 tok/s (base 1959) | 100K rep 2 (cum 369-469K): 1421 | code 1168/1163/1125; p125 909/814/746; dec1 68/77/85 (base 71/81/89); dec8 116/112/111 (base 135-138) |
| b09 | LEAD+RC256+STRICT | 100K reps: 1989/1973 (LEAD-level) | code rep 1 (cum 470-479K): 1516 | code 1474/1405; p125 1040/930/852; dec8 115/112/111 |
| i01 | LEAD+RC256+STRICT | (only 143K total) | none | dec1 71/80/88, dec8 135/136/138 (base level) |
| b08 | base | 100K: 1959/1957 | none | code 1843-1918, p125 2079-2088, dec8 135-138 |

The degraded cells ran last in each block, so the cell order (code, then p125, then decode) is confounded with time.
The size of the loss on each corpus says nothing about the corpus itself: the "p125 is hurt most" reading is
withdrawn.

## Production settings as found (2026-11-13 ~08:30Z)

- Image on both ranks: `tensorfold-dsv41:exl3-tp2-cdd4715dcbe1`. Containers created 2026-11-13T07:07:40Z (rank 0 at
  .339Z, rank 1 at .059Z). `--cpuset-cpus 15-19` on both.
- Full TF_*/TENSORFOLD_*/HF_HUB env saved here: `runs/overnight/production.rank0.env` (49 lines) and
  `production.rank1.env` (35 lines).
- Switches under test, in the live containers on BOTH ranks:
  `TF_DS_ENGRAM_LEAD=1`, `TF_DS_ENGRAM_ROWCACHE_MB=256`, `TF_DS_ENGRAM_STRICT=1`, `TF_DS_SPILL_MIN_TOKENS=2048`.
  Not set (defaults): `TF_DS_ENGRAM_SCALES` (io), `TF_DS_ENGRAM_DEDUPE`, `TF_DS_ENGRAM_FAST`, `TF_DS_T25_DRAFT_CAL` (0),
  `TF_DS_ENGRAM_AIO_SLOTS` (4096). Recipe settings: `GRAPH_CAP=128` (rank 0 `TF_DS_GRAPH_CAP=128`, rank 1
  `--graph-cap 128`), `ENGRAM_AIO=1`.
- Where they are configured:
  - `SPILL_MIN_TOKENS=2048`, along with `PARALLEL=8`, `PORT=8000`, `SPILL_GIB=180`, `SPILL_TURNS=0`, `KV_SHARD=1`,
    `TF_DS_T13_EXPERT_PRUNE=1`, `TF_DS_T26_KEYED_DRAFT=1`, `TF_DS_T27_DEVICE_DRAW=1`, `TF_DS_PREFILL_SLICE=5` and
    `TF_DS_RDMA_CPUS=15`, comes from the head's `~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/.env`.
  - `TF_DS_ENGRAM_LEAD=1`, `TF_DS_ENGRAM_ROWCACHE_MB=256` and `TF_DS_ENGRAM_STRICT=1` are **not in any file**. There is
    no `scripts/local.sh` on the head. They are not in `.env`, `~/.bashrc`, `~/.profile`, `/etc/environment`, or a
    systemd user unit. So they were exported in the shell that ran the 07:07Z restart, most likely
    `just restart-head flags="..."`. **A plain `./start.sh restart` on the head would drop all three.** To restore
    production, run `TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_ROWCACHE_MB=256 TF_DS_ENGRAM_STRICT=1 ./start.sh restart` with
    `.env` as it is now.
- Rank 0's startup log confirms `Engram row cache: 256 MiB (rank 0)`.
- Off values, read from the patches:
  - LEAD, DEDUPE, FAST and STRICT are on only for exactly `1`, so `0` is off.
  - `ROWCACHE_MB`: unset or `0` is off (0114 `budgetOf`).
  - `SCALES`: anything other than `map` is off; `io` is the default.
  - `T25_DRAFT_CAL`: `0` is off.
- Live traffic at 08:31Z: 434 requests since 07:07Z (~5.2/min), mean latency 41.5 s, so mean concurrency is about
  3.6. In a 60 s sample (30 reads, 2 s apart), `requests_running` was 1-2 the whole time and never 0. There were 32
  done lines in the last 10 min, with prompts of 6.9K-209K tokens, mostly cached (e.g. 206,848 of 208,351).
  **This matters for the campaign design.** At this rate almost every timed block would see foreign requests. The
  per-block contamination rule would mark nearly everything "contaminated" unless blocks are made per-request
  (an idle gate plus a request-count delta per rep), or the run waits for a quieter period.

## GPU clock cap (operator requirement: 2400 MHz max on both Sparks for the whole of testing)

Pre-existing clock state, recorded 2026-11-13 08:32Z:

| Spark | GPU | current gr | max gr | applications gr (default) | lock | temp / power |
| --- | --- | --- | --- | --- | --- | --- |
| head (gx10-eb0e) | NVIDIA GB10, driver 580.159.03 | 2411 MHz | 3003 MHz | 2418 (2418) | none visible: all clock event reasons Not Active, no locked-clock field in `nvidia-smi -q` | 74-77 C / 51-60 W |
| worker (10.50.0.11) | NVIDIA GB10 | 2418-2431 MHz | 3003 MHz | 2418 (2418) | none visible | 67-69 C / 53-72 W |

The current clocks under live load already sit slightly above 2400 (2411-2431 MHz). That is enough to disqualify every
block under the "a block where either GPU exceeded 2400 MHz does not count" rule. `SUPPORTED_CLOCKS` lists no
graphics clocks on GB10. The help text shows the lock option is `-lgc/--lock-gpu-clocks=<min,max>` (bare `-lgc --help`
is rejected), plus `-rgc` to reset.

What I tried:

1. `sudo -n true` on both Sparks: passwordless sudo works on both (SUDO_OK).
2. `sudo -n nvidia-smi -lgc 0,2400` on the head, then on the worker through the head. **This was blocked before it
   ran**, by the agent harness's permission policy (the Claude Code auto-mode classifier, category "Destructive or
   Unwanted Actions"), not by the GPU or driver. No `nvidia-smi` lock command reached either Spark.
   Verified afterwards: clocks unchanged (head 2411 MHz, worker 2418 MHz), and both containers are still the 07:07:40Z
   ones.

Not established: whether GB10 / driver 580.159.03 accepts `-lgc` at all, and whether a lock survives a container
restart. Both need the command to run.

To resume, the operator either grants this permission (allow
`sudo -n nvidia-smi -lgc 0,2400` and `sudo -n nvidia-smi -rgc` on both Sparks) or applies the lock by hand. Restoring
afterwards would be `sudo nvidia-smi -rgc` on both (no lock existed before).

The operator then applied the lock themselves; see the Status line at the top.

## Incident before the resume: production down from 16:00Z (no benchmark running)

- At 16:00:06Z, under the production flags, rank 0 logged `the rank pair broke (a round or prompt chunk made no
  progress for 120 s ...)`, then `rank 0's NCCL communicator aborted`, and at 16:00:24Z it exited with status 75.
  - The last request was `prompt=310094`, which failed after 142 s.
  - The request before it (308,612 tokens) had taken 86.6 s TTFT for a 3.4K-token uncached tail and decoded at
    2.5 tok/s.
- Rank 1's container was still `Up` at 16:59Z (orphaned). No supervisor restarted the pair, so the API was down
  from 16:00Z until the campaign's first restart (17:13Z).
- No cause is attributed. When the clock lock was applied relative to 16:00Z is not known.
- Logs: `runs/overnight/break-1600Z.rank0.log` and `break-1600Z.rank1.log`.

## Harness (new files only; nothing synced; image unchanged)

- `tools/bench_div.py`: high-diversity corpus and guarded client.
  - "div" corpus: 406,518 deduplicated English lines from the head's /usr/share/doc, licenses and repo markdown
    (24.6 M chars, 73,680 distinct words), shuffled with the salt in 3-8-line paragraphs.
  - "code" corpus: 8,890 30-line blocks of the Python 3.12 stdlib.
  - "p125": the old 125-word prose, kept for comparison and labelled.
  - Every prefill prompt starts with a fresh salt. Prompt sizes land within about 1% of target (e.g. 8,104-8,210 for
    8K).
  - Decode prompts are distinct per rep and per stream, with fixed seeds, so every block decodes the same prompts
    (paired comparison).
  - Each rep waits for an idle gap (`requests_running == 0`, at most 120 s). It is clean only if the completed-request
    counter delta equals its own requests, nothing is running 10 s after it ends, there are no errors, and (for
    prefill) `cached_tokens == 0`. Up to 3 retries, each with a fresh prompt.
- `scripts/bench_overnight.py`: ABAB runner, detached on the head; the schedule is `runs/overnight/schedule.json`.
  - Per block: wait for idle (at most 10 min), then `./start.sh restart` with every switch explicit.
  - Checks the image tag in the restart log and in `docker inspect` of both ranks.
  - Saves both ranks' TF_* env and checks every armed value (`arm_proof_problems` was empty in every block).
  - Samples clock, temperature, power and MemAvailable every 3 s on both Sparks, starting after the restart.
  - Saves both ranks' engine logs, the done lines and the Engram histogram delta.
  - Restores production in a `finally`.
- `tools/overnight_report.py`: generates the tables at the end of this file.
- Base = `TF_DS_ENGRAM_LEAD=0 TF_DS_ENGRAM_ROWCACHE_MB=0 TF_DS_ENGRAM_STRICT=0 TF_DS_ENGRAM_SCALES=io
  TF_DS_ENGRAM_DEDUPE=0 TF_DS_ENGRAM_FAST=0 TF_DS_T25_DRAFT_CAL=0 GRAPH_CAP=128 ENGRAM_AIO=1`. Everything else is
  production (`.env`, including `SPILL_MIN_TOKENS=2048`). A `DRY_RUN=1` check before the first restart confirmed
  image `cdd4715dcbe1` and both ranks' values.
- Cells per block:
  - warm: one discarded prefill per size (2K, 8K, 32K, 100K), recorded as the cold figure;
  - prefill_div: 2K, 8K and 32K x3, 100K x2;
  - prefill_code 8K x3 and prefill_p125 8K x3;
  - dec1 x3 and dec8 x3 (512 tokens), dec4 x2;
  - longctx (E blocks and their bases only): resumes the 100K prompt, then decodes 1024 tokens at 1 stream (x2) and
    4 streams (x1);
  - think (H block and its bases only): code prompts, thinking on, 768 tokens, 1 stream (x2) and 4 streams (x1).

## Protocol as run

- **Scope cut by the user (18:37Z):** "the minimal test number to get signal ... just a single pass". The queued
  3x-alternation blocks (a01-a08) were removed.
- **Noise:** base-to-base for b01 vs b02 (consecutive restarts):
  - prefill: 2K 8.9%, 8K 0.7%, 32K 0.0%, 100K 0.7%;
  - decode, paired: 1 stream 0.0%, 4 streams 0.6%, 8 streams 0.1%.
  - Across all 14 valid base blocks (b01-b24, c02), 8K prefill ran 1842-1903 tok/s, 32K 1971-1992 and 100K
    1939-1972. 2K prefill is noisy (1475-1608) and is not used for verdicts.
- **Traffic and contamination:** no foreign request arrived after the first restart. Across all saved block logs,
  POSTs not from 127.0.0.1 = **0** (own: 1,629). The production client never
  reconnected (its last request was 15:58Z), and `docker logs` shows only 127.0.0.1 requests.
  - Every rep of every block was clean on its first attempt.
  - The only unclean attempts were the smoke block's 12 dec8 attempts. They were caused by a harness bug (the running
    gauge lags the done counter), fixed before b01. The smoke block (`s00-smoke`) is excluded from comparisons.
- **Clock cap:** every block's maximum readback was <= 2392 MHz (head) and <= 2385 MHz (worker), with a loaded median
  of 2372 MHz, sampled every 3 s on both Sparks (`<block>.{head,worker}.samples`).
  - The lock survived every container restart. No block was invalid.
  - GPU temperature across blocks: head 47-87 C, worker 45-80 C.
- **Restarts:** 30 in total.
  - 1 smoke, 27 blocks (b01 was adopted from the restart the runner had already made), and 2 restores.
  - Each took about 125 s; the first, after the 16:00Z break, took 214 s.
  - One runner relaunch (17:20Z) to fix the gauge-lag bug. Only the runner's Python was stopped; start.sh was left to
    finish.
  - No production client was connected during any restart.

## Measurements (items I, J, K)

- **I profile** (block i01: production flags plus `RANK1_PROFILE=1`, a recipe setting; rank 1's `--profile 1`
  confirmed in its Cmd). Rank 1 prints its profile only when rank 0 stops first, which `stop.sh` does at the next
  restart. The line was read from the worker's saved log (`runs/overnight/i01.rank1.profile.log`):
  `{"rounds": 1620, "enqueue_ms": 14.32, "forward_ms": 54.55, "prefill_engram_ms": {"calls": 200, "rows": 0.00,
  "upload": 0.20, "mm": 4.56, "exchange": 8.31, "gate": 2.78}, "engram_read_ms": 0.674, "engram_hash_ms": 0.975}`.
  - Units: averages per decode round, or per prefill Engram call.
  - The profile synchronizes at every chunk and round, so it is for attribution only.
  - Reading: Engram host work in a decode round is ~1.6 ms (0.67 read + 0.98 hash) of a ~55 ms forward. In prefill,
    the Engram exchange (8.3 ms) and the matmul (4.6 ms) dominate; row reads round to 0.00 ms per call (hidden behind
    compute).
  - Caveat: i01 stayed under the row-cache onset (143K tokens), so this split describes the healthy state, not the
    degraded one.
  - Rank 0's own per-round profile needs the `--profile` serve argument (not an env/recipe setting), so it was not
    captured.
- **J host overhead** (b04, base): a 30,315-token system prompt plus a short user turn, 10 warm repeats (28,672
  cached each).
  - Client TTFT was 1.230-1.298 s (median 1.232); the server done line's `ttft_s` and `prefill_s` read 1.2-1.3.
  - The gap is below the done line's 0.1 s resolution, so render, tokenize and queue time cannot be separated this
    way.
  - Each warm turn still costs ~1.23 s for ~1,643 uncached tokens, against 15.3 s for the cold first send.
- **K SPILL_MIN_TOKENS**:
  - `spill_min_tokens` is in the ranks' spill fingerprint: the same `rank_identity.featureString` as
    `edit_mark`/`keep_protect`, whose toggles the brief forbids because they change the spill namespace and delete
    the disk cache. So **8192 was not run**.
  - At the production value 2048: a 6.1K-token session was seeded in b02, the pair restarted, and the session was
    resumed first thing in b03. Result: `cached_tokens` 4096 of 6103, TTFT 1.55 s; spill reads 1 per rank (3.6 MB on
    rank 0, 0.7 MB on rank 1). Short sessions do recover from disk at 2048.
- **L**: not run. `docs/spill-and-kv-sharding.md` lists `KV_SHARD_DECODE` = empty (`fetch`) or `owner`
  ("owner-computed decode") as a supported setting with `KV_SHARD=1`. It was dropped from the single pass.

## Other measured findings

- **The old 125-word corpus flatters prefill:** on every base block, `p125` at 8K is about 10% faster than diverse
  text (b01: 2073 vs 1891 tok/s), so the earlier matrix's absolute numbers are optimistic.
- **Rank-pair breaks:** both of today's breaks had the row cache armed: production RC256 at 16:00Z, and b23 RC64 at
  20:59Z after a cold 100K prefill. No other block broke, though RC256 also ran in b07, b09 and i01 without breaking.
  That is a correlation in 2 events; no cause is attributed.
- **GRAPH_CAP:** the harness never filled the 128-graph cap (64 kept at every block end, 0 destroyed). Production's
  rank 1 log before the break showed `128 kept (the cap) ... 64 destroyed` within 80 min of agent traffic, so the
  thrash exists in production but was not reproduced, and E stays unproven.
- **Disk:** the spill tier grew from 4.1 to 13.2 GiB (head) and from 4.3 to 13.5 GiB (worker) from the campaign's
  prompts, out of 180 GiB. Nothing was evicted.
- **Memory (MemAvailable low per block):** head 5.3-6.8 GiB, worker 7.0-8.0 GiB.
  - The row-cache arms were lowest (head 5.3-5.4 GiB), about 0.5-1.0 GiB under their bases.
  - The map arms were highest (head 6.8 GiB).

## Not done, and why

- 3x alternation and the POSITIVE label: the user cut the run to a single pass.
- `validate.py --compare` bit-identity for positives: none reached POSITIVE under that scope.
- C64: the pair broke, so it is unmeasured.
- K at 8192: spill namespace risk.
- L: dropped from the pass.
- Row-cache combos (LEAD+RC, LEAD+map+RC): the row cache is individually negative.
- Rank 0's per-round profile: needs a CLI argument.
- The clock lock was left in place, as instructed.

## Final restored state (2026-11-13 21:39Z)

- Restart with `TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_ROWCACHE_MB=256 TF_DS_ENGRAM_STRICT=1 TF_DS_CACHE_METRICS=1` plus the
  head's unchanged `.env` (`SPILL_MIN_TOKENS=2048`).
- The first restore (21:36Z) omitted `TF_DS_CACHE_METRICS=1`. The recorded production env carries it explicitly, with
  the default value. Its diff was exactly that one line per rank; it is kept as
  `restore-try1-missing-cache-metrics.json`. The corrected restore followed at 21:39Z.
- `docker inspect` TF_*/TENSORFOLD_*/HF_HUB env against `production.rank0.env` / `production.rank1.env`:
  **missing [] and extra [] on both ranks** (`restore.json`, `env_match: true`).
- Image `tensorfold-dsv41:exl3-tp2-cdd4715dcbe1` on both ranks, containers created 21:37:33-34Z and running.
- Banners on both ranks: `Engram row cache: 256 MiB` and `engram-lead: on`. 0 `error` lines in either rank's log.
- `/health`: status ok and `pair.broken: false`; the spill tier recovered 845 points per rank.
- A chat completion returned `"OK"` (finish `stop`).
- GPU clocks after the restore: head 2385 MHz, worker 2379 MHz. The lock is still in place (`clocks.max.gr` reads
  3003 MHz regardless).

<!-- TABLES START -->
## Generated tables (tools/overnight_report.py; refreshed at every check-in)

Only clean reps count. Values are the median ± (max-min)/median over clean reps, with n = clean reps. Decode metrics (dec*, lc*, t*) are compared paired per rep index (same prompt and greedy output in every block), and their band is the spread of the paired ratios plus the base-to-base paired deviation. Prefill band = max(per-rep spread of the candidate and of both adjacent bases, the base-to-base gap).

| block | arm | status | clock max H/W | temp H/W | mem low H/W GiB | pd.2000 | pd.8000 | pd.32000 | pd.100000 | ttft.8000 | pc.8000 | pp.8000 | dec1 | dec4 | dec8 | lc1 | lc4 | t1 | t4 | cold.8000 | cold.100000 | clean reps/reps (attempts) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s00-smoke | base | ok | 2392/2385 | 80/74 | 6.18/7.92 | - | - | - | - | - | 1851.5 ±4.3% (n3) | - | 81.3 ±22.0% (n3) | - | - | - | - | - | - | - | - | 6/9 (18) |
| b01 | base | ok | 2392/2385 | 85/78 | 6.08/7.88 | 1605.7 ±8.2% (n3) | 1891.3 ±1.7% (n3) | 1972.7 ±0.6% (n3) | 1956.2 ±0.4% (n2) | 4.3 ±2.7% (n3) | 1868.7 ±1.2% (n3) | 2072.8 ±3.4% (n3) | 80.7 ±22.1% (n3) | 117.8 ±8.7% (n2) | 136.0 ±2.1% (n3) | - | - | - | - | 1868.2 (n1) | 1945.5 (n1) | 25/25 (25) |
| b02 | base | ok | 2385/2385 | 86/79 | 5.99/7.18 | 1474.7 ±7.9% (n3) | 1878.9 ±0.5% (n3) | 1972.5 ±1.5% (n3) | 1943.4 ±0.1% (n2) | 4.3 ±1.2% (n3) | 1862.7 ±4.1% (n3) | 2073.4 ±1.0% (n3) | 80.7 ±21.5% (n3) | 118.5 ±8.7% (n2) | 136.1 ±1.8% (n3) | - | - | - | - | 1885.8 (n1) | 1921.2 (n1) | 25/25 (25) |
| b03 | A-lead | ok | 2385/2385 | 86/80 | 5.62/7.54 | 1572.8 ±6.2% (n3) | 1934.1 ±2.8% (n3) | 2057.9 ±0.2% (n3) | 2014.3 ±0.1% (n2) | 4.2 ±6.7% (n3) | 1925.3 ±2.6% (n3) | 2202.8 ±0.1% (n3) | 80.6 ±21.2% (n3) | 118.0 ±7.9% (n2) | 136.3 ±2.3% (n3) | - | - | - | - | 1962.7 (n1) | 2014.8 (n1) | 25/25 (25) |
| b04 | base | ok | 2385/2385 | 87/79 | 5.88/7.59 | 1489.7 ±8.0% (n3) | 1899.7 ±0.9% (n3) | 1990.5 ±0.8% (n3) | 1966.5 ±0.1% (n2) | 4.3 ±0.2% (n3) | 1905.4 ±2.4% (n3) | 2089.3 ±0.3% (n3) | 80.6 ±21.7% (n3) | 118.3 ±8.6% (n2) | 137.0 ±2.1% (n3) | - | - | - | - | 1906.7 (n1) | 1946.5 (n1) | 25/25 (25) |
| b05 | B-map | ok | 2392/2385 | 87/79 | 6.81/7.86 | 1620.2 ±5.0% (n3) | 2015.6 ±2.6% (n3) | 2083.7 ±0.0% (n3) | 2034.9 ±0.0% (n2) | 4.0 ±4.7% (n3) | 1993.3 ±3.1% (n3) | 2270.6 ±62.4% (n3) | 80.7 ±23.3% (n3) | 115.4 ±4.7% (n2) | 137.4 ±2.4% (n3) | - | - | - | - | 1119.0 (n1) | 1981.9 (n1) | 25/25 (25) |
| b06 | base | ok | 2385/2385 | 86/79 | 5.86/7.92 | 1568.7 ±7.1% (n3) | 1852.5 ±3.6% (n3) | 1972.7 ±1.9% (n3) | 1952.2 ±0.6% (n2) | 4.4 ±6.6% (n3) | 1905.3 ±2.9% (n3) | 2088.9 ±1.2% (n3) | 81.0 ±21.5% (n3) | 117.7 ±7.9% (n2) | 136.1 ±2.2% (n3) | - | - | - | - | 1890.7 (n1) | 1947.9 (n1) | 25/25 (25) |
| b07 | C-rc256 | ok | 2392/2385 | 84/77 | 5.41/7.25 | 1504.6 ±8.0% (n3) | 1869.5 ±1.6% (n3) | 1949.9 ±1.7% (n3) | 1633.2 ±26.0% (n2) | 4.3 ±2.2% (n3) | 1162.9 ±3.7% (n3) | 814.2 ±20.0% (n3) | 77.4 ±22.4% (n3) | 107.1 ±11.5% (n2) | 112.3 ±5.0% (n3) | - | - | - | - | 1866.4 (n1) | 1944.5 (n1) | 25/25 (25) |
| b08 | base | ok | 2385/2385 | 86/78 | 6.37/7.92 | 1593.5 ±4.4% (n3) | 1892.0 ±0.7% (n3) | 1989.2 ±0.2% (n3) | 1958.0 ±0.1% (n2) | 4.3 ±1.7% (n3) | 1894.9 ±4.0% (n3) | 2084.3 ±0.5% (n3) | 80.5 ±21.5% (n3) | 117.9 ±8.6% (n2) | 135.5 ±2.2% (n3) | - | - | - | - | 1893.1 (n1) | 1961.5 (n1) | 25/25 (25) |
| b09 | D-prod | ok | 2392/2385 | 86/80 | 5.32/7.05 | 1485.5 ±9.7% (n3) | 1948.7 ±0.3% (n3) | 2038.5 ±0.9% (n3) | 1980.8 ±0.8% (n2) | 4.1 ±0.2% (n3) | 1474.0 ±7.5% (n3) | 929.7 ±20.2% (n3) | 77.2 ±22.6% (n3) | 105.6 ±11.3% (n2) | 111.7 ±4.0% (n3) | - | - | - | - | 1941.2 (n1) | 2001.5 (n1) | 25/25 (25) |
| b10 | base | ok | 2385/2385 | 87/78 | 5.62/7.44 | 1476.7 ±1.4% (n3) | 1893.5 ±0.8% (n3) | 1981.2 ±0.5% (n3) | 1961.2 ±0.4% (n2) | 4.3 ±0.5% (n3) | 1853.9 ±3.0% (n3) | 2081.5 ±0.6% (n3) | 80.5 ±21.8% (n3) | 117.8 ±8.0% (n2) | 135.6 ±2.0% (n3) | 76.8 ±14.0% (n2) | 119.5 (n1) | - | - | 1849.5 (n1) | 1965.0 (n1) | 28/28 (28) |
| b11 | E-cap512 | ok | 2385/2385 | 86/78 | 5.76/7.43 | 1460.9 ±3.7% (n3) | 1887.5 ±0.2% (n3) | 1979.5 ±1.6% (n3) | 1952.9 ±0.3% (n2) | 4.3 ±2.2% (n3) | 1821.6 ±2.6% (n3) | 2081.8 ±0.3% (n3) | 80.7 ±21.4% (n3) | 117.9 ±8.3% (n2) | 135.8 ±1.9% (n3) | 70.9 ±0.7% (n2) | 119.4 (n1) | - | - | 1846.4 (n1) | 1960.2 (n1) | 28/28 (28) |
| b12 | base | ok | 2385/2385 | 86/78 | 5.69/7.42 | 1600.9 ±8.3% (n3) | 1896.8 ±3.2% (n3) | 1986.4 ±0.4% (n3) | 1954.5 ±0.3% (n2) | 4.3 ±6.4% (n3) | 1857.4 ±4.0% (n3) | 2085.6 ±5.7% (n3) | 80.5 ±22.1% (n3) | 117.8 ±8.6% (n2) | 135.1 ±2.5% (n3) | 73.7 ±4.0% (n2) | 121.1 (n1) | - | - | 1893.9 (n1) | 1956.3 (n1) | 28/28 (28) |
| b13 | E-cap0 | ok | 2385/2379 | 87/79 | 5.52/7.47 | 1458.5 ±4.6% (n3) | 1887.0 ±0.9% (n3) | 1979.9 ±0.2% (n3) | 1946.0 ±0.3% (n2) | 4.3 ±1.5% (n3) | 1867.3 ±3.1% (n3) | 2088.8 ±0.6% (n3) | 80.6 ±21.5% (n3) | 117.7 ±8.4% (n2) | 135.7 ±1.9% (n3) | 71.9 ±10.2% (n2) | 112.9 (n1) | - | - | 1895.2 (n1) | 1952.0 (n1) | 28/28 (28) |
| b14 | base | ok | 2385/2385 | 86/78 | 5.73/7.41 | 1540.2 ±4.5% (n3) | 1883.7 ±0.2% (n3) | 1980.1 ±0.4% (n3) | 1939.2 ±0.7% (n2) | 4.3 ±2.2% (n3) | 1882.1 ±3.7% (n3) | 2080.1 ±0.1% (n3) | 80.5 ±21.6% (n3) | 117.7 ±8.3% (n2) | 134.8 ±2.1% (n3) | 74.8 ±8.8% (n2) | 114.9 (n1) | - | - | 1832.3 (n1) | 1961.8 (n1) | 28/28 (28) |
| b15 | F-aio0 | ok | 2385/2385 | 87/80 | 6.3/7.63 | 1488.1 ±21.4% (n3) | 1959.6 ±0.6% (n3) | 2016.5 ±1.1% (n3) | 2005.1 ±0.9% (n2) | 4.2 ±1.7% (n3) | 1936.3 ±14.3% (n3) | 2132.4 ±12.6% (n3) | 79.5 ±26.9% (n3) | 116.4 ±6.9% (n2) | 136.3 ±2.8% (n3) | - | - | - | - | 1913.4 (n1) | 1895.9 (n1) | 25/25 (25) |
| b16 | base | ok | 2385/2385 | 87/78 | 6.15/7.95 | 1539.1 ±10.2% (n3) | 1892.0 ±0.3% (n3) | 1974.8 ±1.4% (n3) | 1950.9 ±0.2% (n2) | 4.3 ±1.3% (n3) | 1844.6 ±5.0% (n3) | 2088.7 ±0.3% (n3) | 80.6 ±23.5% (n3) | 117.3 ±8.0% (n2) | 135.2 ±2.7% (n3) | - | - | - | - | 1893.7 (n1) | 1961.6 (n1) | 25/25 (25) |
| b17 | G-fast | ok | 2385/2385 | 87/78 | 5.93/7.84 | 1583.8 ±6.7% (n3) | 1847.6 ±0.9% (n3) | 1980.8 ±0.5% (n3) | 1943.5 ±0.4% (n2) | 4.5 ±2.1% (n3) | 1895.4 ±4.1% (n3) | 2087.8 ±0.4% (n3) | 80.6 ±21.8% (n3) | 117.7 ±8.3% (n2) | 136.0 ±2.3% (n3) | - | - | - | - | 1895.2 (n1) | 1964.9 (n1) | 25/25 (25) |
| b18 | base | ok | 2385/2385 | 87/79 | 5.73/7.9 | 1491.1 ±2.3% (n3) | 1886.4 ±0.3% (n3) | 1978.9 ±1.7% (n3) | 1958.4 ±0.1% (n2) | 4.3 ±2.6% (n3) | 1883.7 ±2.0% (n3) | 2079.4 ±0.3% (n3) | 80.7 ±21.5% (n3) | 117.5 ±7.9% (n2) | 135.8 ±2.3% (n3) | - | - | - | - | 1899.2 (n1) | 1946.1 (n1) | 25/25 (25) |
| b19 | G-dedupe | ok | 2385/2379 | 87/78 | 6.29/7.43 | 1492.8 ±8.5% (n3) | 1895.9 ±3.1% (n3) | 1985.4 ±0.9% (n3) | 1957.0 ±0.2% (n2) | 4.3 ±5.3% (n3) | 1950.1 ±1.0% (n3) | 2094.0 ±0.2% (n3) | 80.7 ±22.2% (n3) | 118.3 ±8.6% (n2) | 135.8 ±2.1% (n3) | - | - | - | - | 1902.8 (n1) | 1958.4 (n1) | 25/25 (25) |
| b20 | base | ok | 2385/2385 | 87/78 | 6.15/7.79 | 1536.6 ±7.0% (n3) | 1842.0 ±2.8% (n3) | 1970.5 ±1.0% (n3) | 1954.8 ±0.3% (n2) | 4.4 ±4.4% (n3) | 1909.6 ±3.2% (n3) | 2090.4 ±0.9% (n3) | 80.5 ±22.0% (n3) | 117.7 ±8.0% (n2) | 135.9 ±2.4% (n3) | - | - | 96.7 ±14.9% (n2) | 138.0 (n1) | 1899.9 (n1) | 1970.4 (n1) | 28/28 (28) |
| b21 | H-t25 | ok | 2385/2385 | 87/78 | 6.11/7.67 | 1509.6 ±6.3% (n3) | 1875.2 ±0.3% (n3) | 1969.4 ±0.3% (n3) | 1942.6 ±0.7% (n2) | 4.3 ±0.2% (n3) | 1862.4 ±0.8% (n3) | 2072.4 ±0.0% (n3) | 78.8 ±19.3% (n3) | 109.1 ±16.9% (n2) | 121.3 ±2.3% (n3) | - | - | 96.7 ±14.6% (n2) | 137.6 (n1) | 1886.9 (n1) | 1953.5 (n1) | 28/28 (28) |
| b22 | base | ok | 2385/2385 | 87/78 | 6.14/7.62 | 1567.3 ±6.4% (n3) | 1848.7 ±3.6% (n3) | 1989.5 ±0.3% (n3) | 1955.4 ±0.4% (n2) | 4.4 ±6.3% (n3) | 1860.5 ±3.0% (n3) | 2087.0 ±0.4% (n3) | 80.7 ±21.4% (n3) | 117.6 ±7.7% (n2) | 135.6 ±2.6% (n3) | - | - | 96.7 ±14.7% (n2) | 137.6 (n1) | 1901.5 (n1) | 1962.4 (n1) | 28/28 (28) |
| b23 | C-rc64 | broken BAIL:prefill_div: server down | 2392/2385 | 77/69 | 5.95/7.97 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1862.8 (n1) | 701.9 (n1) | 0/0 (0) |
| b24 | base | ok | 2385/2379 | 85/76 | 6.26/7.8 | 1608.4 ±9.2% (n3) | 1902.7 ±0.2% (n3) | 1992.4 ±0.1% (n3) | 1972.1 ±0.2% (n2) | 4.2 ±2.5% (n3) | 1902.9 ±3.6% (n3) | 2096.6 ±0.4% (n3) | 80.6 ±21.7% (n3) | 117.8 ±8.1% (n2) | 135.7 ±2.2% (n3) | - | - | - | - | 1904.2 (n1) | 1966.5 (n1) | 25/25 (25) |
| c01 | D-lead+map | ok | 2392/2385 | 87/80 | 6.79/8.02 | 1607.3 ±2.4% (n3) | 2033.9 ±1.0% (n3) | 2112.8 ±0.1% (n3) | 2058.1 ±0.2% (n2) | 4.0 ±1.7% (n3) | 1949.8 ±8.3% (n3) | 2180.2 ±41.1% (n3) | 80.7 ±26.5% (n3) | 118.8 ±8.6% (n2) | 137.5 ±2.1% (n3) | - | - | - | - | 1341.4 (n1) | 2066.9 (n1) | 25/25 (25) |
| c02 | base | ok | 2385/2385 | 87/77 | 6.36/7.85 | 1587.2 ±6.7% (n3) | 1891.9 ±3.9% (n3) | 1991.0 ±0.3% (n3) | 1959.1 ±0.2% (n2) | 4.3 ±7.4% (n3) | 1804.4 ±1.9% (n3) | 2088.9 ±0.7% (n3) | 80.5 ±22.0% (n3) | 118.0 ±7.9% (n2) | 136.5 ±1.9% (n3) | - | - | - | - | 1903.0 (n1) | 1957.3 (n1) | 25/25 (25) |
| i01 | I-profile(prod) | ok | 2385/2379 | 81/74 | 5.43/7.38 | - | - | - | - | - | - | - | 80.4 ±20.7% (n3) | - | 136.1 ±2.7% (n3) | - | - | - | - | 1900.2 (n1) | 2004.4 (n1) | 6/6 (6) |

### b03 A-lead vs b02 / b04
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1572.8 | 1474.7 | 1489.7 | +6.6% | +5.6% | 8.0% | within | (3, 3, 3) |
| pd.8000 (primary) | 1934.1 | 1878.9 | 1899.7 | +2.9% | +1.8% | 2.8% | within | (3, 3, 3) |
| pd.32000 (primary) | 2057.9 | 1972.5 | 1990.5 | +4.3% | +3.4% | 1.5% | WIN | (3, 3, 3) |
| pd.100000 (primary) | 2014.3 | 1943.4 | 1966.5 | +3.6% | +2.4% | 1.2% | WIN | (2, 2, 2) |
| ttft.8000 (primary) | 4.2 | 4.3 | 4.3 | +1.9% | +0.6% | 6.7% | within | (3, 3, 3) |
| pc.8000 | 1925.3 | 1862.7 | 1905.4 | +3.4% | +1.0% | 4.1% | within | (3, 3, 3) |
| pp.8000 | 2202.8 | 2073.4 | 2089.3 | +6.2% | +5.4% | 1.0% | WIN | (3, 3, 3) |
| dec1 (primary) | 80.6 | 80.7 | 80.6 | -0.4% | -0.0% | 0.6% | within (paired) | (3, 3, 3) |
| dec4 | 118.0 | 118.5 | 118.3 | -0.4% | -0.3% | 0.8% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 136.3 | 136.1 | 137.0 | -0.0% | -0.4% | 0.6% | within (paired) | (3, 3, 3) |

mem low H/W GiB: cand (5.62, 7.54) vs L (5.99, 7.18) R (5.88, 7.59); engram delta cand {'engram_lookup_seconds_sum': 109.95, 'engram_lookup_seconds_count': 5004.0} L {'engram_lookup_seconds_sum': 105.3, 'engram_lookup_seconds_count': 5138.0}; graphs destroyed (r0/r1) cand [0, 0]

### b05 B-map vs b04 / b06
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1620.2 | 1489.7 | 1568.7 | +8.8% | +3.3% | 8.0% | within | (3, 3, 3) |
| pd.8000 (primary) | 2015.6 | 1899.7 | 1852.5 | +6.1% | +8.8% | 3.6% | WIN | (3, 3, 3) |
| pd.32000 (primary) | 2083.7 | 1990.5 | 1972.7 | +4.7% | +5.6% | 1.9% | WIN | (3, 3, 3) |
| pd.100000 (primary) | 2034.9 | 1966.5 | 1952.2 | +3.5% | +4.2% | 0.7% | WIN | (2, 2, 2) |
| ttft.8000 (primary) | 4.0 | 4.3 | 4.4 | +5.9% | +8.2% | 6.6% | within | (3, 3, 3) |
| pc.8000 | 1993.3 | 1905.4 | 1905.3 | +4.6% | +4.6% | 3.1% | WIN | (3, 3, 3) |
| pp.8000 | 2270.6 | 2089.3 | 2088.9 | +8.7% | +8.7% | 62.4% | within | (3, 3, 3) |
| dec1 (primary) | 80.7 | 80.6 | 81.0 | -0.1% | -0.3% | 2.0% | within (paired) | (3, 3, 3) |
| dec4 | 115.4 | 118.3 | 117.7 | -2.4% | -1.9% | 3.8% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 137.4 | 137.0 | 136.1 | +0.1% | +0.6% | 0.5% | within (paired) | (3, 3, 3) |

mem low H/W GiB: cand (6.81, 7.86) vs L (5.88, 7.59) R (5.86, 7.92); engram delta cand {'engram_lookup_seconds_sum': 68.23, 'engram_lookup_seconds_count': 4968.0} L {'engram_lookup_seconds_sum': 111.4, 'engram_lookup_seconds_count': 5172.0}; graphs destroyed (r0/r1) cand [0, 0]

### b07 C-rc256 vs b06 / b08
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1504.6 | 1568.7 | 1593.5 | -4.1% | -5.6% | 8.0% | within | (3, 3, 3) |
| pd.8000 (primary) | 1869.5 | 1852.5 | 1892.0 | +0.9% | -1.2% | 3.6% | within | (3, 3, 3) |
| pd.32000 (primary) | 1949.9 | 1972.7 | 1989.2 | -1.2% | -2.0% | 1.9% | within | (3, 3, 3) |
| pd.100000 (primary) | 1633.2 | 1952.2 | 1958.0 | -16.3% | -16.6% | 26.0% | within | (2, 2, 2) |
| ttft.8000 (primary) | 4.3 | 4.4 | 4.3 | +1.7% | +0.5% | 6.6% | within | (3, 3, 3) |
| pc.8000 | 1162.9 | 1905.3 | 1894.9 | -39.0% | -38.6% | 4.0% | LOSS | (3, 3, 3) |
| pp.8000 | 814.2 | 2088.9 | 2084.3 | -61.0% | -60.9% | 20.0% | LOSS | (3, 3, 3) |
| dec1 (primary) | 77.4 | 81.0 | 80.5 | -4.4% | -3.8% | 1.1% | LOSS (paired) | (3, 3, 3) |
| dec4 | 107.1 | 117.7 | 117.9 | -9.1% | -9.3% | 3.6% | LOSS (paired) | (2, 2, 2) |
| dec8 (primary) | 112.3 | 136.1 | 135.5 | -17.5% | -17.1% | 7.2% | LOSS (paired) | (3, 3, 3) |

mem low H/W GiB: cand (5.41, 7.25) vs L (5.86, 7.92) R (6.37, 7.92); engram delta cand {'engram_lookup_seconds_sum': 90.5, 'engram_lookup_seconds_count': 4940.0} L {'engram_lookup_seconds_sum': 103.08, 'engram_lookup_seconds_count': 4968.0}; graphs destroyed (r0/r1) cand [0, 0]

### b09 D-prod vs b08 / b10
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1485.5 | 1593.5 | 1476.7 | -6.8% | +0.6% | 9.7% | within | (3, 3, 3) |
| pd.8000 (primary) | 1948.7 | 1892.0 | 1893.5 | +3.0% | +2.9% | 0.8% | WIN | (3, 3, 3) |
| pd.32000 (primary) | 2038.5 | 1989.2 | 1981.2 | +2.5% | +2.9% | 0.9% | WIN | (3, 3, 3) |
| pd.100000 (primary) | 1980.8 | 1958.0 | 1961.2 | +1.2% | +1.0% | 0.8% | WIN | (2, 2, 2) |
| ttft.8000 (primary) | 4.1 | 4.3 | 4.3 | +4.3% | +2.9% | 1.7% | WIN | (3, 3, 3) |
| pc.8000 | 1474.0 | 1894.9 | 1853.9 | -22.2% | -20.5% | 7.5% | LOSS | (3, 3, 3) |
| pp.8000 | 929.7 | 2084.3 | 2081.5 | -55.4% | -55.3% | 20.2% | LOSS | (3, 3, 3) |
| dec1 (primary) | 77.2 | 80.5 | 80.5 | -4.3% | -4.2% | 1.5% | LOSS (paired) | (3, 3, 3) |
| dec4 | 105.6 | 117.9 | 117.8 | -10.5% | -10.4% | 3.3% | LOSS (paired) | (2, 2, 2) |
| dec8 (primary) | 111.7 | 135.5 | 135.6 | -17.6% | -17.6% | 6.2% | LOSS (paired) | (3, 3, 3) |

mem low H/W GiB: cand (5.32, 7.05) vs L (6.37, 7.92) R (5.62, 7.44); engram delta cand {'engram_lookup_seconds_sum': 95.97, 'engram_lookup_seconds_count': 4934.0} L {'engram_lookup_seconds_sum': 102.37, 'engram_lookup_seconds_count': 4976.0}; graphs destroyed (r0/r1) cand [0, 0]

### b11 E-cap512 vs b10 / b12
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1460.9 | 1476.7 | 1600.9 | -1.1% | -8.7% | 8.3% | within | (3, 3, 3) |
| pd.8000 (primary) | 1887.5 | 1893.5 | 1896.8 | -0.3% | -0.5% | 3.2% | within | (3, 3, 3) |
| pd.32000 (primary) | 1979.5 | 1981.2 | 1986.4 | -0.1% | -0.3% | 1.6% | within | (3, 3, 3) |
| pd.100000 (primary) | 1952.9 | 1961.2 | 1954.5 | -0.4% | -0.1% | 0.4% | within | (2, 2, 2) |
| ttft.8000 (primary) | 4.3 | 4.3 | 4.3 | -0.1% | +1.0% | 6.4% | within | (3, 3, 3) |
| pc.8000 | 1821.6 | 1853.9 | 1857.4 | -1.7% | -1.9% | 4.0% | within | (3, 3, 3) |
| pp.8000 | 2081.8 | 2081.5 | 2085.6 | +0.0% | -0.2% | 5.7% | within | (3, 3, 3) |
| dec1 (primary) | 80.7 | 80.5 | 80.5 | +0.2% | +0.2% | 0.7% | within (paired) | (3, 3, 3) |
| dec4 | 117.9 | 117.8 | 117.8 | +0.1% | +0.1% | 0.3% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 135.8 | 135.6 | 135.1 | +0.2% | +0.5% | 0.6% | within (paired) | (3, 3, 3) |
| lc1 | 70.9 | 76.8 | 73.7 | -7.2% | -3.7% | 14.7% | within (paired) | (2, 2, 2) |
| lc4 | 119.4 | 119.5 | 121.1 | -0.1% | -1.4% | 1.3% | within (paired) | (1, 1, 1) |

mem low H/W GiB: cand (5.76, 7.43) vs L (5.62, 7.44) R (5.69, 7.42); engram delta cand {'engram_lookup_seconds_sum': 105.13, 'engram_lookup_seconds_count': 7488.0} L {'engram_lookup_seconds_sum': 104.24, 'engram_lookup_seconds_count': 7360.0}; graphs destroyed (r0/r1) cand [0, 0]

### b13 E-cap0 vs b12 / b14
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1458.5 | 1600.9 | 1540.2 | -8.9% | -5.3% | 8.3% | within | (3, 3, 3) |
| pd.8000 (primary) | 1887.0 | 1896.8 | 1883.7 | -0.5% | +0.2% | 3.2% | within | (3, 3, 3) |
| pd.32000 (primary) | 1979.9 | 1986.4 | 1980.1 | -0.3% | -0.0% | 0.4% | within | (3, 3, 3) |
| pd.100000 (primary) | 1946.0 | 1954.5 | 1939.2 | -0.4% | +0.4% | 0.8% | within | (2, 2, 2) |
| ttft.8000 (primary) | 4.3 | 4.3 | 4.3 | +0.5% | -0.7% | 6.4% | within | (3, 3, 3) |
| pc.8000 | 1867.3 | 1857.4 | 1882.1 | +0.5% | -0.8% | 4.0% | within | (3, 3, 3) |
| pp.8000 | 2088.8 | 2085.6 | 2080.1 | +0.2% | +0.4% | 5.7% | within | (3, 3, 3) |
| dec1 (primary) | 80.6 | 80.5 | 80.5 | +0.1% | +0.2% | 0.6% | within (paired) | (3, 3, 3) |
| dec4 | 117.7 | 117.8 | 117.7 | -0.0% | +0.0% | 0.1% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 135.7 | 135.1 | 134.8 | +0.5% | +0.4% | 0.5% | within (paired) | (3, 3, 3) |
| lc1 | 71.9 | 73.7 | 74.8 | -2.3% | -4.0% | 14.3% | within (paired) | (2, 2, 2) |
| lc4 | 112.9 | 121.1 | 114.9 | -6.7% | -1.7% | 5.1% | within (paired) | (1, 1, 1) |

mem low H/W GiB: cand (5.52, 7.47) vs L (5.69, 7.42) R (5.73, 7.41); engram delta cand {'engram_lookup_seconds_sum': 105.56, 'engram_lookup_seconds_count': 7538.0} L {'engram_lookup_seconds_sum': 104.2, 'engram_lookup_seconds_count': 7368.0}; graphs destroyed (r0/r1) cand [0, 0]

### b15 F-aio0 vs b14 / b16
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1488.1 | 1540.2 | 1539.1 | -3.4% | -3.3% | 21.4% | within | (3, 3, 3) |
| pd.8000 (primary) | 1959.6 | 1883.7 | 1892.0 | +4.0% | +3.6% | 0.6% | WIN | (3, 3, 3) |
| pd.32000 (primary) | 2016.5 | 1980.1 | 1974.8 | +1.8% | +2.1% | 1.4% | WIN | (3, 3, 3) |
| pd.100000 (primary) | 2005.1 | 1939.2 | 1950.9 | +3.4% | +2.8% | 0.9% | WIN | (2, 2, 2) |
| ttft.8000 (primary) | 4.2 | 4.3 | 4.3 | +2.3% | +2.3% | 2.2% | WIN | (3, 3, 3) |
| pc.8000 | 1936.3 | 1882.1 | 1844.6 | +2.9% | +5.0% | 14.3% | within | (3, 3, 3) |
| pp.8000 | 2132.4 | 2080.1 | 2088.7 | +2.5% | +2.1% | 12.6% | within | (3, 3, 3) |
| dec1 (primary) | 79.5 | 80.5 | 80.6 | -1.3% | -1.4% | 5.9% | within (paired) | (3, 3, 3) |
| dec4 | 116.4 | 117.7 | 117.3 | -1.0% | -0.7% | 1.4% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 136.3 | 134.8 | 135.2 | +0.3% | +0.2% | 1.6% | within (paired) | (3, 3, 3) |

mem low H/W GiB: cand (6.3, 7.63) vs L (5.73, 7.41) R (6.15, 7.95); engram delta cand {'engram_lookup_seconds_sum': 77.98, 'engram_lookup_seconds_count': 4968.0} L {'engram_lookup_seconds_sum': 105.33, 'engram_lookup_seconds_count': 7358.0}; graphs destroyed (r0/r1) cand [0, 0]

### b17 G-fast vs b16 / b18
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1583.8 | 1539.1 | 1491.1 | +2.9% | +6.2% | 10.2% | within | (3, 3, 3) |
| pd.8000 (primary) | 1847.6 | 1892.0 | 1886.4 | -2.3% | -2.1% | 0.9% | LOSS | (3, 3, 3) |
| pd.32000 (primary) | 1980.8 | 1974.8 | 1978.9 | +0.3% | +0.1% | 1.7% | within | (3, 3, 3) |
| pd.100000 (primary) | 1943.5 | 1950.9 | 1958.4 | -0.4% | -0.8% | 0.4% | within | (2, 2, 2) |
| ttft.8000 (primary) | 4.5 | 4.3 | 4.3 | -6.0% | -5.8% | 2.6% | LOSS | (3, 3, 3) |
| pc.8000 | 1895.4 | 1844.6 | 1883.7 | +2.8% | +0.6% | 5.0% | within | (3, 3, 3) |
| pp.8000 | 2087.8 | 2088.7 | 2079.4 | -0.0% | +0.4% | 0.4% | within | (3, 3, 3) |
| dec1 (primary) | 80.6 | 80.6 | 80.7 | +0.3% | -0.1% | 2.3% | within (paired) | (3, 3, 3) |
| dec4 | 117.7 | 117.3 | 117.5 | +0.4% | +0.1% | 0.4% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 136.0 | 135.2 | 135.8 | +0.2% | -0.2% | 0.8% | within (paired) | (3, 3, 3) |

mem low H/W GiB: cand (5.93, 7.84) vs L (6.15, 7.95) R (5.73, 7.9); engram delta cand {'engram_lookup_seconds_sum': 102.39, 'engram_lookup_seconds_count': 4982.0} L {'engram_lookup_seconds_sum': 101.92, 'engram_lookup_seconds_count': 4984.0}; graphs destroyed (r0/r1) cand [0, 0]

### b19 G-dedupe vs b18 / b20
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1492.8 | 1491.1 | 1536.6 | +0.1% | -2.9% | 8.5% | within | (3, 3, 3) |
| pd.8000 (primary) | 1895.9 | 1886.4 | 1842.0 | +0.5% | +2.9% | 3.1% | within | (3, 3, 3) |
| pd.32000 (primary) | 1985.4 | 1978.9 | 1970.5 | +0.3% | +0.8% | 1.7% | within | (3, 3, 3) |
| pd.100000 (primary) | 1957.0 | 1958.4 | 1954.8 | -0.1% | +0.1% | 0.3% | within | (2, 2, 2) |
| ttft.8000 (primary) | 4.3 | 4.3 | 4.4 | -0.6% | +2.2% | 5.3% | within | (3, 3, 3) |
| pc.8000 | 1950.1 | 1883.7 | 1909.6 | +3.5% | +2.1% | 3.2% | within | (3, 3, 3) |
| pp.8000 | 2094.0 | 2079.4 | 2090.4 | +0.7% | +0.2% | 0.9% | within | (3, 3, 3) |
| dec1 (primary) | 80.7 | 80.7 | 80.5 | +0.0% | +0.0% | 0.8% | within (paired) | (3, 3, 3) |
| dec4 | 118.3 | 117.5 | 117.7 | +0.6% | +0.5% | 0.7% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 135.8 | 135.8 | 135.9 | +0.0% | +0.2% | 0.6% | within (paired) | (3, 3, 3) |

mem low H/W GiB: cand (6.29, 7.43) vs L (5.73, 7.9) R (6.15, 7.79); engram delta cand {'engram_lookup_seconds_sum': 97.26, 'engram_lookup_seconds_count': 4980.0} L {'engram_lookup_seconds_sum': 102.06, 'engram_lookup_seconds_count': 4976.0}; graphs destroyed (r0/r1) cand [0, 0]

### b21 H-t25 vs b20 / b22
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1509.6 | 1536.6 | 1567.3 | -1.8% | -3.7% | 7.0% | within | (3, 3, 3) |
| pd.8000 (primary) | 1875.2 | 1842.0 | 1848.7 | +1.8% | +1.4% | 3.6% | within | (3, 3, 3) |
| pd.32000 (primary) | 1969.4 | 1970.5 | 1989.5 | -0.1% | -1.0% | 1.0% | within | (3, 3, 3) |
| pd.100000 (primary) | 1942.6 | 1954.8 | 1955.4 | -0.6% | -0.7% | 0.7% | within | (2, 2, 2) |
| ttft.8000 (primary) | 4.3 | 4.4 | 4.4 | +1.6% | +2.3% | 6.3% | within | (3, 3, 3) |
| pc.8000 | 1862.4 | 1909.6 | 1860.5 | -2.5% | +0.1% | 3.2% | within | (3, 3, 3) |
| pp.8000 | 2072.4 | 2090.4 | 2087.0 | -0.9% | -0.7% | 0.9% | within | (3, 3, 3) |
| dec1 (primary) | 78.8 | 80.5 | 80.7 | -2.1% | -2.3% | 2.5% | within (paired) | (3, 3, 3) |
| dec4 | 109.1 | 117.7 | 117.6 | -7.5% | -7.4% | 9.3% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 121.3 | 135.9 | 135.6 | -11.9% | -11.9% | 3.6% | LOSS (paired) | (3, 3, 3) |
| t1 | 96.7 | 96.7 | 96.7 | -0.0% | -0.1% | 0.3% | within (paired) | (2, 2, 2) |
| t4 | 137.6 | 138.0 | 137.6 | -0.3% | -0.0% | 0.2% | within (paired) | (1, 1, 1) |

mem low H/W GiB: cand (6.11, 7.67) vs L (6.15, 7.79) R (6.14, 7.62); engram delta cand {'engram_lookup_seconds_sum': 103.27, 'engram_lookup_seconds_count': 7670.0} L {'engram_lookup_seconds_sum': 103.15, 'engram_lookup_seconds_count': 6384.0}; graphs destroyed (r0/r1) cand [0, 0]

### c01 D-lead+map vs b24 / c02
| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |
|---|---|---|---|---|---|---|---|---|
| pd.2000 | 1607.3 | 1608.4 | 1587.2 | -0.1% | +1.3% | 9.2% | within | (3, 3, 3) |
| pd.8000 (primary) | 2033.9 | 1902.7 | 1891.9 | +6.9% | +7.5% | 3.9% | WIN | (3, 3, 3) |
| pd.32000 (primary) | 2112.8 | 1992.4 | 1991.0 | +6.0% | +6.1% | 0.3% | WIN | (3, 3, 3) |
| pd.100000 (primary) | 2058.1 | 1972.1 | 1959.1 | +4.4% | +5.1% | 0.7% | WIN | (2, 2, 2) |
| ttft.8000 (primary) | 4.0 | 4.2 | 4.3 | +4.2% | +6.2% | 7.4% | within | (3, 3, 3) |
| pc.8000 | 1949.8 | 1902.9 | 1804.4 | +2.5% | +8.1% | 8.3% | within | (3, 3, 3) |
| pp.8000 | 2180.2 | 2096.6 | 2088.9 | +4.0% | +4.4% | 41.1% | within | (3, 3, 3) |
| dec1 (primary) | 80.7 | 80.6 | 80.5 | -0.1% | -0.2% | 5.7% | within (paired) | (3, 3, 3) |
| dec4 | 118.8 | 117.8 | 118.0 | +0.8% | +0.7% | 0.7% | within (paired) | (2, 2, 2) |
| dec8 (primary) | 137.5 | 135.7 | 136.5 | +1.0% | +0.8% | 0.4% | WIN (paired) | (3, 3, 3) |

mem low H/W GiB: cand (6.79, 8.02) vs L (6.26, 7.8) R (6.36, 7.85); engram delta cand {'engram_lookup_seconds_sum': 71.98, 'engram_lookup_seconds_count': 4972.0} L {'engram_lookup_seconds_sum': 101.88, 'engram_lookup_seconds_count': 4980.0}; graphs destroyed (r0/r1) cand [0, 0]

### i01 I-profile(prod): missing adjacent base (L=c02 R=None)


### Runner status log

```
2026-11-13T17:09:57Z campaign runner start pid=3110506
2026-11-13T17:09:57Z s00-smoke base START flags=(base)
2026-11-13T17:13:30Z s00-smoke restart attempt 1 rc=0 in 214s (idle_before=True waited=0s running_before=None)
2026-11-13T17:13:43Z s00-smoke cell prefill_code: 12s attempts=3 clean=3
2026-11-13T17:14:03Z s00-smoke cell dec1: 21s attempts=3 clean=3
2026-11-13T17:20:13Z s00-smoke cell dec8: 369s attempts=12 clean=0
2026-11-13T17:20:13Z s00-smoke base DONE status=ok clean=6/18 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=6.2/7.9 temp H/W max=80/74 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T17:20:13Z b01 base START flags=(base)
2026-11-13T17:20:52Z runner pid 3110506 stopped by operator-agent (dec8 gauge-lag bug); start.sh for b01 left running; relaunch with ADOPT_FIRST
2026-11-13T17:27:32Z campaign runner start pid=3136310
2026-11-13T17:27:32Z b01 base START flags=(base)
2026-11-13T17:27:33Z b01 ADOPTED the running pair (arm proof matches; restart log runs/overnight/blocks/b01.restart.log)
2026-11-13T17:28:47Z b01 cell warm: 74s attempts=4 clean=4
2026-11-13T17:31:35Z b01 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T17:31:47Z b01 cell prefill_code: 12s attempts=3 clean=3
2026-11-13T17:31:58Z b01 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T17:32:20Z b01 cell dec1: 21s attempts=3 clean=3
2026-11-13T17:32:56Z b01 cell dec4: 37s attempts=2 clean=2
2026-11-13T17:34:29Z b01 cell dec8: 93s attempts=3 clean=3
2026-11-13T17:34:30Z b01 base DONE status=ok clean=29/29 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=6.1/7.9 temp H/W max=85/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T17:34:30Z b02 base START flags=(base)
2026-11-13T17:36:35Z b02 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T17:37:50Z b02 cell warm: 74s attempts=4 clean=4
2026-11-13T17:40:38Z b02 cell prefill_div: 169s attempts=11 clean=11
2026-11-13T17:40:51Z b02 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T17:41:02Z b02 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T17:41:24Z b02 cell dec1: 21s attempts=3 clean=3
2026-11-13T17:42:00Z b02 cell dec4: 36s attempts=2 clean=2
2026-11-13T17:43:33Z b02 cell dec8: 93s attempts=3 clean=3
2026-11-13T17:43:40Z b02 cell k_seed: 7s attempts=0 clean=0
2026-11-13T17:43:41Z b02 base DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.0/7.2 temp H/W max=86/79 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T17:43:41Z b03 A-lead START flags=TF_DS_ENGRAM_LEAD=1
2026-11-13T17:45:46Z b03 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T17:45:48Z b03 cell k_resume: 2s attempts=0 clean=0
2026-11-13T17:47:00Z b03 cell warm: 71s attempts=4 clean=4
2026-11-13T17:49:43Z b03 cell prefill_div: 163s attempts=11 clean=11
2026-11-13T17:49:56Z b03 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T17:50:06Z b03 cell prefill_p125: 10s attempts=3 clean=3
2026-11-13T17:50:27Z b03 cell dec1: 21s attempts=3 clean=3
2026-11-13T17:51:04Z b03 cell dec4: 37s attempts=2 clean=2
2026-11-13T17:52:37Z b03 cell dec8: 93s attempts=3 clean=3
2026-11-13T17:52:37Z b03 A-lead DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.6/7.5 temp H/W max=86/80 lead=True/True rc=[] graphs_destroyed=0/0
2026-11-13T17:52:37Z b04 base START flags=(base)
2026-11-13T17:54:43Z b04 restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T17:55:57Z b04 cell warm: 73s attempts=4 clean=4
2026-11-13T17:58:44Z b04 cell prefill_div: 167s attempts=11 clean=11
2026-11-13T17:58:57Z b04 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T17:59:08Z b04 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T17:59:29Z b04 cell dec1: 21s attempts=3 clean=3
2026-11-13T18:00:05Z b04 cell dec4: 36s attempts=2 clean=2
2026-11-13T18:01:38Z b04 cell dec8: 93s attempts=3 clean=3
2026-11-13T18:02:14Z b04 cell hostov: 36s attempts=11 clean=11
2026-11-13T18:02:14Z b04 base DONE status=ok clean=40/40 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.9/7.6 temp H/W max=87/79 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T18:02:14Z b05 B-map START flags=TF_DS_ENGRAM_SCALES=map
2026-11-13T18:04:19Z b05 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T18:05:42Z b05 cell warm: 82s attempts=4 clean=4
2026-11-13T18:08:22Z b05 cell prefill_div: 161s attempts=11 clean=11
2026-11-13T18:08:36Z b05 cell prefill_code: 14s attempts=3 clean=3
2026-11-13T18:08:51Z b05 cell prefill_p125: 16s attempts=3 clean=3
2026-11-13T18:09:12Z b05 cell dec1: 21s attempts=3 clean=3
2026-11-13T18:09:49Z b05 cell dec4: 37s attempts=2 clean=2
2026-11-13T18:11:22Z b05 cell dec8: 92s attempts=3 clean=3
2026-11-13T18:11:22Z b05 B-map DONE status=ok clean=29/29 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=6.8/7.9 temp H/W max=87/79 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T18:11:22Z b06 base START flags=(base)
2026-11-13T18:13:31Z b06 restart attempt 1 rc=0 in 128s (idle_before=True waited=0s running_before=0.0)
2026-11-13T18:14:45Z b06 cell warm: 74s attempts=4 clean=4
2026-11-13T18:17:34Z b06 cell prefill_div: 170s attempts=11 clean=11
2026-11-13T18:17:47Z b06 cell prefill_code: 12s attempts=3 clean=3
2026-11-13T18:17:58Z b06 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T18:18:19Z b06 cell dec1: 21s attempts=3 clean=3
2026-11-13T18:18:55Z b06 cell dec4: 37s attempts=2 clean=2
2026-11-13T18:20:28Z b06 cell dec8: 93s attempts=3 clean=3
2026-11-13T18:20:29Z b06 base DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.9/7.9 temp H/W max=86/79 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T18:20:29Z b07 C-rc256 START flags=TF_DS_ENGRAM_ROWCACHE_MB=256
2026-11-13T18:22:34Z b07 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T18:23:48Z b07 cell warm: 73s attempts=4 clean=4
2026-11-13T18:26:59Z b07 cell prefill_div: 191s attempts=11 clean=11
2026-11-13T18:27:19Z b07 cell prefill_code: 20s attempts=3 clean=3
2026-11-13T18:27:48Z b07 cell prefill_p125: 28s attempts=3 clean=3
2026-11-13T18:28:10Z b07 cell dec1: 22s attempts=3 clean=3
2026-11-13T18:28:50Z b07 cell dec4: 40s attempts=2 clean=2
2026-11-13T18:30:41Z b07 cell dec8: 111s attempts=3 clean=3
2026-11-13T18:30:41Z b07 C-rc256 DONE status=ok clean=29/29 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=5.4/7.3 temp H/W max=84/77 lead=False/False rc=['256'] graphs_destroyed=0/0
2026-11-13T18:30:41Z b08 base START flags=(base)
2026-11-13T18:32:42Z b08 restart attempt 1 rc=0 in 121s (idle_before=True waited=0s running_before=0.0)
2026-11-13T18:33:55Z b08 cell warm: 73s attempts=4 clean=4
2026-11-13T18:36:43Z b08 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T18:36:56Z b08 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T18:36:59Z schedule cut to a single pass per candidate (user request): alternations a01-a08 removed
2026-11-13T18:37:07Z b08 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T18:37:28Z b08 cell dec1: 21s attempts=3 clean=3
2026-11-13T18:38:05Z b08 cell dec4: 37s attempts=2 clean=2
2026-11-13T18:39:38Z b08 cell dec8: 93s attempts=3 clean=3
2026-11-13T18:39:38Z b08 base DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.4/7.9 temp H/W max=86/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T18:39:38Z b09 D-prod START flags=TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_ROWCACHE_MB=256 TF_DS_ENGRAM_STRICT=1
2026-11-13T18:41:43Z b09 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T18:42:55Z b09 cell warm: 71s attempts=4 clean=4
2026-11-13T18:45:41Z b09 cell prefill_div: 166s attempts=11 clean=11
2026-11-13T18:45:57Z b09 cell prefill_code: 16s attempts=3 clean=3
2026-11-13T18:46:22Z b09 cell prefill_p125: 25s attempts=3 clean=3
2026-11-13T18:46:44Z b09 cell dec1: 22s attempts=3 clean=3
2026-11-13T18:47:25Z b09 cell dec4: 40s attempts=2 clean=2
2026-11-13T18:49:16Z b09 cell dec8: 111s attempts=3 clean=3
2026-11-13T18:49:16Z b09 D-prod DONE status=ok clean=29/29 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=5.3/7.1 temp H/W max=86/80 lead=True/True rc=['256'] graphs_destroyed=0/0
2026-11-13T18:49:16Z b10 base START flags=(base)
2026-11-13T18:51:22Z b10 restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T18:52:36Z b10 cell warm: 73s attempts=4 clean=4
2026-11-13T18:55:24Z b10 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T18:55:37Z b10 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T18:55:48Z b10 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T18:56:09Z b10 cell dec1: 21s attempts=3 clean=3
2026-11-13T18:56:46Z b10 cell dec4: 37s attempts=2 clean=2
2026-11-13T18:58:19Z b10 cell dec8: 93s attempts=3 clean=3
2026-11-13T18:59:26Z b10 cell longctx: 67s attempts=3 clean=3
2026-11-13T18:59:26Z b10 base DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.6/7.4 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T18:59:26Z b11 E-cap512 START flags=GRAPH_CAP=512
2026-11-13T19:01:31Z b11 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T19:02:45Z b11 cell warm: 73s attempts=4 clean=4
2026-11-13T19:05:34Z b11 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T19:05:48Z b11 cell prefill_code: 15s attempts=3 clean=3
2026-11-13T19:05:59Z b11 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T19:06:20Z b11 cell dec1: 21s attempts=3 clean=3
2026-11-13T19:06:57Z b11 cell dec4: 37s attempts=2 clean=2
2026-11-13T19:08:30Z b11 cell dec8: 93s attempts=3 clean=3
2026-11-13T19:09:39Z b11 cell longctx: 69s attempts=3 clean=3
2026-11-13T19:09:39Z b11 E-cap512 DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.8/7.4 temp H/W max=86/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T19:09:39Z b12 base START flags=(base)
2026-11-13T19:11:44Z b12 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T19:12:58Z b12 cell warm: 73s attempts=4 clean=4
2026-11-13T19:15:47Z b12 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T19:15:59Z b12 cell prefill_code: 12s attempts=3 clean=3
2026-11-13T19:16:10Z b12 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T19:16:31Z b12 cell dec1: 21s attempts=3 clean=3
2026-11-13T19:17:08Z b12 cell dec4: 37s attempts=2 clean=2
2026-11-13T19:18:41Z b12 cell dec8: 93s attempts=3 clean=3
2026-11-13T19:19:48Z b12 cell longctx: 67s attempts=3 clean=3
2026-11-13T19:19:49Z b12 base DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.7/7.4 temp H/W max=86/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T19:19:49Z b13 E-cap0 START flags=GRAPH_CAP=0
2026-11-13T19:21:54Z b13 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T19:23:09Z b13 cell warm: 74s attempts=4 clean=4
2026-11-13T19:25:58Z b13 cell prefill_div: 169s attempts=11 clean=11
2026-11-13T19:26:12Z b13 cell prefill_code: 14s attempts=3 clean=3
2026-11-13T19:26:23Z b13 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T19:26:44Z b13 cell dec1: 21s attempts=3 clean=3
2026-11-13T19:27:20Z b13 cell dec4: 37s attempts=2 clean=2
2026-11-13T19:28:53Z b13 cell dec8: 93s attempts=3 clean=3
2026-11-13T19:30:05Z b13 cell longctx: 71s attempts=3 clean=3
2026-11-13T19:30:05Z b13 E-cap0 DONE status=ok clean=32/32 clock_max H/W=2385/2379 valid=True mem_low H/W GiB=5.5/7.5 temp H/W max=87/79 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T19:30:05Z b14 base START flags=(base)
2026-11-13T19:32:11Z b14 restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T19:33:24Z b14 cell warm: 73s attempts=4 clean=4
2026-11-13T19:36:14Z b14 cell prefill_div: 169s attempts=11 clean=11
2026-11-13T19:36:27Z b14 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T19:36:38Z b14 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T19:36:59Z b14 cell dec1: 21s attempts=3 clean=3
2026-11-13T19:37:36Z b14 cell dec4: 37s attempts=2 clean=2
2026-11-13T19:39:09Z b14 cell dec8: 93s attempts=3 clean=3
2026-11-13T19:40:19Z b14 cell longctx: 70s attempts=3 clean=3
2026-11-13T19:40:20Z b14 base DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.7/7.4 temp H/W max=86/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T19:40:20Z b15 F-aio0 START flags=ENGRAM_AIO=0
2026-11-13T19:42:19Z b15 restart attempt 1 rc=0 in 119s (idle_before=True waited=0s running_before=0.0)
2026-11-13T19:43:35Z b15 cell warm: 75s attempts=4 clean=4
2026-11-13T19:46:20Z b15 cell prefill_div: 165s attempts=11 clean=11
2026-11-13T19:46:33Z b15 cell prefill_code: 14s attempts=3 clean=3
2026-11-13T19:46:44Z b15 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T19:47:06Z b15 cell dec1: 22s attempts=3 clean=3
2026-11-13T19:47:42Z b15 cell dec4: 36s attempts=2 clean=2
2026-11-13T19:49:14Z b15 cell dec8: 92s attempts=3 clean=3
2026-11-13T19:49:14Z b15 F-aio0 DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.3/7.6 temp H/W max=87/80 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T19:49:14Z b16 base START flags=(base)
2026-11-13T19:51:21Z b16 restart attempt 1 rc=0 in 127s (idle_before=True waited=0s running_before=0.0)
2026-11-13T19:52:35Z b16 cell warm: 73s attempts=4 clean=4
2026-11-13T19:55:24Z b16 cell prefill_div: 169s attempts=11 clean=11
2026-11-13T19:55:36Z b16 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T19:55:48Z b16 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T19:56:09Z b16 cell dec1: 21s attempts=3 clean=3
2026-11-13T19:56:45Z b16 cell dec4: 37s attempts=2 clean=2
2026-11-13T19:58:19Z b16 cell dec8: 93s attempts=3 clean=3
2026-11-13T19:58:19Z b16 base DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.1/7.9 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T19:58:19Z b17 G-fast START flags=TF_DS_ENGRAM_FAST=1
2026-11-13T20:00:25Z b17 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:01:38Z b17 cell warm: 73s attempts=4 clean=4
2026-11-13T20:04:28Z b17 cell prefill_div: 169s attempts=11 clean=11
2026-11-13T20:04:42Z b17 cell prefill_code: 14s attempts=3 clean=3
2026-11-13T20:04:53Z b17 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T20:05:13Z b17 cell dec1: 21s attempts=3 clean=3
2026-11-13T20:05:50Z b17 cell dec4: 37s attempts=2 clean=2
2026-11-13T20:07:23Z b17 cell dec8: 93s attempts=3 clean=3
2026-11-13T20:07:24Z b17 G-fast DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.9/7.8 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T20:07:24Z b18 base START flags=(base)
2026-11-13T20:09:30Z b18 restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:10:44Z b18 cell warm: 74s attempts=4 clean=4
2026-11-13T20:13:33Z b18 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T20:13:45Z b18 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T20:13:56Z b18 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T20:14:17Z b18 cell dec1: 21s attempts=3 clean=3
2026-11-13T20:14:54Z b18 cell dec4: 37s attempts=2 clean=2
2026-11-13T20:16:27Z b18 cell dec8: 93s attempts=3 clean=3
2026-11-13T20:16:28Z b18 base DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=5.7/7.9 temp H/W max=87/79 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T20:16:28Z b19 G-dedupe START flags=TF_DS_ENGRAM_DEDUPE=1
2026-11-13T20:18:33Z b19 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:19:47Z b19 cell warm: 73s attempts=4 clean=4
2026-11-13T20:22:35Z b19 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T20:22:48Z b19 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T20:22:59Z b19 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T20:23:20Z b19 cell dec1: 21s attempts=3 clean=3
2026-11-13T20:23:57Z b19 cell dec4: 37s attempts=2 clean=2
2026-11-13T20:25:30Z b19 cell dec8: 93s attempts=3 clean=3
2026-11-13T20:25:31Z b19 G-dedupe DONE status=ok clean=29/29 clock_max H/W=2385/2379 valid=True mem_low H/W GiB=6.3/7.4 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T20:25:31Z b20 base START flags=(base)
2026-11-13T20:27:36Z b20 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:28:49Z b20 cell warm: 72s attempts=4 clean=4
2026-11-13T20:31:38Z b20 cell prefill_div: 169s attempts=11 clean=11
2026-11-13T20:31:52Z b20 cell prefill_code: 14s attempts=3 clean=3
2026-11-13T20:32:03Z b20 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T20:32:24Z b20 cell dec1: 21s attempts=3 clean=3
2026-11-13T20:33:00Z b20 cell dec4: 37s attempts=2 clean=2
2026-11-13T20:34:34Z b20 cell dec8: 93s attempts=3 clean=3
2026-11-13T20:35:15Z b20 cell think: 41s attempts=3 clean=3
2026-11-13T20:35:16Z b20 base DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.2/7.8 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T20:35:16Z b21 H-t25 START flags=TF_DS_T25_DRAFT_CAL=0.05
2026-11-13T20:37:21Z b21 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:38:35Z b21 cell warm: 73s attempts=4 clean=4
2026-11-13T20:41:23Z b21 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T20:41:37Z b21 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T20:41:48Z b21 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T20:42:09Z b21 cell dec1: 21s attempts=3 clean=3
2026-11-13T20:42:49Z b21 cell dec4: 40s attempts=2 clean=2
2026-11-13T20:44:33Z b21 cell dec8: 104s attempts=3 clean=3
2026-11-13T20:45:14Z b21 cell think: 41s attempts=3 clean=3
2026-11-13T20:45:14Z b21 H-t25 DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.1/7.7 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T20:45:14Z b22 base START flags=(base)
2026-11-13T20:47:20Z b22 restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:48:34Z b22 cell warm: 73s attempts=4 clean=4
2026-11-13T20:51:22Z b22 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T20:51:35Z b22 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T20:51:46Z b22 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T20:52:07Z b22 cell dec1: 21s attempts=3 clean=3
2026-11-13T20:52:44Z b22 cell dec4: 37s attempts=2 clean=2
2026-11-13T20:54:17Z b22 cell dec8: 93s attempts=3 clean=3
2026-11-13T20:54:57Z b22 cell think: 41s attempts=3 clean=3
2026-11-13T20:54:58Z b22 base DONE status=ok clean=32/32 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.1/7.6 temp H/W max=87/78 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T20:54:58Z b23 C-rc64 START flags=TF_DS_ENGRAM_ROWCACHE_MB=64
2026-11-13T20:57:03Z b23 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T20:59:50Z b23 cell warm: 166s attempts=4 clean=4
2026-11-13T21:00:17Z b23 cell prefill_div BAILED: server down
2026-11-13T21:00:17Z b23 C-rc64 DONE status=broken clean=4/4 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=6.0/8.0 temp H/W max=77/69 lead=False/False rc=['64'] graphs_destroyed=0/0
2026-11-13T21:00:17Z b24 base START flags=(base)
2026-11-13T21:03:50Z b24 restart attempt 1 rc=0 in 213s (idle_before=True waited=0s running_before=None)
2026-11-13T21:05:04Z b24 cell warm: 73s attempts=4 clean=4
2026-11-13T21:07:51Z b24 cell prefill_div: 167s attempts=11 clean=11
2026-11-13T21:08:04Z b24 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T21:08:16Z b24 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T21:08:36Z b24 cell dec1: 21s attempts=3 clean=3
2026-11-13T21:09:13Z b24 cell dec4: 37s attempts=2 clean=2
2026-11-13T21:10:46Z b24 cell dec8: 93s attempts=3 clean=3
2026-11-13T21:10:47Z b24 base DONE status=ok clean=29/29 clock_max H/W=2385/2379 valid=True mem_low H/W GiB=6.3/7.8 temp H/W max=85/76 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T21:10:47Z c01 D-lead+map START flags=TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_SCALES=map
2026-11-13T21:12:52Z c01 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T21:14:10Z c01 cell warm: 78s attempts=4 clean=4
2026-11-13T21:16:49Z c01 cell prefill_div: 159s attempts=11 clean=11
2026-11-13T21:17:02Z c01 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T21:17:14Z c01 cell prefill_p125: 12s attempts=3 clean=3
2026-11-13T21:17:36Z c01 cell dec1: 21s attempts=3 clean=3
2026-11-13T21:18:12Z c01 cell dec4: 36s attempts=2 clean=2
2026-11-13T21:19:44Z c01 cell dec8: 92s attempts=3 clean=3
2026-11-13T21:19:45Z c01 D-lead+map DONE status=ok clean=29/29 clock_max H/W=2392/2385 valid=True mem_low H/W GiB=6.8/8.0 temp H/W max=87/80 lead=True/True rc=[] graphs_destroyed=0/0
2026-11-13T21:19:45Z c02 base START flags=(base)
2026-11-13T21:21:54Z c02 restart attempt 1 rc=0 in 129s (idle_before=True waited=0s running_before=0.0)
2026-11-13T21:23:07Z c02 cell warm: 73s attempts=4 clean=4
2026-11-13T21:25:56Z c02 cell prefill_div: 168s attempts=11 clean=11
2026-11-13T21:26:09Z c02 cell prefill_code: 13s attempts=3 clean=3
2026-11-13T21:26:20Z c02 cell prefill_p125: 11s attempts=3 clean=3
2026-11-13T21:26:41Z c02 cell dec1: 21s attempts=3 clean=3
2026-11-13T21:27:17Z c02 cell dec4: 36s attempts=2 clean=2
2026-11-13T21:28:50Z c02 cell dec8: 93s attempts=3 clean=3
2026-11-13T21:28:50Z c02 base DONE status=ok clean=29/29 clock_max H/W=2385/2385 valid=True mem_low H/W GiB=6.4/7.9 temp H/W max=87/77 lead=False/False rc=[] graphs_destroyed=0/0
2026-11-13T21:28:50Z i01 I-profile(prod) START flags=TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_ROWCACHE_MB=256 TF_DS_ENGRAM_STRICT=1 RANK1_PROFILE=1
2026-11-13T21:30:56Z i01 restart attempt 1 rc=0 in 125s (idle_before=True waited=0s running_before=0.0)
2026-11-13T21:32:08Z i01 cell warm: 72s attempts=4 clean=4
2026-11-13T21:32:29Z i01 cell dec1: 21s attempts=3 clean=3
2026-11-13T21:34:00Z i01 cell dec8: 91s attempts=3 clean=3
2026-11-13T21:34:01Z i01 I-profile(prod) DONE status=ok clean=10/10 clock_max H/W=2385/2379 valid=True mem_low H/W GiB=5.4/7.4 temp H/W max=81/74 lead=True/True rc=['256'] graphs_destroyed=0/0
2026-11-13T21:34:01Z schedule exhausted
2026-11-13T21:34:01Z RESTORE production: TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_ROWCACHE_MB=256 TF_DS_ENGRAM_STRICT=1 (+ head .env)
2026-11-13T21:36:07Z restore restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T21:36:08Z RESTORE done ok=False env_match=False diff0={'missing': ['TF_DS_CACHE_METRICS=1'], 'extra': []} diff1={'missing': ['TF_DS_CACHE_METRICS=1'], 'extra': []} lead=True/True rc=['256']/['256'] chat={'error': None, 'prompt_tokens': 8, 'completion_tokens': 2, 'finish': 'stop'}
2026-11-13T21:36:08Z CAMPAIGN DONE
2026-11-13T21:37:17Z RESTORE production: {'TF_DS_ENGRAM_LEAD': '1', 'TF_DS_ENGRAM_ROWCACHE_MB': '256', 'TF_DS_ENGRAM_STRICT': '1', 'TF_DS_CACHE_METRICS': '1'} (+ head .env)
2026-11-13T21:39:22Z restore restart attempt 1 rc=0 in 126s (idle_before=True waited=0s running_before=0.0)
2026-11-13T21:39:23Z RESTORE done ok=True env_match=True diff0={'missing': [], 'extra': []} diff1={'missing': [], 'extra': []} lead=True/True rc=['256']/['256'] chat={'error': None, 'prompt_tokens': 8, 'completion_tokens': 2, 'finish': 'stop'}
```
<!-- TABLES END -->
