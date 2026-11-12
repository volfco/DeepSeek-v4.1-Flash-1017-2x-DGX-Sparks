# To benchmark: Engram A/B and draft depth

Forward-looking checklist for the Engram switches and the draft-depth experiments. Nothing here is claimed as measured:
the Engram patches (`zig/patches/0094`-`0098`) and the depth toggles (`zig/patches/0103`-`0106`) are host-tested only,
and the knobs in the "Existing engine knobs" and "Proposed" tables are either existing settings or not implemented yet. Run them, record the
numbers, and only then decide what to keep.

**Rule for every row: one switch at a time, everything else identical** (image patch hash, `CONTEXT`, `POOL`,
`PARALLEL`, `ROUND_ROWS`, `GRAPHS`, `TF_FILL_*`, `SPILL_GIB`, `KV_SHARD`, the prompt salt policy). Two runs that
differ in more than the switch are not an A/B.

## Before the first run

- The image must be built from this recipe's patch series (`./start.sh` rebuilds it; the tag changes with
  `zig/patches/`). Both Sparks need the Engram tables on local disk (`HEAD_ENGRAM_DIR` on the head,
  `WORKER_ENGRAM_DIR` on the worker): over NFS a random 4 KiB read was ~940 µs against ~90 µs local.
- `TF_DS_ENGRAM_METRICS` is on by default (1): `tensorfold:engram_lookup_seconds` is the read-latency histogram, with
  `_count` (reads) and `_sum` (seconds). It is the only per-read signal; use it for every Engram row below.
- Identity harness: `tools/exact.py` (greedy, 39 prompts through 100k tokens, resumes and 8 concurrent) must report
  IDENTICAL for each switch before its timing counts. `tools/validate.py` is the quicker smoke (text, thinking, tool
  call, image, 8 concurrent).
- Timing: `tools/bench.py` on **natural prompts** with a fresh `--salt` per run, medians of 3 alternating runs, and
  the draft acceptance rate recorded next to the speed (`docs/measurements.md`: filler prompts moved 30-60 % where
  natural text moved ~0 %). Treat differences under ~2 % prefill and ~3 % decode as noise.
- Record the head's and worker's lowest `MemAvailable` for every run: `SCALES=map`, a warm mapping and a row cache
  all trade memory or page cache for speed.

## Implemented switches (`zig/patches/0094`-`0098`)

All off by default; every one is bit-identical, so `exact.py` must pass on each.

| Run | Flags (export; `start.sh` gives them to both ranks) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Trim | `TF_DS_ENGRAM_TRIM=1` (mounts `/engram` rw; a hardlinked table is skipped) | `du -B1 -s $HEAD_ENGRAM_DIR` and `$WORKER_ENGRAM_DIR` before/after a start; `find -printf '%p %s\n'` unchanged; `exact.py`; prefill/decode unchanged | ~101 GB freed per Spark that owns a real copy, start still passes `port_engram_check`, replies identical. No speed change expected |
| Scales mapped | `TF_DS_ENGRAM_SCALES=map` | `engram_lookup_seconds` count and sum per prompt (bench.py `--metrics`), prefill 8K/32K, decode 1 stream | Count and sum fall; prefill or decode does not regress. If it regresses, the page-fault path is the cost (see `map-warm` below) |
| Dedupe | `TF_DS_ENGRAM_DEDUPE=1` | `engram_lookup_seconds_count` per prompt, prefill on prose *and* on a prompt with a repeated paragraph (the ceiling), decode | Count drops by the repeat rate with identical replies; prefill improves. The repeated-paragraph run shows the maximum the switch can buy |
| Fast | `TF_DS_ENGRAM_FAST=1`; optionally `TF_DS_ENGRAM_AIO_SLOTS=8192` | prefill 2K/8K/32K, decode; with `RANK1_PROFILE=1` the `prefill_engram_ms` split (rows/upload/mm/exchange/gate) | Any gain above noise. Expected under 1 % each; the switch exists to measure, not to claim |
| Strict | `TF_DS_ENGRAM_STRICT=1` | the pair starts, `exact.py` identical, no new log refusals | It must be a no-op on this checkpoint; a refusal means the tables are not what the decode assumes |

Combinations worth one run each (same identity rule): `SCALES=map`+`DEDUPE=1`, `DEDUPE=1`+`LEAD=1`, `FAST=1`+`DEDUPE=1`,
`AIO=0`+`SCALES=map`.

## Existing engine knobs (no patch; put these in the matrix)

| Run | Flags | Measure | Why |
| --- | --- | --- | --- |
| Lead readers | `TF_DS_ENGRAM_LEAD=1` (needs `ENGRAM_AIO=1`, the default) | prefill 2K/8K/32K, `engram_lookup_seconds`, `exact.py` | Reads each layer a chunk ahead on a ring of its own; never measured on this recipe. The cleanest test of whether overlap is the limit, and it shares the dedupe path |
| Buffered reads | `ENGRAM_AIO=0` and `ENGRAM_AIO=0 TF_DS_ENGRAM_SCALES=map` | prefill 2K/8K/32K, decode | O_DIRECT bypasses the page cache; repeated 2-gram rows in decode could be served from cache. Once measured at ~0.5 s against ~0.07 s per chunk — retest only with the scales mapped |
| Metric overhead | `TF_DS_ENGRAM_METRICS=0` | prefill/decode against the default | Control: two clock reads per gather, not per row. Confirm it is noise before trusting histogram deltas |

## Proposed (not implemented yet; needs `0099`+)

| Run | Flag | What it would tell you |
| --- | --- | --- |
| Warm scales | `TF_DS_ENGRAM_SCALES=map-warm` (`madvise(MADV_WILLNEED)` on the mapping) | Whether `map`'s page-fault path is the cost: warmed, every scale row is a resident load instead of a fault-in. This decides whether `map` is a win or a wash |
| Decode threads | `TF_DS_ENGRAM_DECODE_THREADS=4` / `5` / `8` | The container has 5 fast cores; 8 decode threads oversubscribe and steal from the engine's threads |
| Pool threads | `TF_DS_ENGRAM_POOL=8` / `16` / `64` (with `AIO=0`) | The pool's size when the ring is off |
| CPU placement | `TF_DS_ENGRAM_CPUS=10-14` with `CPUSET=10-19` | Moves hashing, decoding and AIO reaping off the X925 cores the engine uses; the only CPU-contention experiment |
| Row cache | `TF_DS_ENGRAM_ROWCACHE_MB=0` / `64` / `256` | Repeats across chunks and rounds (0096 only covers one chunk). Watch `MemAvailable` |
| Read depth | `TF_DS_ENGRAM_SPLIT=2` (two rings) or `TF_DS_ENGRAM_AIO_BATCH` | Whether 0.07 s per chunk is the drive or the single reaping thread |
| Host timing | `engram_host_ms` on the `done` line or in `EngramStats` | Makes the hash/decode items of `FAST` visible without `--profile`'s stream synchronization |

## Commands (Engram arms)

```bash
# identity first, per switch (the record must reuse the base salt and shapes)
python3 -I tools/exact.py record --out runs/exact-base.json --label base
TF_DS_ENGRAM_FAST=1 ./start.sh restart
python3 -I tools/exact.py record --out runs/exact-fast.json --label fast --like runs/exact-base.json
python3 -I tools/exact.py compare runs/exact-base.json runs/exact-fast.json

# timing, alternating with the base run, fresh salt per run (so the prefix cache does not hide the Engram work)
python3 -I tools/bench.py --suite prefill --prefill-sizes 2000,8000,32000 --reps 3 --metrics --label base --out runs/engram-base.json
python3 -I tools/bench.py --suite decode,multi --streams 1,4 --reps 3 --label base-dec --out runs/engram-base-dec.json
python3 -I tools/bench.py --compare runs/engram-base.json runs/engram-fast.json

# the read histogram (rank 0 serves /metrics on this deployment's API port, PORT=8000 in .env — export PORT=8000
# or pass --url, or bench.py/exact.py fall back to 8888), before and after one known prompt
curl -s http://127.0.0.1:8000/metrics | grep 'tensorfold:engram_lookup_seconds'

# trim: blocks, not apparent size; the listing must not change
du -B1 -s "$WORKER_ENGRAM_DIR"; stat -c '%s %n' "$WORKER_ENGRAM_DIR"/*
find "$WORKER_ENGRAM_DIR" -maxdepth 1 -type f -printf '%f %s\n' | sort
```

## Draft depth and calibration

Two questions: whether the round loop's depth choice fits this traffic, and whether the depth under test is the depth
that runs. The toggles (`0103`-`0106`) pin every head draft, so a curve is a constant rather than a rule output;
`TF_DS_T25_DRAFT_CAL` is the existing per-stream calibration (off in `.env`).

Baseline on this pair (300 recent agent requests, thinking on, `PARALLEL=8`, from the `0070` done lines): acceptance
median 88.1% (p10 82.5, min 59), 2.35 tokens/round median (p90 3.4, max 4.5), 60 ms/token per stream, TTFT median 36 s
against 2.1 s prefill. The same fields are on every done line (`accept`, `tokens_per_round`, `s_per_token`, `rounds`),
so an arm is scored from the log, not from a client average.

| Run | Flags | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Calibration | `TF_DS_T25_DRAFT_CAL=0.05`, then `0.2` | accept, tokens/round, s_per_token, decode tok/s at 1/4/8 streams, natural prompts, the quality set | more tokens/round and faster decode at the same acceptance. Author's line: greedy prose 79.7 → 85.4 tok/s, code unchanged. Not bit-identical when set: a sampled stream's draws change |
| Depth 2 | `TF_DS_DRAFT_DEPTH2=1` | tokens/round, accept, s_per_token, rows/round, `MemAvailable` low | a depth whose tokens/round per second beats the unpinned run |
| Depth 3 | `TF_DS_DRAFT_DEPTH3=1` | as depth 2 | as depth 2 |
| Depth 4 | `TF_DS_DRAFT_DEPTH4=1` | as depth 2 | as depth 2 |
| Depth 5 | `TF_DS_DRAFT_DEPTH5=1` | as depth 2 | as depth 2 |
| Pinned + calibration | `TF_DS_DRAFT_DEPTH3=1 TF_DS_T25_DRAFT_CAL=0.05` | tokens/round must equal the pinned run's | control: a pin bypasses the allocator, so T25 cannot move it; a difference means the pin is not hard |
| Copies off | `TF_DS_NO_COPIES=1` alone, then with a pin | how many rounds are copy rounds (1 row) and how many are head rounds | a depth curve read without copy rounds confounding it |
| Fix-K control | `TF_DS_FIX_K=3` against `TF_DS_DRAFT_DEPTH3=1` | tokens/round, accept | shows the older knob feeds chances (the allocator can still grant fewer) while the pin is hard; keep one |

Rules for these arms:

- At most one depth toggle may be on; two refuse the start (`error.BadDraftPin`). The checkpoint's draft count and the
  stream's room clamp a pin, so a 5-deep toggle on a checkpoint with fewer draft blocks drafts fewer.
- Greedy replies keep their bits under a pin (same accept/reject path), so `tools/exact.py` is still the identity gate;
  it does not exercise sampled draws, so score sampled arms on the quality set and `accept` as well.
- A pin changes the round's rows; `TF_DS_PLAIN_ROWS` keeps the mixed pack's cuBLAS shapes fixed, so leave it at its
  default. Copies draft 1 row and are not governed by a pin: set `TF_DS_NO_COPIES=1` when a curve must be head drafts
  only, and report which arm had it.
- `TF_DS_T25_CAL_LOG=n` prints every n-th calibrated verify (diagnostic only).

```bash
# one arm per restart; the log is the evidence (rank 0's done lines)
TF_DS_DRAFT_DEPTH3=1 ./start.sh restart
docker logs dsv41-1017-exl3-tf 2>&1 | grep 'done ' | tail -300 > runs/done-depth3.log
# mean acceptance and tokens/round of the arm
sed -n 's/.*accept=\([0-9.]*\)%.*tokens_per_round=\([0-9.]*\).*/\1 \2/p' runs/done-depth3.log |
  awk '{a+=$1; t+=$2; n++} END {print n, a/n, t/n}'
```

## Results

Measured 2026-11-12 on the pair (image `exl3-tp2-6d345f841886`, all rows one switch off/on against it). Protocol
(signal grade, operator's call; `tools/bench_matrix.sh`): one bench block per row (prefill 2K/8K/32K + decode/multi
1,4, reps 3, fresh salt), identity gate `tools/validate.py --save/--compare` against the base save ("ident" below;
per-switch `exact.py` records are deferred), quality set not scored (undefined in this repo), TRIM skipped
(destructive). Cross-image gate first: `exact.py` old vs new image, all switches off: **IDENTICAL, 39/39 cases**.
Noise band, from the `metrics0` control and the `base-final` drift bookend (2088.9, -1.2% after 1.5 h): prefill
about +-4%, decode about +-3%. Engram rows are pass 2 (21:39-22:27Z); pass 1 (20:04-21:33Z, same arms) is quoted
in the notes where it changes a read. counts/sums are the engine histogram delta over the bench block (bench.py's
own snapshots truncate before the series; see `scripts/bench_matrix.sh`).

### Engram

| Run | Flags | Prefill 8K tok/s | Decode tok/s | engram reads (count) | engram read sum (s) | exact | MemAvailable low (H/W GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | - | 2097.5 | 61.2 | 2608 | 23.7 | - | 6.1/7.8 | pass 1: 2113.9/63.5. `base-final` bookend 2088.9 (-1.2%) |
| trim | `TF_DS_ENGRAM_TRIM=1` | not run | | | | | | skipped (operator): punches ~101 GB per Spark irreversibly (nlink=1 both); deferred to todo.md |
| scales=map | `TF_DS_ENGRAM_SCALES=map` | 1981.1 (-5%) | 62.8 | 2662 | 20.5 | ident | 6.1/7.8 | -11% on pass 1 (cold mapping), -5% warm: the page-fault path is the cost. count unchanged, read time -13%. The proposed `map-warm` run decides keep/drop |
| dedupe | `TF_DS_ENGRAM_DEDUPE=1` | 2083.2 | 62.5 | 2668 | 22.2 | ident | 6.0/7.3 | no count reduction anywhere: same-prompt cold probe is 114 = 114 (see below). Neutral: not worth keeping as measured |
| fast | `TF_DS_ENGRAM_FAST=1` | 2072.5 | 64.3 | 2638 | 23.5 | ident | 6.0/7.7 | neutral, as the doc expects (measures, does not claim) |
| strict | `TF_DS_ENGRAM_STRICT=1` | 2078.9 | 63.9 | 2586 | 23.7 | ident | 6.1/7.8 | pair starts, no new log refusals: no-op as required |
| lead | `TF_DS_ENGRAM_LEAD=1` | 2227.9 (+6%) | 63.0 | 2676 | 26.3 | ident | 6.0/7.7 | **+5.0% (pass 1) and +6.2% (pass 2) prefill: the one reproducible win.** read sum up (extra read-ahead) but hidden |
| aio=0 | `ENGRAM_AIO=0` | 2121.3 | 61.9 | 2604 | 18.4 | ident | 6.0/7.7 | per-read time -22% (pass 1: -45%) but prefill unchanged: the AIO ring already hides the reads |
| metrics=0 | `TF_DS_ENGRAM_METRICS=0` | 2075.0 | 64.8 | n/a | n/a | ident | 6.1/7.7 | control, in band. histogram series absent -> flag passthrough proven end to end |

### Combinations (one run each)

| Run | Flags | Prefill 8K tok/s | Decode tok/s | engram reads (count) | engram read sum (s) | exact | MemAvailable low (H/W GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| map+dedupe | `SCALES=map DEDUPE=1` | 1711.8 (-18%) | 62.1 | 2682 | 21.4 | ident | 6.1/7.7 | pass 1: -25%. map's cost composes; dedupe adds nothing |
| dedupe+lead | `DEDUPE=1 LEAD=1` | (2101.5 prefill only) | crashed | 19 | - | ident | 6.0/7.8 | **rank pair broke in decode, both passes** (`NCCL communicator aborted`, rank 1 closed): reproducible engine bug, do not ship the combo |
| fast+dedupe | `FAST=1 DEDUPE=1` | 2091.4 | 62.8 | 2670 | 22.2 | ident | 6.1/7.6 | neutral, composes cleanly |
| aio=0+map | `AIO=0 SCALES=map` | 1163.4 (-44%) | 64.9 | 2582 | 15.4 | ident | 6.1/7.4 | pass 1: -35%. worst row of the matrix: buffered reads + mapped scales are anti-synergistic |

### Dedupe same-prompt probe

Natural prose and the repeated-paragraph ceiling (the doc's measure), identical prompts in every leg, prompt
cache flushed between legs and `cached_tokens` 0 throughout (`scripts/dedupe_probe.sh`; one prefill block 8K/32K
x3 per leg, one 2431-token repeated-paragraph request). Counts are deterministic on identical prompts (base1 ==
base2), so the delta is exact:

| Leg | prose block reads | repeated-paragraph reads |
| --- | --- | --- |
| base (cold) | 114 | 10 |
| base repeat (cold) | 114 | 10 |
| dedupe (cold) | 114 | 10 |

**Zero change on either probe.** The doc's "count drops by the repeat rate" does not materialize even on the
maximum-repetition paragraph: either the switch's rows-within-gather saving is invisible to
`engram_lookup_seconds_count` (which may count gather events) or the dedupe path does not engage on this
checkpoint. With the timing rows also neutral, dedupe signs **no measurable effect**; keeping it would need a
rows-level metric showing the saving. (An earlier probe attempt showed 114 -> 12 - that was the prompt cache's
disk tier, not the switch; the flushed design above supersedes it.)

### Draft depth

accept/tokens_per_round/s_per_token are done-line means over the bench window (median accept in the notes where
it differs); quality set not scored (operator). Pins break **concurrent** identity (`concurrent == solo` fails in
validate.py) while keeping sequential identity ("equal to the saved replies" passes) - the doc's "greedy replies
keep their bits" holds per request only. Traffic note: bench prose drafts less than the doc's agent baseline
(69.9% / 2.00 here vs 88.1% / 2.35 there); rows compare within this session.

| Run | Flags | accept % | tokens/round | s_per_token | decode tok/s | quality set | MemAvailable low (H/W GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | - | 69.9 | 2.00 | 0.0239 | 68.2 | not run | 6.1/7.8 | multi-4 132.7 |
| T25 0.05 | `TF_DS_T25_DRAFT_CAL=0.05` | 68.7 | 1.93 | 0.0241 | 66.6 | not run | 6.1/7.6 | tokens/round -3.5%: the author's "more tokens/round at the same acceptance" does not reproduce here |
| T25 0.2 | `TF_DS_T25_DRAFT_CAL=0.2` | 69.2 | 1.91 | 0.0237 | 65.9 | not run | 6.1/7.6 | same direction: neutral to negative |
| depth 2 | `TF_DS_DRAFT_DEPTH2=1` | 49.4 | 1.98 | 0.0250 | 62.0 | not run | 6.1/7.8 | accept -20 pp for no speed: net negative |
| depth 3 | `TF_DS_DRAFT_DEPTH3=1` | 37.7 | 2.13 | 0.0255 | 62.7 | not run | 6.0/7.9 | multi-4 112.5 (4 stream flakes) |
| depth 4 | `TF_DS_DRAFT_DEPTH4=1` | 29.1 | 2.17 | 0.0176* | 56.8 | not run | 6.2/7.9 | *survivor bias (n=6 done lines); all 12 multi-4 requests errored |
| depth 5 | `TF_DS_DRAFT_DEPTH5=1` | 25.7 | 2.27 | 0.0179* | 56.4 | not run | 6.2/7.8 | as depth 4; multi-4 dead |
| depth 3 + T25 | `DRAFT_DEPTH3=1 T25_DRAFT_CAL=0.05` | 37.7 | 2.14 | 0.0267 | 61.5 | not run | 5.9/7.8 | control passes: tokens/round and accept equal depth 3's - a pin is hard, T25 cannot move it |
| copies off | `TF_DS_NO_COPIES=1` | 67.2 | 1.94 | 0.0244 | 65.3 | not run | 6.1/7.8 | identity kept (gate ident); mildly negative - copy rounds are cheap |
| copies off + depth 3 | `NO_COPIES=1 DRAFT_DEPTH3=1` | 38.3 | 2.13 | 0.0266 | 64.3 | not run | 6.1/7.8 | equals depth 3: copy rounds do not confound the depth curve |
| fix-k 3 | `TF_DS_FIX_K=3` | 38.7 | 2.14 | 0.0274 | 61.9 | not run | 5.8/7.8 | depth-3's numbers with identity kept (gate ident, concurrent too): if a fixed-K curve is wanted, keep FIX_K over the pins |

Cross-checks from the lab's OpenObserve store (`tools/obs_arm_metrics.py`, per-arm windows of both passes):
`TF_DS_ENGRAM_METRICS=0` zeroes the histogram series; `aio0` halves read seconds; MTP accepted/drafted collapses
to 45-55% under pins (matching the done lines). Its 60 s scrape cadence is too coarse for the count columns
above (the runner's own histogram scrape is the source) and `pair_broken` cannot see the dedupe+lead crash (the
dying container is never scraped); the engine log is the evidence there.

## Open questions this should answer

- Is the read-ahead hiding the Engram I/O completely, or is the drive/CPU contention visible in prefill?
- Does the scale half really cost a page per row, and does warming remove it?
- How much do repeated n-grams buy on real prose and on code, and is a cross-chunk cache worth its memory?
- Are 8 decode threads on 5 cores a loss, and does moving the host work off the engine's cores help?
- Does `LEAD` beat the plain read-ahead on this recipe, and does it compose with `DEDUPE`/`FAST`?
