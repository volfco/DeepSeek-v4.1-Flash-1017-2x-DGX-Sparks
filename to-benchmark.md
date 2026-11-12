# Engram A/B: what to benchmark

Forward-looking checklist for the Engram switches and experiments. Nothing here is claimed as measured: the five
patches (`zig/patches/0094`-`0098`) are host-tested only, and the knobs in the last two tables are either existing
engine settings or not implemented yet. Run them, record the numbers, and only then decide what to keep.

**Rule for every row: one switch at a time, everything else identical** (image patch hash, `CONTEXT`, `POOL`,
`PARALLEL`, `ROUND_ROWS`, `GRAPHS`, `TF_FILL_*`, `SPILL_GIB`, `KV_SHARD`, the prompt salt policy). Two runs that
differ in more than the switch are not an A/B.

## Before the first run

- The image must be built from this recipe's patch series (`./start.sh` rebuilds it; the tag changes with
  `zig/patches/`). Both Sparks need the Engram tables on local disk (`HEAD_ENGRAM_DIR` on the head,
  `WORKER_ENGRAM_DIR` on the worker): over NFS a random 4 KiB read was ~940 µs against ~90 µs local.
- `TF_DS_ENGRAM_METRICS` is on by default (1): `tensorfold:engram_lookup_seconds` is the read-latency histogram, with
  `_count` (reads) and `_sum` (seconds). It is the only per-read signal; use it for every row below.
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

## Commands

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

# the read histogram (rank 0 serves /metrics), before and after one known prompt
curl -s http://127.0.0.1:8888/metrics | grep 'tensorfold:engram_lookup_seconds'

# trim: blocks, not apparent size; the listing must not change
du -B1 -s "$WORKER_ENGRAM_DIR"; stat -c '%s %n' "$WORKER_ENGRAM_DIR"/*
find "$WORKER_ENGRAM_DIR" -maxdepth 1 -type f -printf '%f %s\n' | sort
```

## Results (fill in)

| Run | Flags | Prefill 8K tok/s | Decode tok/s | engram reads (count) | engram read sum (s) | exact | MemAvailable low | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | — | | | | | — | | |
| trim | | | | | | | | |
| scales=map | | | | | | | | |
| dedupe | | | | | | | | |
| fast | | | | | | | | |
| strict | | | | | | | | |
| lead | | | | | | | | |
| aio=0 | | | | | | | | |
| metrics=0 | | | | | | | | |

## Open questions this should answer

- Is the read-ahead hiding the Engram I/O completely, or is the drive/CPU contention visible in prefill?
- Does the scale half really cost a page per row, and does warming remove it?
- How much do repeated n-grams buy on real prose and on code, and is a cross-chunk cache worth its memory?
- Are 8 decode threads on 5 cores a loss, and does moving the host work off the engine's cores help?
- Does `LEAD` beat the plain read-ahead on this recipe, and does it compose with `DEDUPE`/`FAST`?
