# Energy per request (phase 1: the log line and rank 0's metric)

`TF_DS_ENERGY` measures the pair's GPU energy per request from NVML's cumulative counter
(`nvmlDeviceGetTotalEnergyConsumption`, millijoules since the driver started) and puts it where an operator can read it:
rank 0's `done` line, rank 0's `/metrics`, and a new per-frame line on rank 1's follower log. The reply's `tensorfold`
block is phase 2 and stays unchanged here. Patch: [`zig/patches/0117-energy-request.patch`](../zig/patches/0117-energy-request.patch);
tool: [`tools/energy.py`](../tools/energy.py).

Verified on the pair 2026-11-14 (head `gx10-eb0e`, worker `gx10-235f`, driver 580.159.03): both GB10s answer
`nvmlDeviceGetTotalEnergyConsumption` with rc 0 (head 12,619,882,529 mJ, rising ~58 J a second; worker
13,293,409,885 mJ), so a request's energy is an exact counter delta, not an integral of 3 s samples. The engine also
resolves `nvmlDeviceGetPowerUsage` (mW) as an optional cross-check read.

## The switch

- `TF_DS_ENERGY=1` or `=on` turns it on; unset, `0`, `off`, `""` or any other value leave it off (the series' rule: off
  by default, and `0`/`off` are the documented off spellings).
- Off: no driver call at all, no allocation, no new field. The `done` line, `/metrics` and the reply are byte for byte
  the base ones (the golden server tests and the `TF_DS_CACHE_METRICS=0` body test pin this).
- On: one NVML read at each request's arrival and finish on rank 0, and one read per frame on rank 1's follower (below).
- The switch is per rank and needs no agreement between the ranks, but a run is only joinable when both have it on: rank 0
  logs the ordinals, rank 1 logs the counter.
- The device read is `TF_CUDA_DEVICE`'s ordinal, else 0: each Spark of the pair has one GPU, so rank 0 reads the head's
  GPU and rank 1 the worker's. `TF_CUDA_DEVICE` is not set by `start.sh`; do not set it to another ordinal than the GPU
  the engine uses.
- A driver without `nvmlDeviceGetTotalEnergyConsumption` (or without `nvmlInit_v2`/the handle) does not fail anything:
  the reader still opens, the fields print `-`, and the P2P/KV/Engram paths are untouched (the optional lookup is its
  own, not a new required entry in an all-or-nothing table).

## What is logged

Rank 0's `done` line keeps every field it had and ends with four new ones, after them, so
`tools/zig/reuse_served.py` and any other parser of the old line still match:

```
... tokens_per_round=10.1 energy_j=6.412 idle_j=5.123 round_from=1041 round_to=1080
```

- `energy_j`: the counter's rise over this request's own window (arrival to the reply leaving), in joules with
  millijoule resolution. This is a **gross** figure: the head also runs a desktop and ~18 containers, so their draw is in
  it.
- `idle_j`: the counter's rise over an **equal window just before** the request, scaled from the gap between the last
  finished request and this one's arrival (`rise * request_window / gap`). `-` when there is no such gap or it is
  shorter than the request (then a request ran in it, so it is not an idle baseline). No sampling thread and no
  integration: it is one counter sample against another, normalised to the request's length. The marginal figure an
  operator wants is `energy_j - idle_j`; the tool prints both so the numbers can say which is worth keeping.
- `round_from`/`round_to`: the rank pair's wire frame ordinals the request's rounds ran in (the first frame the request
  needed and the last one it used). They are the join key for rank 1's line. `-` without a rank pair.

Rank 1 runs `tf-dsv41-lanes` (the follower): no HTTP server, no `/metrics`, no lane core and no `done` line — the
request boundaries only exist on rank 0, and per the recorded decision the rank-1-to-rank-0 exchange is phase 2. So rank
1 logs a **new line category of its own** while the switch is on, one line per frame it receives:

```
info: energy step=1041 mj=12619882529
```

`mj` is read before frame `step` runs. Rank 1 refuses a frame whose step is not the next one (`error.OutOfStep`), so both
ranks count the same ordinals: the worker GPU's figure for a request is the counter at `round_from` (before the first
frame) to the counter at the first frame after `round_to` (after the last one ran). That is an exact delta between two
ordinals, not a sample average. No existing parser matches this line (`reuse_served.py` reads `done` only). Volume: one
line a frame, about 25 lines a second at one stream (more with more streams), and only with the switch on — a long run
writes ~90 MB a day per rank at 1 stream (docker's `LOG_MAX_SIZE`/`LOG_MAX_FILE` rotate it).

## Rank 0's metric

With the switch on, `/metrics` carries `tensorfold:joules_total` (a counter, joules) beside
`tensorfold:generation_tokens_total`: the sum of finished requests' own windows on this rank's GPU. It is absent with the
switch off, so the base body is unchanged. Rank 1 has no `/metrics` (the follower is not a server), so its number is the
log line only; `tools/energy.py` adds the two.

## The tool

```
python3 -I tools/energy.py runs/energy/<tag>.rank0.log runs/energy/<tag>.rank1.log --streams 1
```

It prints, per request: the pair's J, each rank's J, the idle J, J per 1K prompt tokens, J per 1K decode tokens, the
accept rate and tokens/round (drafting and energy trade off, so they belong on the same row), then medians and the gross
and idle totals for the block. It **exits 1** and prints `rank 1 window not found for request <id>` when a request's
ordinal window has no rank 1 coverage (the worker log started late, the switch was off there, or a lone-driver path) —
never a silent average. `--json OUT` writes the rows. `--selftest` checks the parsing, the join and the arithmetic on
synthetic lines, with no pair.

## Phase 2 (deferred)

Only if the numbers are worth showing a caller: the pair total in the reply's `tensorfold` block as `energy_joules`. That
needs one small exchange at request end so rank 1's delta reaches rank 0, and it changes reply bytes, so it is its own
patch with its own `exact.py` identity gate. Not in this patch: the reply block, sampling threads, integration,
per-stream attribution (a 4-stream request's gross figure holds its siblings' draw too — the tool reports it as it is).

## LIVE GATES (what still needs the pair; none of it was run here — no GPU on the writing host)

Run in this order, and stop at the first failure:

1. **Image**: `scripts/prepare.sh` builds the image with 0117 applied (the Zig build of `tensorfold-native` and
   `tf-dsv41-lanes` is the first real compile of `native/cuda.zig`'s ordinal wrapper and the follower's line; host tests
   do not build `tf-dsv41-lanes`).
2. **Ordinal alignment (new)**: `just restart-head flags="TF_DS_ENERGY=1"` on both ranks, then one chat completion, then
   `just energy streams=1`. Every request must join (`rank 1 window not found` = the ordinals do not align: check that
   both ranks' logs start before the request and that `TF_DS_ENERGY=1` reached both containers — `docker exec … env`).
   Record the join count. This is the gate the host cannot do.
3. **Hardware smoke**: `./start.sh restart` with the switch on, one chat completion, the reply unchanged (`smoke.py`).
4. **Identity**: `tools/exact.py` and `tools/validate.py --compare` with the switch **on**, then with it **off**: replies,
   token shas and the 8-of-8 concurrency identity must match the recorded base (`runs/exact-oldimg.json`). An NVML read
   must not perturb a round.
5. **The row**: `to-benchmark.md`, one row at 1 stream and one at 4 streams, from two `just energy` runs:
   J/request, J/1K prompt tokens, J/1K decode tokens, accept rate and tokens/round, gross and idle apart.
6. Only then keep the switch on for a full block and decide (with the idle baseline) whether phase 2 is worth the reply
   change.

Rollback: unset `TF_DS_ENERGY` (or `=0`) and restart; nothing else changed.

## What was verified here (no GPU)

- `zig build test` and `zig/tests/server/zig_test.sh` pass with the patch (the switch off: every existing test's bytes
  unchanged).
- The switch parse, the off path (no counter read), the window arithmetic and both labels, the idle window's "no gap"
  case, the absent-driver/absent-symbol degradation to `-`, and the suffix's exact text after the old fields are host
  tests in `zig/src/core/energy.zig`.
- `tools/energy.py --selftest` passes.
- Not run here: any NVML call (this host has no `libnvidia-ml`), the ordinal alignment, the log volume, and the identity
  gates.
