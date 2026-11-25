# Capacity arms: ROUND_ROWS and PARALLEL at the live shape (2026-11-25)

Queue item 1. Runtime-only: no image build, no patch. Production's image `bench-fused-next-2bb052f` in every arm.

## Why this is the highest-value unmeasured experiment in the record

Live traffic on 2026-11-23 05:31–06:22 UTC ran **8 streams with up to 35 waiting, a 422 s mean time to first token, and
113 of 153 requests ending by client disconnect** (600 s cap). The box is capacity-bound. Two engine settings decide how
much work a round can carry, and neither has ever been varied on hardware:

- `round_rows` (16 in production) bounds a round's verified rows. With 8 streams and 16 rows the drafter's `room` rule
  (`lanes.zig:828`) gives each stream `16/8 − 1 = 1` draft a round, while the drafter's pass computes `N × n = 8 × 5 =
  40` block rows to score 8 (the pass costs the same whatever `steps` is). **Drafts 2–5 per stream are therefore free in
  the drafter**; the only marginal cost is the round's extra rows (3.15 ms a row, `runs/trace-review-20261123/`).
- `PARALLEL` (8 in production, `.env` line 12) is the engine's stream cap and `start.sh` allows 1–16. At 16 streams and
  16 rows `room = 0`, so **no stream gets a draft at all** — the ceiling has to move with the stream count.

`ROUND_ROWS` was blocked by `start.sh:64`, which refused anything but 16 on the belief that "the pinned kit has no wider
row kernels (issue #23)". That is stale: `0017-dsv41-kit-row-split.patch` is in the series and runs a gate the kit cannot
launch whole as 16-row pieces, `native.zig:904` documents `TF_DS_ROUND_ROWS` as 16..48, and `round.zig:46`'s `max_rows`
is 48. The guard was relaxed to 16..48 (and `scripts/test_round_rows.py` updated) for this suite.

**The split path is "expected, not yet shown on a GPU"** (`tri_norm.zig:167`): it is row-invariant by argument, and
`TF_DS_PLAIN_ROWS=16` keeps the cuBLAS shapes fixed so a wider round cannot move bits through cuBLAS's kernel pick. The
exact panel is the gate; if 32 or 48 rows move bits, that is the finding and production stays at 16.

## Arms

`arms-s8.txt` (8 streams, today's `PARALLEL`): `ctl8` (16 rows, the live shape), `s8rr32`, `s8rr48`, `ctl8b`.
`arms-s16.txt` (`PARALLEL=16`): `ctl16` (16 rows — room 0, no drafts, the counterfactual), `s16rr32`, `s16rr48`, `ctl16b`.

Every arm runs the standing exact reference `runs/deploy-10c9866-20261122/exact.json`, the sampled panel against the
first arm, and `runs/kvreuse-size-20261122/workload.py` at `DEPTHS=4096,32768 STREAMS=<arm's PARALLEL>`, which decodes
320 tokens a stream after prefilling each history one at a time. Per arm: `done` lines (ttft_s, queue_s, rounds,
accepted/drafted), `/metrics`, banner and error counts on both ranks. Production is restored and validated after each
run.

## Scoring

`score.py` reports per cell the aggregate completion tokens over wall (the capacity number), tokens a stream a round,
and the per-request ttft/queue distribution. **Read aggregate tok/s and TTFT, not round latency**: a wider round is
longer by construction, and the live constraint is throughput and wait, not a round's duration.

Stop rules: a `ROUND_ROWS` arm that fails the exact panel is a bit change, not a speed result — record it and stop.
An arm whose `pass_ms` rises without a rounds fall is not a capacity win.
