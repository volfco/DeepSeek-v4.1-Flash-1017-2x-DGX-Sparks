# k272: 0272 ring trace + 0273/0274 bug fixes + 0275 COPY_MIN on 10c9866

Image `10c9866` = `kernel-stack-0272` = recipe-0264 `27baa93` + 0272 + 0273 + 0274 + 0275. Production env + KEPT=2.
Arms: ctl1, cm3a, tr16a, ctl2, tr16b, cm3b (20:21–21:20 UTC). Sampled probe with streamed bursts + workload. Restore:
validate rc=0.

## Verdicts

| Patch | Arms | Exact / sampled | Result | Call |
|---|---|---|---|---|
| 0273 failed-capture join (no switch) | ctl1, ctl2 | IDENTICAL ×2 vs production | forward_ms 38.89 / 38.96 (k269 controls 38.85 / 38.87) | **adopted** (deployed 21:25 UTC) |
| 0274 tool-call first token (no switch) | ctl1, ctl2 | IDENTICAL ×2 | functional check after deploy: `tool_choice: required` + thinking returns `get_weather({"city":"Paris"})` | **adopted** |
| 0275 `TF_DS_COPY_MIN=3` | cm3a, cm3b | IDENTICAL ×2 | probe +0.14%, client tok/s +0.11%, chunk-gap p95 −0.02%; forward_ms 38.92 / 38.91; 0.1–0.3% fewer rounds | neutral; held, off by default. Banner on rank 0 only (the leader proposes drafts) |
| 0272 `TF_DS_RING_TRACE=16` | tr16a, tr16b | IDENTICAL ×2 | forward_ms 39.06 / 38.93 (+0.2%), x4 chunk-gap p95 −2.4% | diagnostic; stays off. ~99K records a rank, 0 drops |

Deploy: `runs/deploy-10c9866-20261122/` (exact 39/39 IDENTICAL vs `sched2/ctl1`, sampled IDENTICAL, toolcall OK, validate
rc=0). 0272 and 0275 are in the image, unset.

## Scheduling round B: what the trace shows (both arms agree)

Same-clock intervals paired by ring sequence (`h0272/summarize.py`), p50 per op and rank, tr16a / tr16b:

| op (samples/arm) | doorbell→flag r0 | doorbell→flag r1 | producer r0 | producer r1 | proxy doorbell→post | proxy completion→flag |
|---|---:|---:|---:|---:|---:|---:|
| attn_gather (18.6K) | 7.4 / 6.7 µs | 20.8 / 20.9 µs | 227.0 / 227.6 µs | 220.8 / 220.1 µs | 0.69–0.72 µs | 32 ns |
| moe_gather (18.8K) | 8.2 / 7.8 µs | 18.2 / 18.8 µs | 590.5 / 590.2 µs | 582.0 / 583.6 µs | 0.69–0.74 µs | 32 ns |
| kv (3.7K) | 47.3 / 48.2 µs | 5.9 / 6.1 µs | 196.8 / 197.0 µs | 199.7 / 201.9 µs | 0.69–0.74 µs | 16–32 ns |
| engram (0.9K) | 16.5 / 13.8 µs | 51.7 / 52.0 µs | 220.4 / 244.2 µs | 212.8 / 219.0 µs | ~0.7 µs | 32 ns |

- **The proxy is not the cost.** Doorbell→last write posted is ~0.7 µs, and completion→flag published is 16–32 ns on
  both ranks. With A2's zero involuntary switches, proxy service is ruled out.
- **Rank 0 is the late producer on the hidden gathers, at low row counts.** Its GPU time between gathers is longer,
  and rank 1 waits more (p50 18–21 µs vs 7–8 µs). **Correction (B2):** that is not ~1 ms a round. Each rank proceeds
  when its peer's data lands, so imbalance costs about half the producer difference per gather. Paired by sequence and
  averaged: at 1–7 rows the mean producer gap (r0 − r1) is +6.2 / +6.7 µs, so ≈3.2 µs a gather, **≈0.25 ms a round
  (~0.7%)**. At 8+ rows it *reverses* (−5.4 / −6.0 µs: rank 1 late), ≈0.23 ms a round. The balanced wait (mean of the
  two ranks' waits) is 19.6 µs at 1–7 rows and 28 µs at 8+, which is the transfer floor, not the fast side's 7–8 µs.
  See `runs/sched-20261122/b2/` for the root-cause work.
- KV exchanges run the other way (rank 0 waits ~47 µs, rank 1 ~6 µs). They are ~5× rarer.
- **Not the cause** (checked): GPU clocks (both ~2,380 MHz under a 4-stream burst, no throttle reasons); configuration
  (rank 1's command line carries `--graphs 1 --side 1 --prefetch 1 --aio 1`, the same as rank 0's env); other GPU
  processes (none on either node; no compositor). Rank 0's process uses ~1.4 GiB more GPU memory than rank 1's.
- **Next (round B2):** an nsys capture of a few graphs-on decode rounds on both ranks, then a per-kernel duration diff
  per layer. That finds which kernels run longer on rank 0 (expert-shard load from `PLACE_BOUND`/routing skew, the
  leader's extra work on its stream, or memory-bandwidth contention from rank 0's CPU-side serving). SCHEDULING.md
  row 1 (move the MoE prefetch fork) only applies if the extra time sits next to the prefetch fork.

## Tables

k272 finished 21:22:58
metric: tok_s
| tetris sampled x1 stream | 129.92 | 129.43 | 129.60 | 129.22 | 129.68 | 129.41 | **-0.20%** |  |
| tetris sampled x4 stream | 214.45 | 214.05 | 215.80 | 215.95 | 214.25 | 215.88 | **+0.76%** |  |
mean change: +0.14% over 20 rows (positive = better)
metric: client_tok_s
mean change: +0.11% over 16 rows (positive = better)
metric: chunk_gap_ms_p95
| tetris sampled x1 stream | 45.51 | 45.85 | 45.93 | 45.77 | 45.68 | 45.85 | **-0.37%** |  |
| tetris sampled x4 stream | 70.72 | 71.21 | 71.48 | 71.53 | 70.97 | 71.50 | **-0.76%** |  |
mean change: -0.02% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | cm3a | cm3b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.72 | 34.48 | 33.06 | 33.03 | 34.10 | 33.04 | **-3.10%** | +2.26% | 2/2 |
| 4K × 4 | 54.67 | 54.75 | 56.20 | 55.10 | 54.71 | 55.65 | **+1.71%** | +0.14% | 0/2 |
| 32K × 1 | 32.66 | 32.73 | 32.10 | 32.67 | 32.69 | 32.39 | **-0.93%** | +0.21% | 1/2 |
| 32K × 4 | 60.96 | 61.60 | 62.78 | 62.25 | 61.28 | 62.52 | **+2.02%** | +1.04% | 0/2 |
| 128K × 1 | 33.33 | 33.99 | 32.25 | 32.78 | 33.66 | 32.51 | **-3.40%** | +1.99% | 2/2 |
| 128K × 4 | 58.50 | 57.72 | 57.25 | 58.24 | 58.11 | 57.75 | **-0.63%** | -1.33% | 1/2 |

mean test vs control: -0.72%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact cm3a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact cm3b: RESULT: IDENTICAL (0 problem(s), 39 cases)
metric: tok_s
| tetris sampled x1 stream | 129.92 | 129.43 | 129.27 | 129.24 | 129.68 | 129.25 | **-0.32%** |  |
| tetris sampled x4 stream | 214.45 | 214.05 | 214.05 | 215.05 | 214.25 | 214.55 | **+0.14%** |  |
mean change: -0.12% over 20 rows (positive = better)
metric: client_tok_s
mean change: -0.18% over 16 rows (positive = better)
metric: chunk_gap_ms_p95
| tetris sampled x1 stream | 45.51 | 45.85 | 45.93 | 46.20 | 45.68 | 46.06 | **-0.84%** |  |
| tetris sampled x4 stream | 70.72 | 71.21 | 73.31 | 72.00 | 70.97 | 72.66 | **-2.38%** |  |
mean change: -0.23% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | tr16a | tr16b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.72 | 34.48 | 33.16 | 33.77 | 34.10 | 33.46 | **-1.86%** | +2.26% | 1/2 |
| 4K × 4 | 54.67 | 54.75 | 56.07 | 54.62 | 54.71 | 55.34 | **+1.15%** | +0.14% | 1/2 |
| 32K × 1 | 32.66 | 32.73 | 33.44 | 32.71 | 32.69 | 33.08 | **+1.18%** | +0.21% | 0/2 |
| 32K × 4 | 60.96 | 61.60 | 61.61 | 61.39 | 61.28 | 61.50 | **+0.36%** | +1.04% | 0/2 |
| 128K × 1 | 33.33 | 33.99 | 33.51 | 32.22 | 33.66 | 32.87 | **-2.35%** | +1.99% | 1/2 |
| 128K × 4 | 58.50 | 57.72 | 58.35 | 58.03 | 58.11 | 58.19 | **+0.14%** | -1.33% | 0/2 |

mean test vs control: -0.23%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact tr16a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact tr16b: RESULT: IDENTICAL (0 problem(s), 39 cases)
  trace lines: r0 98712 r1 98712
  trace lines: r0 99038 r1 99038
restored: validate rc=0
all passed
ctl1 pass_ms 4.21 forward_ms 38.89 rounds 7503 rows 6.34
ctl2 pass_ms 4.22 forward_ms 38.96 rounds 7505 rows 6.34
cm3a pass_ms 4.22 forward_ms 38.92 rounds 7483 rows 6.35
cm3b pass_ms 4.2 forward_ms 38.91 rounds 7499 rows 6.35
tr16a pass_ms 4.26 forward_ms 39.06 rounds 7484 rows 6.35
tr16b pass_ms 4.23 forward_ms 38.93 rounds 7508 rows 6.34
