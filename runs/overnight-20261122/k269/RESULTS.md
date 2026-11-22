# k269: patches 0269 / 0270 / 0271, each switch alone on the stacked image 5c69584

Image `5c69584` (recipe-0264 `27baa93` + 0269 + 0270 + 0271), production env + KEPT=2. Arms: ctl1, t269a, p270a, r271a,
ctl2, r271b, p270b, t269b (18:52–20:10 UTC). First suite with the client-timed probe (`--narrow --sampled
--stream-bursts`: chunk-gap p50/p95, client tok/s, streamed x1/x4). Restore: validate rc=0.

## Verdict: all three exact, none measurably faster, so hold all three (production unchanged)

| Patch | Switch | Exact / sampled | Banner | Probe tok/s | client tok/s | chunk-gap p95 | Round time (mean of 6 cells) | prefill_ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| 0269 | `TF_DS_DRAFT_TAPS_DIRECT` | IDENTICAL ×2 | both ranks | −0.13% | −0.05% | −0.17% | +0.30% | 723.8 / 726.9 |
| 0270 | `TF_DS_PROMPT_PICKS_ONCE` | IDENTICAL ×2 | both ranks | −0.16% | −0.15% | −0.26% | +0.52% | 724.0 / 725.5 |
| 0271 | `TF_DS_RESUME_ASSEMBLE` | IDENTICAL ×2 | **none: never engaged** | −0.05% | −0.03% | −0.14% | +0.75% | 725.3 / 725.8 |

(Probe columns: positive = better. Round time: positive = slower. Control prefill_ms 725.3 / 725.9, the rank-0
lifetime mean of 114 prefills.)

- **0269:** engages on every verify round (banner on both ranks). It saves one ~210 KiB copy launch a round, below
  what any metric here resolves.
- **0270:** engages (banner) and saves 60 launches per prefill chunk per rank, but mean prefill moves by 0.1%, inside
  noise. Prefill is dominated by GEMMs and attention, not launch overhead.
- **0271:** no request in this suite took the exact-resume prepend path. It needs a multi-turn replay to be measured.
  Its arms are therefore an **A/A test** of the image, and that is the most useful result here (next section).

## Methodology finding: the round-time workload's noise floor

r271a/r271b ran code identical to the control, yet the workload shows +3.02% at 32K×1, +1.95% at 128K×1 and +0.75%
mean, with 0/2 "below all controls" in those cells. ctl1 vs ctl2 differ by 1.98% at 4K×1. So:
- one cell from 2 arms has an arm-to-arm noise of about ±2–3% (the run order puts the r271 arms between the control
  arms, so this isn't drift);
- the "both test arms below both controls" rule is satisfied by chance in about 1 of 6 cells for a no-op change, which
  is likely what k265's 0265 / 0268 "−1.8% at 4K×1" was (it did not reproduce in kall);
- for a future round-time claim, require it across several cells and replicated in a second suite, or use the rank-0
  lifetime `forward_ms` (38.85–39.03 here, ±0.25%), which pools ~7,000 rounds and is far tighter.

## New probe metrics (first use)

- Streamed x1 tetris: 129.6–129.9 tok/s, chunk-gap p95 45.4–45.7 ms (controls).
- Streamed x4: 214.6–217.0 tok/s aggregate, chunk-gap p95 70.8–71.1 ms.
- Arm-to-arm spread on chunk-gap p95 is about ±0.5%, so it resolves finer than tok/s.

## Tables

k269 finished 20:13:08
metric: tok_s
| tetris sampled x1 stream | 129.58 | 129.86 | 129.84 | 129.12 | 129.72 | 129.48 | **-0.19%** |  |
| tetris sampled x4 stream | 216.99 | 214.62 | 214.81 | 214.78 | 215.81 | 214.80 | **-0.47%** |  |
mean change: -0.13% over 20 rows (positive = better)
metric: client_tok_s
mean change: -0.05% over 16 rows (positive = better)
metric: chunk_gap_ms_p95
| tetris sampled x1 stream | 45.66 | 45.36 | 45.59 | 45.51 | 45.51 | 45.55 | **-0.09%** |  |
| tetris sampled x4 stream | 70.82 | 71.12 | 71.14 | 71.58 | 70.97 | 71.36 | **-0.55%** |  |
mean change: -0.17% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | t269a | t269b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 34.41 | 33.73 | 32.99 | 33.88 | 34.07 | 33.43 | **-1.88%** | -1.98% | 1/2 |
| 4K × 4 | 54.50 | 54.52 | 54.97 | 54.67 | 54.51 | 54.82 | **+0.56%** | +0.03% | 0/2 |
| 32K × 1 | 32.05 | 32.05 | 32.63 | 32.20 | 32.05 | 32.41 | **+1.14%** | +0.02% | 0/2 |
| 32K × 4 | 62.16 | 62.36 | 61.31 | 62.51 | 62.26 | 61.91 | **-0.56%** | +0.33% | 1/2 |
| 128K × 1 | 32.18 | 32.19 | 32.72 | 32.34 | 32.19 | 32.53 | **+1.08%** | +0.02% | 0/2 |
| 128K × 4 | 57.17 | 57.20 | 57.84 | 58.20 | 57.19 | 58.02 | **+1.46%** | +0.04% | 0/2 |

mean test vs control: +0.30%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact t269a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact t269b: RESULT: IDENTICAL (0 problem(s), 39 cases)
metric: tok_s
| tetris sampled x1 stream | 129.58 | 129.86 | 129.40 | 128.89 | 129.72 | 129.14 | **-0.44%** |  |
| tetris sampled x4 stream | 216.99 | 214.62 | 215.94 | 213.45 | 215.81 | 214.69 | **-0.51%** |  |
mean change: -0.16% over 20 rows (positive = better)
metric: client_tok_s
mean change: -0.15% over 16 rows (positive = better)
metric: chunk_gap_ms_p95
| tetris sampled x1 stream | 45.66 | 45.36 | 45.92 | 46.01 | 45.51 | 45.97 | **-1.00%** |  |
| tetris sampled x4 stream | 70.82 | 71.12 | 71.29 | 71.34 | 70.97 | 71.31 | **-0.49%** |  |
mean change: -0.26% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | p270a | p270b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 34.41 | 33.73 | 33.77 | 33.14 | 34.07 | 33.45 | **-1.82%** | -1.98% | 1/2 |
| 4K × 4 | 54.50 | 54.52 | 55.00 | 55.73 | 54.51 | 55.36 | **+1.56%** | +0.03% | 0/2 |
| 32K × 1 | 32.05 | 32.05 | 32.72 | 32.79 | 32.05 | 32.76 | **+2.21%** | +0.02% | 0/2 |
| 32K × 4 | 62.16 | 62.36 | 60.95 | 62.69 | 62.26 | 61.82 | **-0.71%** | +0.33% | 1/2 |
| 128K × 1 | 32.18 | 32.19 | 32.81 | 32.90 | 32.19 | 32.86 | **+2.08%** | +0.02% | 0/2 |
| 128K × 4 | 57.17 | 57.20 | 56.77 | 57.33 | 57.19 | 57.05 | **-0.23%** | +0.04% | 1/2 |

mean test vs control: +0.52%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact p270a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact p270b: RESULT: IDENTICAL (0 problem(s), 39 cases)
metric: tok_s
| tetris sampled x1 stream | 129.58 | 129.86 | 130.09 | 130.22 | 129.72 | 130.16 | **+0.34%** |  |
| tetris sampled x4 stream | 216.99 | 214.62 | 216.06 | 215.07 | 215.81 | 215.56 | **-0.11%** |  |
mean change: -0.05% over 20 rows (positive = better)
metric: client_tok_s
mean change: -0.03% over 16 rows (positive = better)
metric: chunk_gap_ms_p95
| tetris sampled x1 stream | 45.66 | 45.36 | 45.05 | 45.61 | 45.51 | 45.33 | **+0.40%** |  |
| tetris sampled x4 stream | 70.82 | 71.12 | 71.07 | 71.70 | 70.97 | 71.38 | **-0.58%** |  |
mean change: -0.14% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | r271a | r271b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 34.41 | 33.73 | 33.75 | 33.77 | 34.07 | 33.76 | **-0.91%** | -1.98% | 0/2 |
| 4K × 4 | 54.50 | 54.52 | 54.98 | 54.22 | 54.51 | 54.60 | **+0.15%** | +0.03% | 1/2 |
| 32K × 1 | 32.05 | 32.05 | 33.32 | 32.71 | 32.05 | 33.02 | **+3.02%** | +0.02% | 0/2 |
| 32K × 4 | 62.16 | 62.36 | 62.60 | 61.47 | 62.26 | 62.03 | **-0.37%** | +0.33% | 1/2 |
| 128K × 1 | 32.18 | 32.19 | 32.82 | 32.80 | 32.19 | 32.81 | **+1.95%** | +0.02% | 0/2 |
| 128K × 4 | 57.17 | 57.20 | 58.11 | 56.98 | 57.19 | 57.55 | **+0.64%** | +0.04% | 1/2 |

mean test vs control: +0.75%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact r271a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact r271b: RESULT: IDENTICAL (0 problem(s), 39 cases)
2026-11-22T18:52:55Z arm ctl1 image=tensorfold-dsv41:bench-fused-next-5c69584 env=[]
2026-11-22T19:02:36Z arm t269a image=tensorfold-dsv41:bench-fused-next-5c69584 env=[TF_DS_DRAFT_TAPS_DIRECT=1]
2026-11-22T19:12:13Z arm p270a image=tensorfold-dsv41:bench-fused-next-5c69584 env=[TF_DS_PROMPT_PICKS_ONCE=1]
2026-11-22T19:21:51Z arm r271a image=tensorfold-dsv41:bench-fused-next-5c69584 env=[TF_DS_RESUME_ASSEMBLE=1]
2026-11-22T19:31:28Z arm ctl2 image=tensorfold-dsv41:bench-fused-next-5c69584 env=[]
2026-11-22T19:41:05Z arm r271b image=tensorfold-dsv41:bench-fused-next-5c69584 env=[TF_DS_RESUME_ASSEMBLE=1]
2026-11-22T19:50:42Z arm p270b image=tensorfold-dsv41:bench-fused-next-5c69584 env=[TF_DS_PROMPT_PICKS_ONCE=1]
2026-11-22T20:00:20Z arm t269b image=tensorfold-dsv41:bench-fused-next-5c69584 env=[TF_DS_DRAFT_TAPS_DIRECT=1]
restored: validate rc=0
all passed
ctl1 pass_ms 4.19 forward_ms 38.85 prefills 114 prefill_ms 725.3
ctl2 pass_ms 4.22 forward_ms 38.87 prefills 114 prefill_ms 725.9
t269a pass_ms 4.2 forward_ms 38.85 prefills 114 prefill_ms 723.8
t269b pass_ms 4.23 forward_ms 39.03 prefills 114 prefill_ms 726.9
p270a pass_ms 4.19 forward_ms 38.91 prefills 114 prefill_ms 724.0
p270b pass_ms 4.21 forward_ms 39.0 prefills 114 prefill_ms 725.5
r271a pass_ms 4.22 forward_ms 38.93 prefills 114 prefill_ms 725.3
r271b pass_ms 4.21 forward_ms 38.9 prefills 114 prefill_ms 725.8

[exited with code 0]
