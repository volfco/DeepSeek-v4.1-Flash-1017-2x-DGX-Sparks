# kall: 0265 + 0266 + 0268 together (and + 0267) on 0f89e5a, vs its all-off control

Arms: ctl1, all3a, ctl2, all3b, all4a, all4b (17:39–18:35 UTC). Production env + KEPT=2. Sampled probe + workload.
Restore: production `.env`, validate rc=0.

## Verdict: hold. Exact, but no measurable gain, so production stays on 27baa93

- Exact IDENTICAL ×4 (all3 ×2, all4 ×2), sampled IDENTICAL ×4. Both ranks show the engagement banners: 0265's in all3,
  plus 0267's in all4.
- **all3** (0265 + 0266 + 0268): sampled probe −0.08% mean (inside the probe's ~0.5% quantum, every solo row's round
  count matches the control). Workload round time **+0.13% mean**:
  - 4K×1: +0.16%. The individual k265 arms' −1.75% (0265) and −1.89% (0268) did not add up to the expected ≈ −3.5%,
    and did not reproduce at all.
  - 4K×4: +1.08%, 32K×4: +0.68%. Slightly slower, both test arms above both controls.
- **all4** (+0267): probe −0.13%, rounds +1.25% (128K×4 +3.97%). 0267 stays rejected (k267).
- Reading: the single-switch 4K×1 gains in k265 were most likely run-order noise (≈0.6 ms on a 34 ms round, two arms
  each); nothing here clears the control drift. The adopt rule needs exact plus faster or neutral, with faster
  standalone. These are exact and neutral in serving but not faster anywhere, so they stay as off-by-default patches
  in the stack (`kernel-stack-0265`). Rerun them if a future change makes the drafter pass a larger share of the round.
- rank-0 profile (whole lifetime), pass_ms / forward_ms: ctl1 4.16 / 38.35, ctl2 4.14 / 38.26, all3a 4.13 / 38.36,
  all3b 4.14 / 38.37.

## Tables

metric: tok_s
| Probe row | ctl1 | ctl2 | all3a | all3b | control | test | test vs control | rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tetris sampled s700 | 128.00 | 128.81 | 128.00 | 128.00 | 128.41 | 128.00 | **-0.32%** | match |
| code sampled s700 | 122.88 | 122.88 | 122.88 | 122.88 | 122.88 | 122.88 | **+0.00%** | match |
| json sampled s700 | 105.93 | 106.67 | 105.93 | 105.93 | 106.30 | 105.93 | **-0.35%** | match |
| prose sampled s700 | 60.00 | 60.47 | 60.47 | 60.47 | 60.23 | 60.47 | **+0.39%** | match |
| tetris sampled s701 | 121.18 | 121.18 | 121.18 | 121.18 | 121.18 | 121.18 | **+0.00%** | match |
| code sampled s701 | 120.94 | 120.94 | 120.94 | 120.94 | 120.94 | 120.94 | **+0.00%** | match |
| json sampled s701 | 105.21 | 105.21 | 105.21 | 105.21 | 105.21 | 105.21 | **+0.00%** | match |
| prose sampled s701 | 58.63 | 58.63 | 58.63 | 58.63 | 58.63 | 58.63 | **+0.00%** | match |
| go sampled s700 | 128.00 | 126.73 | 128.00 | 128.00 | 127.37 | 128.00 | **+0.50%** | match |
| ts sampled s700 | 129.29 | 129.29 | 129.29 | 129.29 | 129.29 | 129.29 | **+0.00%** | match |
| sql sampled s700 | 110.34 | 111.30 | 110.34 | 110.34 | 110.82 | 110.34 | **-0.43%** | match |
| yaml sampled s700 | 126.73 | 126.73 | 126.73 | 126.73 | 126.73 | 126.73 | **+0.00%** | match |
| go sampled s701 | 134.74 | 134.74 | 134.74 | 134.74 | 134.74 | 134.74 | **+0.00%** | match |
| ts sampled s701 | 133.33 | 133.33 | 133.33 | 133.33 | 133.33 | 133.33 | **+0.00%** | match |
| sql sampled s701 | 114.29 | 114.29 | 113.27 | 114.29 | 114.29 | 113.78 | **-0.45%** | match |
| yaml sampled s701 | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 | 128.00 | **+0.00%** | match |
| tetris sampled x2 | 168.40 | 169.05 | 168.17 | 168.21 | 168.73 | 168.19 | **-0.32%** |  |
| tetris sampled x4 | 219.93 | 220.03 | 219.62 | 218.38 | 219.98 | 219.00 | **-0.45%** |  |

mean change: -0.08% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | all3a | all3b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.97 | 33.88 | 34.63 | 33.32 | 33.92 | 33.98 | **+0.16%** | -0.27% | 1/2 |
| 4K × 4 | 54.69 | 54.46 | 54.97 | 55.35 | 54.57 | 55.16 | **+1.08%** | -0.42% | 0/2 |
| 32K × 1 | 32.36 | 32.84 | 32.36 | 32.37 | 32.60 | 32.37 | **-0.71%** | +1.50% | 0/2 |
| 32K × 4 | 61.80 | 61.44 | 62.06 | 62.02 | 61.62 | 62.04 | **+0.68%** | -0.58% | 0/2 |
| 128K × 1 | 32.49 | 32.39 | 31.95 | 32.51 | 32.44 | 32.23 | **-0.65%** | -0.32% | 1/2 |
| 128K × 4 | 57.13 | 56.97 | 57.18 | 57.15 | 57.05 | 57.16 | **+0.21%** | -0.28% | 0/2 |

mean test vs control: +0.13%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact all3a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact all3b: RESULT: IDENTICAL (0 problem(s), 39 cases)
metric: tok_s
| Probe row | ctl1 | ctl2 | all4a | all4b | control | test | test vs control | rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tetris sampled s700 | 128.00 | 128.81 | 128.00 | 128.00 | 128.41 | 128.00 | **-0.32%** | match |
| code sampled s700 | 122.88 | 122.88 | 122.88 | 122.88 | 122.88 | 122.88 | **+0.00%** | match |
| json sampled s700 | 105.93 | 106.67 | 105.93 | 105.93 | 106.30 | 105.93 | **-0.35%** | match |
| prose sampled s700 | 60.00 | 60.47 | 60.47 | 60.47 | 60.23 | 60.47 | **+0.39%** | match |
| tetris sampled s701 | 121.18 | 121.18 | 121.18 | 120.47 | 121.18 | 120.83 | **-0.29%** | match |
| code sampled s701 | 120.94 | 120.94 | 120.94 | 120.94 | 120.94 | 120.94 | **+0.00%** | match |
| json sampled s701 | 105.21 | 105.21 | 105.21 | 105.21 | 105.21 | 105.21 | **+0.00%** | match |
| prose sampled s701 | 58.63 | 58.63 | 58.63 | 58.63 | 58.63 | 58.63 | **+0.00%** | match |
| go sampled s700 | 128.00 | 126.73 | 128.00 | 128.00 | 127.37 | 128.00 | **+0.50%** | match |
| ts sampled s700 | 129.29 | 129.29 | 129.29 | 129.29 | 129.29 | 129.29 | **+0.00%** | match |
| sql sampled s700 | 110.34 | 111.30 | 111.30 | 110.34 | 110.82 | 110.82 | **+0.00%** | match |
| yaml sampled s700 | 126.73 | 126.73 | 126.73 | 126.73 | 126.73 | 126.73 | **+0.00%** | match |
| go sampled s701 | 134.74 | 134.74 | 133.33 | 134.74 | 134.74 | 134.04 | **-0.52%** | match |
| ts sampled s701 | 133.33 | 133.33 | 133.33 | 133.33 | 133.33 | 133.33 | **+0.00%** | match |
| sql sampled s701 | 114.29 | 114.29 | 112.28 | 114.29 | 114.29 | 113.28 | **-0.88%** | match |
| yaml sampled s701 | 128.00 | 128.00 | 126.73 | 128.00 | 128.00 | 127.37 | **-0.50%** | match |
| tetris sampled x2 | 168.40 | 169.05 | 168.75 | 168.27 | 168.73 | 168.51 | **-0.13%** |  |
| tetris sampled x4 | 219.93 | 220.03 | 219.62 | 219.62 | 219.98 | 219.62 | **-0.16%** |  |

mean change: -0.13% over 18 rows (positive = better)
| Depth × streams | ctl1 | ctl2 | all4a | all4b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.97 | 33.88 | 34.00 | 33.34 | 33.92 | 33.67 | **-0.74%** | -0.27% | 1/2 |
| 4K × 4 | 54.69 | 54.46 | 55.76 | 55.41 | 54.57 | 55.59 | **+1.86%** | -0.42% | 0/2 |
| 32K × 1 | 32.36 | 32.84 | 32.39 | 32.98 | 32.60 | 32.69 | **+0.28%** | +1.50% | 0/2 |
| 32K × 4 | 61.80 | 61.44 | 62.81 | 62.63 | 61.62 | 62.72 | **+1.79%** | -0.58% | 0/2 |
| 128K × 1 | 32.49 | 32.39 | 32.55 | 32.55 | 32.44 | 32.55 | **+0.33%** | -0.32% | 0/2 |
| 128K × 4 | 57.13 | 56.97 | 60.43 | 58.20 | 57.05 | 59.31 | **+3.97%** | -0.28% | 0/2 |

mean test vs control: +1.25%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact all4a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact all4b: RESULT: IDENTICAL (0 problem(s), 39 cases)
