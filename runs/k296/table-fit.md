
| One-stream cells: forward ms a round (GPU clock) | ctl1 | ctl2 | fit_a | fit_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 (rounds differ: [95, 97]) | 33.84 | 33.71 | 34.36 | 34.36 | +1.73% | 0.38% | 2/2 |
| 32K × 1 (rounds differ: [108, 114]) | 33.07 | 33.19 | 34.17 | 34.17 | +3.13% | 0.38% | 2/2 |
| 128K × 1 (rounds differ: [125, 126]) | 33.84 | 33.87 | 34.56 | 34.56 | +2.08% | 0.08% | 2/2 |

| Multi-stream cells: total forward ms (rounds × ms a round) | ctl1 | ctl2 | fit_a | fit_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 4 | 8557 | 8673 | 8576 | 8576 | -0.46% | 1.35% | 0/2 |
| 32K × 4 | 8051 | 7975 | 7999 | 7999 | -0.17% | 0.95% | 0/2 |
| 128K × 4 | 8891 | 9123 | 8963 | 8963 | -0.49% | 2.57% | 0/2 |

rounds: 4K×4 154/158/157/157; 32K×4 130/128/131/131; 128K×4 152/156/153/153

| Wall seconds a cell (includes the cell's prefill) | ctl1 | ctl2 | fit_a | fit_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 5.040 | 5.020 | 5.010 | 5.010 | -0.40% | 0.40% | 2/2 |
| 4K × 4 | 14.910 | 15.050 | 14.920 | 14.920 | -0.40% | 0.93% | 0/2 |
| 32K × 1 | 5.660 | 5.670 | 5.550 | 5.550 | -2.03% | 0.18% | 2/2 |
| 32K × 4 | 14.610 | 14.480 | 14.550 | 14.550 | +0.03% | 0.89% | 0/2 |
| 128K × 1 | 6.300 | 6.310 | 6.360 | 6.360 | +0.87% | 0.16% | 2/2 |
| 128K × 4 | 15.960 | 16.100 | 16.000 | 16.000 | -0.19% | 0.87% | 0/2 |

host stalls (GPU idle between stretches, ms a round; a one-off event in that cell): ctl1 128K×4 0.33

Per-round distributions (ms; mean/median/p95/p99; forward = the GPU clock's stretches a round, wait = GPU idle between stretches a round, pass = the drafter pass a pass):
| cell | metric | ctl1 | ctl2 | fit_a | fit_a |
|---|---|---:|---:|---:|---:|
| 4K × 1 | forward | 33.84/-/-/- | 33.71/-/-/- | 34.36/-/-/- | 34.36/-/-/- |
| 4K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 4K × 1 | wait | 0.02/-/-/- | 0.02/-/-/- | 0.02/-/-/- | 0.02/-/-/- |
| 4K × 4 | forward | 55.57/-/-/- | 54.89/-/-/- | 54.62/-/-/- | 54.62/-/-/- |
| 4K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 4K × 4 | wait | 0.14/-/-/- | 0.15/-/-/- | 0.13/-/-/- | 0.13/-/-/- |
| 32K × 1 | forward | 33.07/-/-/- | 33.19/-/-/- | 34.17/-/-/- | 34.17/-/-/- |
| 32K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 32K × 1 | wait | 0.07/-/-/- | 0.07/-/-/- | 0.08/-/-/- | 0.08/-/-/- |
| 32K × 4 | forward | 61.93/-/-/- | 62.30/-/-/- | 61.06/-/-/- | 61.06/-/-/- |
| 32K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 32K × 4 | wait | 0.15/-/-/- | 0.19/-/-/- | 0.17/-/-/- | 0.17/-/-/- |
| 128K × 1 | forward | 33.84/-/-/- | 33.87/-/-/- | 34.56/-/-/- | 34.56/-/-/- |
| 128K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 128K × 1 | wait | 0.02/-/-/- | 0.07/-/-/- | 0.06/-/-/- | 0.06/-/-/- |
| 128K × 4 | forward | 58.50/-/-/- | 58.48/-/-/- | 58.58/-/-/- | 58.58/-/-/- |
| 128K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 128K × 4 | wait | 0.33/-/-/- | 0.14/-/-/- | 0.25/-/-/- | 0.25/-/-/- |

| Rank 0 lifetime profile (all of the arm's traffic) | ctl1 | ctl2 | fit_a | fit_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| pass_ms | 4.16 | 4.16 | 4.18 | 4.18 | +0.48% | 0.00% | 2/2 |
| absorb_ms | 0.26 | 0.25 | 0.26 | 0.26 | +1.96% | 3.92% | 0/2 |
| logits_ms | 0.05 | 0.05 | 0.05 | 0.05 | +0.00% | 0.00% | 0/2 |
| forward_ms | 38.58 | 38.60 | 38.72 | 38.72 | +0.34% | 0.05% | 2/2 |
| rows_a_round | 6.09 | 6.09 | 6.12 | 6.12 | +0.49% | 0.00% | 2/2 |
| streams_a_pass | 1.36 | 1.36 | 1.37 | 1.37 | +0.74% | 0.00% | 2/2 |

markov bias cache miss rate: ctl1 53.2%, ctl2 53.3%, fit_a 53.2%, fit_a 53.2%
ctl1 pass_counts: {"passes": 7166, "rows": 53395, "steps": {"1": 508, "2": 16, "3": 61, "4": 277, "5": 1787, "6": 4517, "7": 0, "8": 0}, "hits": 23374, "misses": 26616, "miss_steps": 22114}
ctl2 pass_counts: {"passes": 7160, "rows": 53492, "steps": {"1": 509, "2": 16, "3": 62, "4": 270, "5": 1785, "6": 4518, "7": 0, "8": 0}, "hits": 23391, "misses": 26644, "miss_steps": 22077}
fit_a pass_counts: {"passes": 7152, "rows": 53533, "steps": {"1": 511, "2": 10, "3": 77, "4": 288, "5": 1754, "6": 4512, "7": 0, "8": 0}, "hits": 23397, "misses": 26608, "miss_steps": 22080}
fit_a pass_counts: {"passes": 7152, "rows": 53533, "steps": {"1": 511, "2": 10, "3": 77, "4": 288, "5": 1754, "6": 4512, "7": 0, "8": 0}, "hits": 23397, "misses": 26608, "miss_steps": 22080}
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact fit_a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact fit_a: RESULT: IDENTICAL (0 problem(s), 39 cases)
