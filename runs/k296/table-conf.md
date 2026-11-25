
| One-stream cells: forward ms a round (GPU clock) | ctl1 | ctl2 | conf_a | conf_b | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.84 | 33.71 | 33.76 | 33.80 | +0.02% | 0.38% | 0/2 |
| 32K × 1 | 33.07 | 33.19 | 33.10 | 32.94 | -0.34% | 0.38% | 1/2 |
| 128K × 1 | 33.84 | 33.87 | 33.84 | 33.65 | -0.32% | 0.08% | 1/2 |

| Multi-stream cells: total forward ms (rounds × ms a round) | ctl1 | ctl2 | conf_a | conf_b | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 4 | 8557 | 8673 | 8748 | 8607 | +0.72% | 1.35% | 1/2 |
| 32K × 4 | 8051 | 7975 | 8050 | 8098 | +0.76% | 0.95% | 1/2 |
| 128K × 4 | 8891 | 9123 | 8809 | 9090 | -0.64% | 2.57% | 1/2 |

rounds: 4K×4 154/158/159/157; 32K×4 130/128/129/132; 128K×4 152/156/144/156

| Wall seconds a cell (includes the cell's prefill) | ctl1 | ctl2 | conf_a | conf_b | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 5.040 | 5.020 | 5.030 | 5.040 | +0.10% | 0.40% | 0/2 |
| 4K × 4 | 14.910 | 15.050 | 15.140 | 14.970 | +0.50% | 0.93% | 1/2 |
| 32K × 1 | 5.660 | 5.670 | 5.670 | 5.640 | -0.18% | 0.18% | 1/2 |
| 32K × 4 | 14.610 | 14.480 | 14.590 | 14.640 | +0.48% | 0.89% | 1/2 |
| 128K × 1 | 6.300 | 6.310 | 6.300 | 6.280 | -0.24% | 0.16% | 1/2 |
| 128K × 4 | 15.960 | 16.100 | 15.760 | 16.100 | -0.62% | 0.87% | 1/2 |

host stalls (GPU idle between stretches, ms a round; a one-off event in that cell): ctl1 128K×4 0.33

Per-round distributions (ms; mean/median/p95/p99; forward = the GPU clock's stretches a round, wait = GPU idle between stretches a round, pass = the drafter pass a pass):
| cell | metric | ctl1 | ctl2 | conf_a | conf_b |
|---|---|---:|---:|---:|---:|
| 4K × 1 | forward | 33.84/-/-/- | 33.71/-/-/- | 33.76/-/-/- | 33.80/-/-/- |
| 4K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 4K × 1 | wait | 0.02/-/-/- | 0.02/-/-/- | 0.02/-/-/- | 0.03/-/-/- |
| 4K × 4 | forward | 55.57/-/-/- | 54.89/-/-/- | 55.02/-/-/- | 54.82/-/-/- |
| 4K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 4K × 4 | wait | 0.14/-/-/- | 0.15/-/-/- | 0.16/-/-/- | 0.16/-/-/- |
| 32K × 1 | forward | 33.07/-/-/- | 33.19/-/-/- | 33.10/-/-/- | 32.94/-/-/- |
| 32K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 32K × 1 | wait | 0.07/-/-/- | 0.07/-/-/- | 0.07/-/-/- | 0.09/-/-/- |
| 32K × 4 | forward | 61.93/-/-/- | 62.30/-/-/- | 62.40/-/-/- | 61.35/-/-/- |
| 32K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 32K × 4 | wait | 0.15/-/-/- | 0.19/-/-/- | 0.15/-/-/- | 0.13/-/-/- |
| 128K × 1 | forward | 33.84/-/-/- | 33.87/-/-/- | 33.84/-/-/- | 33.65/-/-/- |
| 128K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 128K × 1 | wait | 0.02/-/-/- | 0.07/-/-/- | 0.03/-/-/- | 0.02/-/-/- |
| 128K × 4 | forward | 58.50/-/-/- | 58.48/-/-/- | 61.17/-/-/- | 58.27/-/-/- |
| 128K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 128K × 4 | wait | 0.33/-/-/- | 0.14/-/-/- | 0.15/-/-/- | 0.16/-/-/- |

| Rank 0 lifetime profile (all of the arm's traffic) | ctl1 | ctl2 | conf_a | conf_b | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| pass_ms | 4.16 | 4.16 | 4.21 | 4.17 | +0.72% | 0.00% | 2/2 |
| absorb_ms | 0.26 | 0.25 | 0.26 | 0.25 | +0.00% | 3.92% | 0/2 |
| logits_ms | 0.05 | 0.05 | 0.05 | 0.05 | +0.00% | 0.00% | 0/2 |
| forward_ms | 38.58 | 38.60 | 38.57 | 38.46 | -0.19% | 0.05% | 2/2 |
| rows_a_round | 6.09 | 6.09 | 6.08 | 6.07 | -0.25% | 0.00% | 2/2 |
| streams_a_pass | 1.36 | 1.36 | 1.36 | 1.36 | +0.00% | 0.00% | 0/2 |

markov bias cache miss rate: ctl1 53.2%, ctl2 53.3%, conf_a 53.2%, conf_b 53.2%
ctl1 pass_counts: {"passes": 7166, "rows": 53395, "steps": {"1": 508, "2": 16, "3": 61, "4": 277, "5": 1787, "6": 4517, "7": 0, "8": 0}, "hits": 23374, "misses": 26616, "miss_steps": 22114}
ctl2 pass_counts: {"passes": 7160, "rows": 53492, "steps": {"1": 509, "2": 16, "3": 62, "4": 270, "5": 1785, "6": 4518, "7": 0, "8": 0}, "hits": 23391, "misses": 26644, "miss_steps": 22077}
conf_a pass_counts: {"passes": 7178, "rows": 53498, "steps": {"1": 508, "2": 16, "3": 64, "4": 277, "5": 1791, "6": 4522, "7": 0, "8": 0}, "hits": 23403, "misses": 26653, "miss_steps": 22163}
conf_b pass_counts: {"passes": 7190, "rows": 53516, "steps": {"1": 509, "2": 17, "3": 67, "4": 266, "5": 1799, "6": 4532, "7": 0, "8": 0}, "hits": 23427, "misses": 26635, "miss_steps": 22173}
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact conf_a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact conf_b: RESULT: IDENTICAL (0 problem(s), 39 cases)
