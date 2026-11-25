
| One-stream cells: forward ms a round (GPU clock) | ctl1 | ctl2 | dm065_a | dm065_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.84 | 33.71 | 33.62 | 33.62 | -0.45% | 0.38% | 2/2 |
| 32K × 1 | 33.07 | 33.19 | 33.05 | 33.05 | -0.24% | 0.38% | 2/2 |
| 128K × 1 | 33.84 | 33.87 | 33.78 | 33.78 | -0.22% | 0.08% | 2/2 |

| Multi-stream cells: total forward ms (rounds × ms a round) | ctl1 | ctl2 | dm065_a | dm065_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 4 | 8557 | 8673 | 8652 | 8652 | +0.43% | 1.35% | 0/2 |
| 32K × 4 | 8051 | 7975 | 7969 | 7969 | -0.55% | 0.95% | 2/2 |
| 128K × 4 | 8891 | 9123 | 8870 | 8870 | -1.53% | 2.57% | 2/2 |

rounds: 4K×4 154/158/158/158; 32K×4 130/128/127/127; 128K×4 152/156/152/152

| Wall seconds a cell (includes the cell's prefill) | ctl1 | ctl2 | dm065_a | dm065_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 5.040 | 5.020 | 5.010 | 5.010 | -0.40% | 0.40% | 2/2 |
| 4K × 4 | 14.910 | 15.050 | 15.100 | 15.100 | +0.80% | 0.93% | 2/2 |
| 32K × 1 | 5.660 | 5.670 | 5.650 | 5.650 | -0.26% | 0.18% | 2/2 |
| 32K × 4 | 14.610 | 14.480 | 14.470 | 14.470 | -0.52% | 0.89% | 2/2 |
| 128K × 1 | 6.300 | 6.310 | 6.310 | 6.310 | +0.08% | 0.16% | 0/2 |
| 128K × 4 | 15.960 | 16.100 | 15.860 | 15.860 | -1.06% | 0.87% | 2/2 |

host stalls (GPU idle between stretches, ms a round; a one-off event in that cell): ctl1 128K×4 0.33

Per-round distributions (ms; mean/median/p95/p99; forward = the GPU clock's stretches a round, wait = GPU idle between stretches a round, pass = the drafter pass a pass):
| cell | metric | ctl1 | ctl2 | dm065_a | dm065_a |
|---|---|---:|---:|---:|---:|
| 4K × 1 | forward | 33.84/-/-/- | 33.71/-/-/- | 33.62/-/-/- | 33.62/-/-/- |
| 4K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 4K × 1 | wait | 0.02/-/-/- | 0.02/-/-/- | 0.03/-/-/- | 0.03/-/-/- |
| 4K × 4 | forward | 55.57/-/-/- | 54.89/-/-/- | 54.76/-/-/- | 54.76/-/-/- |
| 4K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 4K × 4 | wait | 0.14/-/-/- | 0.15/-/-/- | 0.16/-/-/- | 0.16/-/-/- |
| 32K × 1 | forward | 33.07/-/-/- | 33.19/-/-/- | 33.05/-/-/- | 33.05/-/-/- |
| 32K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 32K × 1 | wait | 0.07/-/-/- | 0.07/-/-/- | 0.07/-/-/- | 0.07/-/-/- |
| 32K × 4 | forward | 61.93/-/-/- | 62.30/-/-/- | 62.75/-/-/- | 62.75/-/-/- |
| 32K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 32K × 4 | wait | 0.15/-/-/- | 0.19/-/-/- | 0.12/-/-/- | 0.12/-/-/- |
| 128K × 1 | forward | 33.84/-/-/- | 33.87/-/-/- | 33.78/-/-/- | 33.78/-/-/- |
| 128K × 1 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 128K × 1 | wait | 0.02/-/-/- | 0.07/-/-/- | 0.07/-/-/- | 0.07/-/-/- |
| 128K × 4 | forward | 58.50/-/-/- | 58.48/-/-/- | 58.35/-/-/- | 58.35/-/-/- |
| 128K × 4 | pass | -/-/-/- | -/-/-/- | -/-/-/- | -/-/-/- |
| 128K × 4 | wait | 0.33/-/-/- | 0.14/-/-/- | 0.15/-/-/- | 0.15/-/-/- |

| Rank 0 lifetime profile (all of the arm's traffic) | ctl1 | ctl2 | dm065_a | dm065_a | test vs control | control spread | test outside control range |
|---|---:|---:|---:|---:|---:|---:|---:|
| pass_ms | 4.16 | 4.16 | 4.18 | 4.18 | +0.48% | 0.00% | 2/2 |
| absorb_ms | 0.26 | 0.25 | 0.25 | 0.25 | -1.96% | 3.92% | 0/2 |
| logits_ms | 0.05 | 0.05 | 0.05 | 0.05 | +0.00% | 0.00% | 0/2 |
| forward_ms | 38.58 | 38.60 | 38.46 | 38.46 | -0.34% | 0.05% | 2/2 |
| rows_a_round | 6.09 | 6.09 | 6.07 | 6.07 | -0.33% | 0.00% | 2/2 |
| streams_a_pass | 1.36 | 1.36 | 1.36 | 1.36 | +0.00% | 0.00% | 0/2 |

markov bias cache miss rate: ctl1 53.2%, ctl2 53.3%, dm065_a 53.3%, dm065_a 53.3%
ctl1 pass_counts: {"passes": 7166, "rows": 53395, "steps": {"1": 508, "2": 16, "3": 61, "4": 277, "5": 1787, "6": 4517, "7": 0, "8": 0}, "hits": 23374, "misses": 26616, "miss_steps": 22114}
ctl2 pass_counts: {"passes": 7160, "rows": 53492, "steps": {"1": 509, "2": 16, "3": 62, "4": 270, "5": 1785, "6": 4518, "7": 0, "8": 0}, "hits": 23391, "misses": 26644, "miss_steps": 22077}
dm065_a pass_counts: {"passes": 7183, "rows": 53459, "steps": {"1": 508, "2": 15, "3": 62, "4": 268, "5": 1802, "6": 4528, "7": 0, "8": 0}, "hits": 23398, "misses": 26677, "miss_steps": 22176}
dm065_a pass_counts: {"passes": 7183, "rows": 53459, "steps": {"1": 508, "2": 15, "3": 62, "4": 268, "5": 1802, "6": 4528, "7": 0, "8": 0}, "hits": 23398, "misses": 26677, "miss_steps": 22176}
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact dm065_a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact dm065_a: RESULT: IDENTICAL (0 problem(s), 39 cases)
