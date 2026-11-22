# k265: patches 0265 / 0266 / 0268, each switch alone on the stacked image

Image `bench-fused-next-5ec7aba` (recipe-0264 `27baa93` + 0265 + 0266 + 0268), production env + KEPT=2.
Arms in order: ctl1, h265a, p266a, q268a, ctl2, q268b, p266b, h265b (15:40–16:55 UTC). Gates per arm: exact panel
vs `sched2/ctl1.exact.json`, sampled identity vs `sched2/ctl1.sampled.json`, `suite_probe.py --narrow --sampled`, the
kvreuse workload (forward_ms per round, rank 0).

## Verdict

| Patch | Switch | Exact (39) | Sampled | Probe tok/s (18 rows) | Round time (6 cells) | Call |
|---|---|---|---|---:|---:|---|
| — | ctl1 on 5ec7aba, all off | IDENTICAL to production | IDENTICAL | — | — | image is a no-op when off |
| 0265 | `TF_DS_DRAFT_HEAD_SCORED=1` | IDENTICAL ×2 | IDENTICAL ×2 | −0.04% | −0.06% mean; **4K×1 −1.75% (2/2 below both controls)** | not slower; adopt pending combined arm |
| 0266 | `TF_DS_COMPRESS2_PAIR_ONCE=1` | IDENTICAL ×2 | IDENTICAL ×2 | −0.03% | +0.08% mean; nothing outside control drift | no measurable effect; exact |
| 0268 | `TF_DS_DRAFT_QKV_NORM=1` | IDENTICAL ×2 | IDENTICAL ×2 | +0.01% | −0.59% mean; **4K×1 −1.89%, 32K×4 −1.55% (both 2/2)** | not slower; adopt pending combined arm |

(Round-time column: negative = faster rounds.)

Reading the numbers:
- Probe tok/s is quantized (128.0, 121.2, 133.3 ...): the probe cannot resolve changes under ~0.5%, so "±0.0x%" means
  "not slower", not "neutral to 0.01%".
- The resolving signal is the workload's forward_ms per round. Control drift between ctl1 and ctl2 reaches 1.8–2.2%
  at 32K×1 and 128K×4, so only cells where both test arms sit below both controls count.
- Engagement: only 0265 logs a banner. 0266 resolves its two `_pair_kernel`s at load with `try` (a missing kernel
  refuses startup), and its call site engages under `TF_DS_FUSED_COMPRESS2_NORM` (production sets it) for d=128/512.
  0268's gate is `wq_a.n <= 2048` and the drafter's is 1280. Engagement at run time is therefore expected, but the
  rank-0 profile (pass_ms 4.11–4.17, forward_ms 38.21–38.35 across all 8 arms) cannot show it independently.

## Next

A combined arm on one image is required before deploy (0265 and 0268 both touch the drafter pass, and the stack
needed a hand-resolved conflict). If both 4K×1 gains are real, the combined 4K×1 should drop by roughly 3.5%.
Planned after k267: `ctl1, all3a, ctl2, all3b` (+ an all4 pair if 0267 is exact) on `0f89e5a`.

## Tables

| Probe row | ctl1 | ctl2 | h265a | h265b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris sampled s700 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| code sampled s700 | 122.9 | 123.9 | 122.9 | 123.9 | 123.4 | 123.4 | **+0.00%** |
| json sampled s700 | 105.9 | 106.7 | 105.9 | 106.7 | 106.3 | 106.3 | **+0.00%** |
| prose sampled s700 | 60.5 | 60.5 | 60.5 | 60.5 | 60.5 | 60.5 | **+0.00%** |
| tetris sampled s701 | 121.2 | 121.2 | 121.2 | 121.2 | 121.2 | 121.2 | **+0.00%** |
| code sampled s701 | 120.9 | 121.9 | 120.9 | 120.9 | 121.4 | 120.9 | **-0.40%** |
| json sampled s701 | 105.2 | 105.9 | 105.2 | 105.2 | 105.6 | 105.2 | **-0.34%** |
| prose sampled s701 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | **+0.00%** |
| go sampled s700 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| ts sampled s700 | 129.3 | 130.6 | 129.3 | 129.3 | 129.9 | 129.3 | **-0.51%** |
| sql sampled s700 | 110.3 | 111.3 | 111.3 | 111.3 | 110.8 | 111.3 | **+0.43%** |
| yaml sampled s700 | 126.7 | 126.7 | 126.7 | 126.7 | 126.7 | 126.7 | **+0.00%** |
| go sampled s701 | 134.7 | 134.7 | 134.7 | 134.7 | 134.7 | 134.7 | **+0.00%** |
| ts sampled s701 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | **+0.00%** |
| sql sampled s701 | 114.3 | 114.3 | 114.3 | 114.3 | 114.3 | 114.3 | **+0.00%** |
| yaml sampled s701 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| tetris sampled x2 | 168.4 | 168.5 | 168.5 | 168.6 | 168.5 | 168.5 | **+0.04%** |
| tetris sampled x4 | 219.8 | 217.6 | 217.3 | 220.0 | 218.7 | 218.6 | **-0.02%** |

mean tok/s change: -0.04% over 18 rows (positive = faster)
| Depth × streams | ctl1 | ctl2 | h265a | h265b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.88 | 33.87 | 33.30 | 33.26 | 33.87 | 33.28 | **-1.75%** | -0.04% | 2/2 |
| 4K × 4 | 54.45 | 54.59 | 54.68 | 54.65 | 54.52 | 54.67 | **+0.26%** | +0.27% | 0/2 |
| 32K × 1 | 32.84 | 32.25 | 32.34 | 32.89 | 32.55 | 32.62 | **+0.21%** | -1.80% | 0/2 |
| 32K × 4 | 61.78 | 61.72 | 61.80 | 60.90 | 61.75 | 61.35 | **-0.64%** | -0.10% | 1/2 |
| 128K × 1 | 32.40 | 32.39 | 32.48 | 32.44 | 32.40 | 32.46 | **+0.20%** | -0.03% | 0/2 |
| 128K × 4 | 57.84 | 56.58 | 57.49 | 58.46 | 57.21 | 57.98 | **+1.34%** | -2.18% | 0/2 |

mean test vs control: -0.06%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact h265a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact h265b: RESULT: IDENTICAL (0 problem(s), 39 cases)
| Probe row | ctl1 | ctl2 | p266a | p266b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris sampled s700 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| code sampled s700 | 122.9 | 123.9 | 123.9 | 123.9 | 123.4 | 123.9 | **+0.40%** |
| json sampled s700 | 105.9 | 106.7 | 105.9 | 106.7 | 106.3 | 106.3 | **+0.00%** |
| prose sampled s700 | 60.5 | 60.5 | 60.0 | 60.5 | 60.5 | 60.2 | **-0.39%** |
| tetris sampled s701 | 121.2 | 121.2 | 121.2 | 121.2 | 121.2 | 121.2 | **+0.00%** |
| code sampled s701 | 120.9 | 121.9 | 120.9 | 120.9 | 121.4 | 120.9 | **-0.40%** |
| json sampled s701 | 105.2 | 105.9 | 105.2 | 105.2 | 105.6 | 105.2 | **-0.34%** |
| prose sampled s701 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | **+0.00%** |
| go sampled s700 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| ts sampled s700 | 129.3 | 130.6 | 129.3 | 129.3 | 129.9 | 129.3 | **-0.51%** |
| sql sampled s700 | 110.3 | 111.3 | 111.3 | 111.3 | 110.8 | 111.3 | **+0.43%** |
| yaml sampled s700 | 126.7 | 126.7 | 126.7 | 126.7 | 126.7 | 126.7 | **+0.00%** |
| go sampled s701 | 134.7 | 134.7 | 134.7 | 134.7 | 134.7 | 134.7 | **+0.00%** |
| ts sampled s701 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | **+0.00%** |
| sql sampled s701 | 114.3 | 114.3 | 114.3 | 114.3 | 114.3 | 114.3 | **+0.00%** |
| yaml sampled s701 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| tetris sampled x2 | 168.4 | 168.5 | 168.5 | 168.8 | 168.5 | 168.6 | **+0.10%** |
| tetris sampled x4 | 219.8 | 217.6 | 217.5 | 220.4 | 218.7 | 218.9 | **+0.12%** |

mean tok/s change: -0.03% over 18 rows (positive = faster)
| Depth × streams | ctl1 | ctl2 | p266a | p266b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.88 | 33.87 | 33.92 | 33.97 | 33.87 | 33.94 | **+0.21%** | -0.04% | 0/2 |
| 4K × 4 | 54.45 | 54.59 | 54.35 | 55.27 | 54.52 | 54.81 | **+0.54%** | +0.27% | 1/2 |
| 32K × 1 | 32.84 | 32.25 | 32.30 | 32.37 | 32.55 | 32.34 | **-0.65%** | -1.80% | 0/2 |
| 32K × 4 | 61.78 | 61.72 | 60.72 | 62.65 | 61.75 | 61.68 | **-0.10%** | -0.10% | 1/2 |
| 128K × 1 | 32.40 | 32.39 | 32.43 | 31.97 | 32.40 | 32.20 | **-0.62%** | -0.03% | 1/2 |
| 128K × 4 | 57.84 | 56.58 | 58.59 | 57.08 | 57.21 | 57.83 | **+1.09%** | -2.18% | 0/2 |

mean test vs control: +0.08%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact p266a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact p266b: RESULT: IDENTICAL (0 problem(s), 39 cases)
| Probe row | ctl1 | ctl2 | q268a | q268b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris sampled s700 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| code sampled s700 | 122.9 | 123.9 | 122.9 | 123.9 | 123.4 | 123.4 | **+0.00%** |
| json sampled s700 | 105.9 | 106.7 | 105.9 | 106.7 | 106.3 | 106.3 | **+0.00%** |
| prose sampled s700 | 60.5 | 60.5 | 60.5 | 60.5 | 60.5 | 60.5 | **+0.00%** |
| tetris sampled s701 | 121.2 | 121.2 | 121.2 | 121.2 | 121.2 | 121.2 | **+0.00%** |
| code sampled s701 | 120.9 | 121.9 | 120.9 | 120.9 | 121.4 | 120.9 | **-0.40%** |
| json sampled s701 | 105.2 | 105.9 | 105.2 | 105.9 | 105.6 | 105.6 | **+0.00%** |
| prose sampled s701 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | **+0.00%** |
| go sampled s700 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| ts sampled s700 | 129.3 | 130.6 | 129.3 | 130.6 | 129.9 | 129.9 | **+0.00%** |
| sql sampled s700 | 110.3 | 111.3 | 111.3 | 111.3 | 110.8 | 111.3 | **+0.43%** |
| yaml sampled s700 | 126.7 | 126.7 | 126.7 | 126.7 | 126.7 | 126.7 | **+0.00%** |
| go sampled s701 | 134.7 | 134.7 | 134.7 | 134.7 | 134.7 | 134.7 | **+0.00%** |
| ts sampled s701 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | **+0.00%** |
| sql sampled s701 | 114.3 | 114.3 | 114.3 | 114.3 | 114.3 | 114.3 | **+0.00%** |
| yaml sampled s701 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| tetris sampled x2 | 168.4 | 168.5 | 168.7 | 168.8 | 168.5 | 168.8 | **+0.17%** |
| tetris sampled x4 | 219.8 | 217.6 | 219.7 | 217.6 | 218.7 | 218.7 | **-0.01%** |

mean tok/s change: +0.01% over 18 rows (positive = faster)
| Depth × streams | ctl1 | ctl2 | q268a | q268b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.88 | 33.87 | 33.29 | 33.17 | 33.87 | 33.23 | **-1.89%** | -0.04% | 2/2 |
| 4K × 4 | 54.45 | 54.59 | 54.88 | 54.30 | 54.52 | 54.59 | **+0.12%** | +0.27% | 1/2 |
| 32K × 1 | 32.84 | 32.25 | 32.34 | 32.79 | 32.55 | 32.57 | **+0.05%** | -1.80% | 0/2 |
| 32K × 4 | 61.78 | 61.72 | 60.54 | 61.03 | 61.75 | 60.79 | **-1.55%** | -0.10% | 2/2 |
| 128K × 1 | 32.40 | 32.39 | 32.46 | 32.34 | 32.40 | 32.40 | **+0.00%** | -0.03% | 1/2 |
| 128K × 4 | 57.84 | 56.58 | 57.08 | 57.04 | 57.21 | 57.06 | **-0.26%** | -2.18% | 0/2 |

mean test vs control: -0.59%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact q268a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact q268b: RESULT: IDENTICAL (0 problem(s), 39 cases)
