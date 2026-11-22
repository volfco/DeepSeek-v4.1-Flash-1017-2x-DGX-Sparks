# k267: 0267 TF_DS_MARKOV_FUSED (fused greedy Markov bias miss + tile scoring)

Image `0f89e5a` (recipe-0264 + 0265..0268), production env + KEPT=2. Arms: ctl1, m267a, ctl2, m267b (17:04–17:39 UTC).
Greedy probe (the fused arm engages on all-greedy passes only) + sampled rows + kvreuse workload.

## Verdict: reject (exact, but greedy decode −6 to −10%)

- Exact panel IDENTICAL ×2, sampled IDENTICAL ×2, banner on both ranks in both test arms (the arm ran).
- Every greedy probe row is slower: tetris −6.9%, code −6.3%, json −7.1%, prose −10.2%, go/ts/sql/yaml −5.9..−7.6%.
  Sampled rows are unchanged (the arm doesn't engage there). Workload rounds are +0.95% slower (greedy workload).
- The fused kernel is slower than the separate bias-miss + scoring launches it replaces: one block computes the bias
  tile on a miss and scores in the same launch, so with ~52% misses the scoring waits on the bias computation
  instead of overlapping with it. Production traffic is sampled (T=1.0), so the item's upside was small anyway.

## Bias-cache counter (the todo item's open question)

Hits/misses per arm, rank 0, whole engine lifetime (exact panel + probe + workload):

| Arm | hits | misses | miss rate |
|---|---:|---:|---:|
| ctl1 | 18,824 | 20,198 | 51.8% |
| ctl2 | 18,799 | 20,127 | 51.7% |
| m267a | 18,813 | 20,120 | 51.7% |
| m267b | 18,831 | 20,136 | 51.7% |

**Counter bug:** the raw values print as 12297829382473053234 etc.: both counters start at Zig's `undefined` fill
(0xAAAAAAAAAAAAAAAA). The table subtracts that base. The fields need an explicit `= 0` before the counter is reusable.

## Tables

| Probe row | ctl1 | ctl2 | m267a | m267b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris greedy | 121.2 | 121.2 | 113.2 | 112.5 | 121.2 | 112.8 | **-6.88%** |
| tetris sampled | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | **+0.00%** |
| code greedy | 120.9 | 120.0 | 112.9 | 112.9 | 120.5 | 112.9 | **-6.25%** |
| json greedy | 107.4 | 107.4 | 99.7 | 99.7 | 107.4 | 99.7 | **-7.14%** |
| prose greedy | 60.5 | 60.0 | 54.1 | 54.1 | 60.2 | 54.1 | **-10.22%** |
| tetris sampled x2 | 168.5 | 168.6 | 168.8 | 168.4 | 168.5 | 168.6 | **+0.03%** |
| tetris sampled x4 | 218.1 | 219.4 | 218.2 | 217.1 | 218.7 | 217.7 | **-0.49%** |
| go greedy | 136.2 | 136.2 | 126.7 | 126.7 | 136.2 | 126.7 | **-6.93%** |
| ts greedy | 132.0 | 132.0 | 121.9 | 121.9 | 132.0 | 121.9 | **-7.62%** |
| sql greedy | 117.4 | 117.4 | 110.3 | 109.4 | 117.4 | 109.9 | **-6.44%** |
| yaml greedy | 133.3 | 133.3 | 125.5 | 125.5 | 133.3 | 125.5 | **-5.88%** |

mean tok/s change: -5.26% over 11 rows (positive = faster)
| Depth × streams | ctl1 | ctl2 | m267a | m267b | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.61 | 33.58 | 33.82 | 33.72 | 33.59 | 33.77 | **+0.54%** | -0.08% | 0/2 |
| 4K × 4 | 54.80 | 54.40 | 55.47 | 55.63 | 54.60 | 55.55 | **+1.74%** | -0.74% | 0/2 |
| 32K × 1 | 32.45 | 32.41 | 33.09 | 32.57 | 32.43 | 32.83 | **+1.24%** | -0.14% | 0/2 |
| 32K × 4 | 61.84 | 61.77 | 60.86 | 62.19 | 61.80 | 61.52 | **-0.45%** | -0.11% | 1/2 |
| 128K × 1 | 32.43 | 32.39 | 32.65 | 32.95 | 32.41 | 32.80 | **+1.20%** | -0.14% | 0/2 |
| 128K × 4 | 56.90 | 56.95 | 57.81 | 57.65 | 56.92 | 57.73 | **+1.42%** | +0.09% | 0/2 |

mean test vs control: +0.95%
exact ctl1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact ctl2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact m267a: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact m267b: RESULT: IDENTICAL (0 problem(s), 39 cases)
