# 0260 TF_DS_CONF_PACK + 0261 TF_DS_TOPK_MERGE, November 22: exact (greedy and sampled), speed-neutral

One image (`bench-fused-next-864d32f` = recipe tree + 0259 + 0260 + 0261, switches off by default), six arms 08:50–09:27 UTC, with production's `.env` (`TF_DS_KV_FETCH_REUSE=1`, `TF_DS_T27_DEVICE_DRAW=1`) in each. Production was restored and validated (all passed).

- **Exact panel: IDENTICAL 39/39 in every arm** against the [reference](../../kvreuse-ab-20261122/off1.exact.json).
- **Sampled identity ([sampled.py](../sampled.py)): IDENTICAL in conf1, merge1, both1, both2 and base2** against base1. That is three waves of four concurrent seeded T=1.0 requests with top_k 1/5/20/56, 3/8/40/56 and 20×4 (the round's kmax varies), plus four solo top_k 20, 160 tokens each. **This is 0261's real test, because the greedy panel never takes its path.**
- **Speed (greedy round-time workload, so 0261 is not exercised there):** both1/both2 against base1/base2 is **+0.16%** (control drift ≤0.6%). conf1 alone read +0.77%, which the both arms do not reproduce.
- **Engagement not proven here.** Neither patch logged when its path ran. A banner commit (`086e8cb`) adds a one-time log at each fused call site; the [lineage](../lineage/) run checks it.
- Cross-suite caution: this suite's 4K cells ran 98 vs 131 rounds against the recipe gate for the same 320 tokens, because the workload corpus (built from the repo's RESULTS.md files) had changed between suites. Within a suite the arms share the corpus. From here the corpus is frozen (`runs/overnight-20261122/corpus.json`).

### conf1 vs base1,base2

| Depth × streams | base1 | base2 | conf1 | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.31 | 33.40 | 33.49 | 33.35 | 33.49 | **+0.43%** | +0.27% | 0/1 |
| 4K × 4 | 54.41 | 54.41 | 54.72 | 54.41 | 54.72 | **+0.58%** | -0.00% | 0/1 |
| 32K × 1 | 32.42 | 32.30 | 32.69 | 32.36 | 32.69 | **+1.02%** | -0.39% | 0/1 |
| 32K × 4 | 61.92 | 61.52 | 62.66 | 61.72 | 62.66 | **+1.52%** | -0.63% | 0/1 |
| 128K × 1 | 32.64 | 32.64 | 32.67 | 32.64 | 32.67 | **+0.09%** | +0.02% | 0/1 |
| 128K × 4 | 57.21 | 56.98 | 57.67 | 57.10 | 57.67 | **+1.01%** | -0.39% | 0/1 |

mean test vs control: +0.77%
exact base1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact base2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact conf1: RESULT: IDENTICAL (0 problem(s), 39 cases)

### merge1 vs base1,base2

| Depth × streams | base1 | base2 | merge1 | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.31 | 33.40 | 33.51 | 33.35 | 33.51 | **+0.46%** | +0.27% | 0/1 |
| 4K × 4 | 54.41 | 54.41 | 54.34 | 54.41 | 54.34 | **-0.13%** | -0.00% | 1/1 |
| 32K × 1 | 32.42 | 32.30 | 32.53 | 32.36 | 32.53 | **+0.52%** | -0.39% | 0/1 |
| 32K × 4 | 61.92 | 61.52 | 61.23 | 61.72 | 61.23 | **-0.79%** | -0.63% | 1/1 |
| 128K × 1 | 32.64 | 32.64 | 32.42 | 32.64 | 32.42 | **-0.66%** | +0.02% | 1/1 |
| 128K × 4 | 57.21 | 56.98 | 59.34 | 57.10 | 59.34 | **+3.93%** | -0.39% | 0/1 |

mean test vs control: +0.56%
exact base1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact base2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact merge1: RESULT: IDENTICAL (0 problem(s), 39 cases)

### both1,both2 vs base1,base2

| Depth × streams | base1 | base2 | both1 | both2 | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.31 | 33.40 | 33.36 | 33.32 | 33.35 | 33.34 | **-0.03%** | +0.27% | 0/2 |
| 4K × 4 | 54.41 | 54.41 | 54.76 | 54.47 | 54.41 | 54.61 | **+0.38%** | -0.00% | 0/2 |
| 32K × 1 | 32.42 | 32.30 | 32.57 | 32.48 | 32.36 | 32.52 | **+0.51%** | -0.39% | 0/2 |
| 32K × 4 | 61.92 | 61.52 | 62.01 | 61.92 | 61.72 | 61.96 | **+0.39%** | -0.63% | 0/2 |
| 128K × 1 | 32.64 | 32.64 | 32.51 | 32.56 | 32.64 | 32.54 | **-0.31%** | +0.02% | 2/2 |
| 128K × 4 | 57.21 | 56.98 | 57.04 | 57.19 | 57.10 | 57.12 | **+0.03%** | -0.39% | 0/2 |

mean test vs control: +0.16%
exact base1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact base2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact both1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact both2: RESULT: IDENTICAL (0 problem(s), 39 cases)

