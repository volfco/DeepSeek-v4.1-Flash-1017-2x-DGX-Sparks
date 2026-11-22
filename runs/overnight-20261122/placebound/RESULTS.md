# 0264 TF_DS_PLACE_BOUND_ALL, November 22: exact, rounds −0.41%, more at four streams. Adopted.

One image (`bench-fused-next-27baa93` = `recipe-0264`: recipe-0261 + 0264), ABBA off/on/on/off 10:20–10:42 UTC, production's `.env` (0259/0260/0261 on) and the frozen corpus. **Exact: IDENTICAL 39/39 in all arms.** Rounds are −0.41% mean. The gain grows with rows, as an emptier grid should: at four streams −0.25% (4K), −0.91% (32K) and −2.15% (128K, both on arms below both controls); single-stream cells are flat within drift. No banner (engagement shows as the four-stream gain). Production moved 10:42–10:44 UTC ([deploy](../../deploy-0264-20261122/), validated 8/8) and `zig/patches/0264` was exported (`TF_REF + zig/patches = recipe-0264`, verified).

| Depth × streams | off1 | off2 | on1 | on2 | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.08 | 33.25 | 33.43 | 33.47 | 33.17 | 33.45 | **+0.85%** | +0.51% | 0/2 |
| 4K × 4 | 54.67 | 54.95 | 54.35 | 54.99 | 54.81 | 54.67 | **-0.25%** | +0.50% | 1/2 |
| 32K × 1 | 32.33 | 32.43 | 32.47 | 32.34 | 32.38 | 32.41 | **+0.09%** | +0.31% | 0/2 |
| 32K × 4 | 60.59 | 61.63 | 60.71 | 60.39 | 61.11 | 60.55 | **-0.91%** | +1.71% | 1/2 |
| 128K × 1 | 32.22 | 32.68 | 32.51 | 32.32 | 32.45 | 32.42 | **-0.11%** | +1.42% | 0/2 |
| 128K × 4 | 58.27 | 58.03 | 56.95 | 56.84 | 58.15 | 56.90 | **-2.15%** | -0.40% | 2/2 |

mean test vs control: -0.41%
exact off1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact off2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact on1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact on2: RESULT: IDENTICAL (0 problem(s), 39 cases)
