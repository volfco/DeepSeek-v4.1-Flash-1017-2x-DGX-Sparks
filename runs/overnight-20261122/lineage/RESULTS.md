# Production onto the recipe lineage with 0260/0261, November 22: exact everywhere, speed-neutral. Adopted.

Three ABBA pairs, 09:35–10:09 UTC, with production's `.env` and the frozen corpus in every arm. Production (`bench-fused-next-69335b8`: the fused-next lineage + 0259) against `bench-fused-next-086e8cb` (the recipe tree `TF_REF + zig/patches 0001–0259` + 0260 + 0261 + one-time engagement banners) with `TF_DS_CONF_PACK=1 TF_DS_TOPK_MERGE=1`.

- **Exact:** IDENTICAL 39/39 in all six arms. **Sampled** (three seeded waves with mixed top_k, plus solo): the new arms are identical to production's own sampled replies.
- **Engagement:** `TF_DS_CONF_PACK` and `TF_DS_TOPK_MERGE` banners on **both ranks in every new arm** (3 banners each, with 0259's).
- **Speed:** +0.30% mean against control drift up to 2.3%. The recipe gate's +3.2% at 128K × 4 does not reproduce (−0.69% here, 2 of 3 new arms below every production arm).
- **Decision (operator's adopt rule):** production moved at 10:11–10:13 UTC ([deploy](../../deploy-recipe-20261122/)). Production now runs **exactly what `zig/patches/` builds**: `recipe-0261` (0260/0261 with their banners folded in) has tree `bfc02e5`, identical to the image's `086e8cb`, and `scripts/export-fnext-series.sh` verified `TF_REF + zig/patches = recipe-0261`.
- **Not measured:** 0261's speed on sampled traffic. The round-time workload is greedy. The patch removes a 258 KB-per-row collective on T>0 rounds, a small expected gain.

| Depth × streams | prod1 | prod2 | prod3 | new1 | new2 | new3 | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 33.38 | 33.44 | 33.47 | 33.49 | 33.40 | 33.38 | 33.43 | 33.42 | **-0.02%** | +0.28% | 1/3 |
| 4K × 4 | 54.19 | 54.59 | 54.04 | 55.16 | 54.72 | 54.47 | 54.27 | 54.79 | **+0.94%** | -0.26% | 0/3 |
| 32K × 1 | 32.40 | 32.36 | 32.50 | 32.47 | 32.55 | 32.38 | 32.42 | 32.47 | **+0.13%** | +0.32% | 0/3 |
| 32K × 4 | 61.97 | 61.23 | 60.56 | 61.37 | 62.59 | 61.87 | 61.25 | 61.94 | **+1.13%** | -2.28% | 0/3 |
| 128K × 1 | 32.62 | 32.55 | 32.41 | 32.63 | 32.66 | 32.58 | 32.53 | 32.62 | **+0.29%** | -0.66% | 0/3 |
| 128K × 4 | 57.01 | 57.27 | 58.07 | 57.17 | 57.01 | 56.97 | 57.45 | 57.05 | **-0.69%** | +1.85% | 2/3 |

mean test vs control: +0.30%
exact prod1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact prod2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact prod3: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact new1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact new2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact new3: RESULT: IDENTICAL (0 problem(s), 39 cases)
