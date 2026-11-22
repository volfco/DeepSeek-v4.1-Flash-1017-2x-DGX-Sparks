# Recipe-tree gate, November 22: TF_REF + zig/patches 0001–0259 is exact on hardware

The first hardware run of the rebased recipe tree (`recipe-0259` `a5a5822`, image `bench-fused-next-a5a5822`, a full source build) against production (`bench-fused-next-69335b8`). Every arm used production's `.env` (`TF_DS_KV_FETCH_REUSE=1`) and the exact panel ran `--like` [kvreuse-ab off1](../../kvreuse-ab-20261122/off1.exact.json). prod1 ran 08:02–08:06 UTC. A runner bug (ssh swallowed the rest of the arms file) skipped the others, which were rerun 08:09–08:24 UTC. Production was restored and validated (all passed).

**Exact: IDENTICAL 39/39 in all four arms.** The rebased chain reproduces production's outputs. **Speed: neutral overall (+0.12%), but 128K × 4 is +3.2% in both recipe arms** (58.88/58.78 vs 57.34/56.71 ms, control drift −1.1%). This is possibly real and unattributed. The rebased tree carries the recipe's 0182–0211 forms where the fused-next lineage carries prune-six's, not just the two hand-resolved conflicts. Use the recipe image, not production, as the control for patches built on it (0260+). Banners: reuse engaged on both ranks in every arm. The rank-0 "error" count of 1 in the prod arms is the startup smoke request's known `CudaFailed` (prompt=8, tokens=1).

| Depth × streams | prod1 | prod2 | rec1 | rec2 | control mean | test mean | test vs control | control drift | test below all controls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 29.55 | 29.72 | 29.63 | 29.62 | 29.64 | 29.62 | **-0.04%** | +0.58% | 0/2 |
| 4K × 4 | 56.50 | 57.53 | 57.09 | 56.58 | 57.02 | 56.83 | **-0.32%** | +1.83% | 0/2 |
| 32K × 1 | 32.75 | 33.11 | 32.98 | 32.70 | 32.93 | 32.84 | **-0.27%** | +1.09% | 1/2 |
| 32K × 4 | 63.17 | 63.02 | 62.35 | 61.73 | 63.09 | 62.04 | **-1.67%** | -0.23% | 2/2 |
| 128K × 1 | 32.94 | 33.02 | 32.91 | 32.93 | 32.98 | 32.92 | **-0.17%** | +0.24% | 2/2 |
| 128K × 4 | 57.34 | 56.71 | 58.88 | 58.78 | 57.02 | 58.83 | **+3.17%** | -1.11% | 0/2 |

mean test vs control: +0.12%
exact prod1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact prod2: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact rec1: RESULT: IDENTICAL (0 problem(s), 39 cases)
exact rec2: RESULT: IDENTICAL (0 problem(s), 39 cases)
