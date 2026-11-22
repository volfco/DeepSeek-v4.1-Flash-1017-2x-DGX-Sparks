# Engine Host & 0231 Multi-Chunk GPU Parity Results

## Commits on `fused-next` (`/home/colum/Workspace/tfref-fused-next`)

- **`e5110a0` (`0256`)** — `0256 dsv41: make fused_trace thread-safe, unify call-site arm dispatch and cover 0254/0255 (review S1, S4, S5)`
  - Thread-safe `fused_trace.zig` (`std.atomic.Mutex` + atomic `count`/`enabled_cache`), bounded at 128 distinct tuples, with concurrent race test across `count == 127`.
  - Single-source arm helpers (`idxArm`, `topkIndicesArm`, `candFastArm`, `compressArm`, `keyArm`, `latentArm`, `queryArm`, `pruneArm`) shared directly with `round.zig`, `prompt.zig`, `ops.zig`, and `exact.zig`.
  - Zero served-byte, launch-grid/arg, or CUDA-graph-key changes when `TF_DS_FUSED_TRACE` is unset.
- **`9ece091` (`0257`)** — `0257 dsv41: extract 0231 prompt source dispatch plan with multi-chunk tests and fix tile-prefix warp zeroing (review S9, S10, S12)`
  - Extracted `KvSource2Plan` + `planKvSource2` in `zig/src/families/dsv41/prompt.zig` and `gridRows` in `zig/src/families/dsv41/compress2_norm.zig` (called by `kvSourceUpdate` and `Exact.compress2Norm`), enforcing `max_kv_source_groups` (`ChunkTooLong`) consistently regardless of fusion flags.
  - Added Zig host tests covering multi-chunk progressions (`(0,50) -> (50,1) -> (51,19) -> (70,131)` and `(0,3) -> (4,65) -> (70,1) -> (70,129)`), partial-fusion fallbacks (`compress2_norm` vs `rope_store`, optional `idx_wk`, non-fused `id`/`hd` widths), and `ChunkTooLong`.
  - Fixed Review S9 stale `8192` cap comment in `zig/src/families/dsv41/idx_fused.zig` (`max_nbk` is `16,384`).
  - Fixed Review S12 `tf_ds_tile_prefix` `warp_end` zeroing in `zig/kernels/cuda/dsv41_ops.cu:817` (`(blockDim.x + 31) >> 5`).

## Host Build & Unit Test Validation

- `/home/colum/.local/zig-0.17.0/zig build` — **PASS** (0 errors, 0 warnings).
- `/home/colum/.local/zig-0.17.0/zig build test --summary all` — **630/632 passed** (`473/474` in `dsv41`, including `prompt.test.0231 planKvSource2 matches unfused and fused multi-chunk ring and group progressions`); only the 2 pre-existing baseline failures remain (`prompt_cache.test.0178` and `round.test.TF_DS_EXPERT_HOT_CACHE: hotPick`).

## Head `nvcc` & 4-Chunk 0231 GPU Byte-Parity (`100.95.49.21:~/tf-bench/pre-bench-20261121/p0231/`)

- **Compiler**: `/usr/local/cuda/bin/nvcc -cubin -O3 -arch=sm_121 --fmad=false --ftz=false` (CUDA 13.0, V13.0.88)
  - `bin/dsv41_ops.cubin`: SHA-256 `02cb1e260fa06be444964941d4a9440041a205729c7ecd553fc6764f56cd219e`
  - `bin/dsv41_torch.cubin`: SHA-256 `a55c9fee2d3bd3c380262bc97ab4fd9f92083fa154e7e56d491e221eea3b571c`
  - Kit AOT `/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins`:
    - `_fp4_store` (`D=128, BLOCK=32, E4M3=0, PDL=1`): `34543814a21ab766f744ba6c83c13b2ef1917d1c66013bb490900951665b8e03.cubin`
    - `_fp4_store` (`D=512, BLOCK=16, E4M3=1, PDL=1`): `9cd357f1c8586168ef430112df7d213db67ae95e6b240d40422b67bfd7b8643a.cubin`
- **Resource & Stream Safety**:
  - Ran under `flock -w 120 ~/tf-bench/gpu.lock`
  - `MemAvailable`: `5.39 GiB` start / `5.16 GiB` end (`>= 3.5 GiB` guard enforced)
  - Peak VRAM: **`4.419 MiB`** (`4,633,856` bytes, cap `256 MiB`)
  - Single `CU_STREAM_NON_BLOCKING` stream for all `cuMemcpyHtoDAsync_v2`, kernel launches, `cuMemcpyDtoHAsync_v2`, and `cuStreamSynchronize` (no stream-0 mixing); elapsed `0.642s`.
- **4-Chunk Stateful Sequence** (verified across 4 suites: `seed231_normal_pdl0`, `seed231_normal_pdl1`, `seed2026_wide_edge_pdl0`, `seed2026_wide_edge_pdl1`, comparing `unfused`, `fused`, `compress2_norm`-only, and `rope_store`-only arms + CPU modular-ring ground truth):

| Chunk | `(start, n)` | Edge Case Exercised | `full` | `keep` | `raw_first` | `raw_pos0` | `grid_rows` | `scatter` | Byte Diffs (all 7 buffers & 4 suites) | `seed231` SHA-256 (`raw_kv` / `lat` / `idx_codes` / `comp_codes`) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `(0, 50)` | Even initial chunk | 25 | 50 | 0 | 0 | 25 | 0 | **0** | `11da203fd61adaad` / `810c8797a23d6c38` / `389a6f8a87b5ca6d` / `c936aa66e7c6c44a` |
| 1 | `(50, 1)` | Single row (`full==0`, scatter fallback) | 0 | 1 | 0 | 50 | 1 | 1 | **0** | `d863495dbf490867` / `810c8797a23d6c38` / `389a6f8a87b5ca6d` / `c936aa66e7c6c44a` |
| 2 | `(51, 19)` | Odd + wraps `64` (`51..63,0..5`) + `rmsLanes=64` + trailing CTA `g=9` | 9 | 19 | 0 | 51 | 10 | 0 | **0** | `46163415fb04b2eb` / `72630eb76dba6bcd` / `f5f2c7b409bfcb89` / `91be9aa76f9b7cf6` |
| 3 | `(70, 131)` | Odd + `raw_first=67>0` (pair `g=33` split) + wraps `64` (`9..63,0..8`) + trailing CTA `g=65` | 65 | 64 | 67 | 9 | 66 | 0 | **0** | `7eaaccb6ec4341eb` / `2aef2193f5db3156` / `799979c5ade9d613` / `315b13c733c7ca08` |

- **Artifacts**:
  - Script: `/tmp/pre-bench-20261121/engine-host/verify_0231_chunks.py`
  - JSON report: `/tmp/pre-bench-20261121/engine-host/verify_0231_chunks_result.json` (`total_failures: 0`, `pass: true`)
