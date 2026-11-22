# Audit Report: (1) Patch 0254 Tied-Score Cost & Served Frequency and (2) `zig/patches/` 0213–0257 Lineage Gap

## Executive Summary

1. **0254 Tied-Score Audit (`todo.md` item 8):**
   - At `128 × 32,768 (k=512)`, `tf_ds_topk_indices_cut_kernel` takes **`100.717 µs`** on random scores (`2.9922` mean radix passes; `3.9922` global row scans = `63.875 MiB`; `33,531` shared `atomicAdd`s/row) versus **`138.788 µs`** on all `-∞` (`5.0000` radix passes `d = 7, 6, 5, 4, 1` — `d = 3, 2` skipped by `column_bytes = 2` and `d = 0` skipped because `k = 512 = 2 × 256` fills bins `0xFF` and `0xFE` completely; `6.0000` global row scans = `96.000 MiB`; `163,840` shared `atomicAdd`s/row, 100% single-address per warp).
   - **Decomposition of the `+38.071 µs` (`+37.8%`) gap:** Cross-calibrating `old` vs `new` GB10 timings (`old random = 237.022 µs`, `old flat = 227.199 µs` at equal 8 passes) and the `2,392 MHz` SM clock bound across 3 CTA waves (`48 + 48 + 32` CTAs at 1 CTA/SM) proves that `sm_121` hardware/NVCC already coalesces intra-warp uniform-address `atomicAdd`. **`~82–86%` (`~31–33 µs`)** of the `+38.071 µs` penalty comes from the **`+2.0078` extra radix passes (`+32.125 MiB` of row reads + `6` extra CTA barriers)**, while **`~14–18%` (`~5–7 µs`)** comes from cross-warp single-bank contention on `hist[0]` (`d=7`), `hist[127]` (`d=6`), and `hist[255]` (`d=5,4`).
   - **Served frequency under `TF_DS_IDX_FUSED=1`:** Cutoff-straddling `-∞` rows (`5–6` passes) are **virtually unreachable in served traffic (`< 0.03%` wall-clock impact everywhere)**:
     - **`applyPool`-masked rows (`prompt.zig:1559-1561`, layers 24/28/32/36, `len > 16,384`):** Retain `16,384` finite columns (`>> k = 512`). Cutoff never reaches `-∞`; rows resolve in **`2.8906` mean radix passes** (`62.25 MiB` at 32K), slightly *fewer* passes than unmasked random. Only Pass 1 (`d=7`) incurs `(n - 16,384)` `-∞` `atomicAdd`s to `hist[0]`, and bounded replay (`replay = len -| 128`) caps total execution across an entire 32K–300K prompt to `128` rows × 4 layers (`~0.4–2.5 ms` out of `15.7–167 s`, **`< 0.003%`**).
     - **Causal early prompt rows (`vis < 512`):** For `len <= 512`, patch `0253` (`prompt.zig:1470`) bypasses `topkIndices` entirely via `iotaVis`. For `len >= 640`, `replay = len - 128 >= 512`, so `0` rows have `vis < 512`. Only exact prompt lengths **`513 <= len <= 638`** send `639 - len` rows (`1..126` rows) with `vis < 512` at tiny width `n = 513..638` (`< 25 µs`/request total). Moreover, for *every* `0 <= vis < 512` with `k = 512`, `hist[0xFF] + hist[0xFE] == 512 - finite_count == want_s` holds identically at `d = 1`, so `d = 0` never runs (`5` passes exact).
     - **Deep fallback decode (`round.zig:1289`, layer 20, `nbk > 16,384`, `rd.bucket >= 262,144`, `k = 2,048`):** Never runs at context `<= 131,072` (`0195`/`0215`/`0217` handle `nbk <= 16,384`). Above `131,072`, homogeneous deep streams have `vis_nbk >= 16,385 >> 2,048` (`3.14` mean passes, `0%` cutoff ties). Only a **mixed-depth round** pairing a `> 131,072`-token stream with a concurrent `< 16,384`-token stream (`vis_nbk < 2,048`) triggers `5` passes (`6` at 1M) on the short stream's CTAs — once per round at layer 20 (`< 25 µs` out of `~80 ms`, **`< 0.03%`**).
   - **Verdict:** Close `todo.md` item 8 as **no kernel change needed** ahead of the paired Spark serving replay.

2. **Patch-Series Lineage Audit (`zig/patches/` vs `tfref-fused-next`):**
   - In a clean temporary worktree at `fbbc971`, **all 136 patches** in `zig/patches/*.patch` (`134` tracked `0001..0211` + `2` untracked `0254..0255`) pass `git apply --check` and apply cleanly in both `LC_ALL=C` and `en_US.UTF-8` filename orders.
   - `fbbc971` + `0001..0211` (`134` patches) reproduces `remotes/recipe/recipe-refold` (`c3b3859`, tree `e449cb68f56cc592c7972d0ab5201f62518bc2e9`) **byte for byte**.
   - **Yes, `0254` and `0255` apply cleanly onto `0211` without `0213–0253`** because the untracked files in `zig/patches/` were manually ported/re-contextualized onto `c3b3859` (including `0182`'s `topk_indices_fast` context lines), **not** raw exports of `tfref-fused-next` `b4f6d18`/`d0c9fb1`. Consequently, an image built from `zig/patches/` today has `0254/0255` on top of `0211`, **without** the `0213–0253` + `0256–0257` fused-next series (**16 commits missing**: 14 in `0213–0253` + 2 in `0256–0257`).
   - **Base divergence (`0182–0190zc`):** `tfref-fused-next` (`fused-next`) branched from `a149718` (`pinned` image `234f9080b326`, `0001..0171`), omitting all **12 recipe patches `0182..0190zc`** (`914+ / 166-` across 19 files at `0211`). Exporting `0213–0257` onto `zig/patches/` conflicts first at **`0217` (`d01975d`)** in `ops.zig` and introduces a **precedence/trace bug at `0256` (`e5110a0`)** against `0182`'s `TF_DS_TOPK_FAST`.

---

## Part 1: Patch 0254 Tied-Score Frequency & Radix-Pass Audit

### 1.1 Kernel & Key-Order Mechanics (`zig/kernels/cuda/dsv41_ops.cu:129-260`)

`tf_ds_score_key(float x, int j)` maps `(score, col)` into a strictly unique unsigned 64-bit key `u` (`-0.0f` canonicalized to `+0.0f`):
- Upper 32 bits (`d = 7, 6, 5, 4`): IEEE-754 sign-magnitude converted to unsigned total order (`NaN > +inf > finite > -inf`). Specifically, `x = -inf` (`0xFF800000`) maps to upper 32 bits **`0x007FFFFF`** (`d=7: 0x00`, `d=6: 0x7F`, `d=5: 0xFF`, `d=4: 0xFF`). Every finite float `x > -inf` has upper 32 bits `>= 0x00800000` (for normal scores in `[-16, +16]`, `d=7` lies in `0x3E..0xC1`).
- Lower 32 bits (`d = 3, 2, 1, 0`): inverted column `0xFFFFFFFFu - (unsigned)j` (ties break to lower column `j`).
- `tf_ds_topk_indices_cut_kernel` (`EarlyCut = true`, 1 CTA of 1,024 threads per row):
  1. Computes `column_bytes = (32 - __clz((unsigned)(n - 1)) + 7) / 8` (`2` for `257 <= n <= 65,536`, `3` for `65,537 <= n <= 16,777,216`), presets `prefix_s = mask_s = (0xffffffffull << (8 * column_bytes)) & 0xffffffffull`, and steps `d = 7, 6, 5, 4, column_bytes - 1, ..., 0`.
  2. Each pass reads **all `n` floats** (`row[i]`) from global/L2 memory, executes `atomicAdd(&hist[bin], 1u)` in shared memory for active keys `(u & mask) == prefix`, and uses Warp 0 (`8` descending bins per lane + 5-step `__shfl_up_sync`) to pick the cutoff bin `b` and residual `rank = target - above_count`.
  3. Sets `want_s = (rank == hist[b]) ? 0 : rank` and **breaks** when `want_s == 0`.
  4. Runs 1 final global/L2 row scan collecting the exact `k` columns with `u >= prefix_s`, bitonic-sorts `cols[0..p]` (`p = next_pow2(k)`), and writes `cols[i] < vis[r] ? cols[i] : -1`.

### 1.2 Exact Simulation Results (`sim_0254.py`, 100% Bit-Exact Parity Verified)

`/tmp/pre-bench-20261121/topk-patches/sim_0254.py` simulates `tf_ds_topk_indices<true>` and `<false>` per row and per 32-lane warp (asserting exact output equality against `np.argsort` on every row), including exact reproduction of `tools/fused_bench/bench.py:case_topk_indices` (`np.random.default_rng(254)`, `stride = width + 3`):

| Case & Shape | Radix Passes (hist) | Mean Passes | Row Reads | Logical Bytes Read (total) | Mean `hist` `atomicAdd`s / Row | Pass-0 Peak Single-Bin `atomicAdd`s | Warp-Serial Atomic Steps / Row |
|---|---|---:|---:|---:|---:|---:|---:|
| **(a) Random `N(0,1)`** (`128 × 32,768, k=512`, `seed=254`, **0254 new**) | `{2: 7, 3: 115, 4: 6}` | **2.9922** | 3.9922 | **63.875 MiB** (`511.0 KiB`/row) | **33,530.8** | 9,422.3 (`0xBF`/`0x3F`) | 11,736.8 |
| *(Ref) Random `N(0,1)` (`128 × 32,768, k=512`, old 8-pass)* | `{8: 128}` | *8.0000* | *9.0000* | *144.000 MiB* (`1,152.0 KiB`/row) | *33,536.8* | *9,422.3* | *11,742.8* |
| **(e) All `-∞` (`flat`)** (`128 × 32,768, k=512`, **0254 new**) | `{5: 128}` | **5.0000** | 6.0000 | **96.000 MiB** (`768.0 KiB`/row) | **163,840.0** | **32,768.0** (`hist[0]`) | **163,840.0** |
| *(Ref) All `-∞` (`flat`) (`128 × 32,768, k=512`, old 8-pass)* | `{8: 128}` | *8.0000* | *9.0000* | *144.000 MiB* (`1,152.0 KiB`/row) | *229,632.0* | *32,768.0 (`hist[0]`)* | *229,384.0* |
| **(b) `applyPool`-masked L24/28/32/36** (`128 × 32,768, k=512`, `16,384` finite + `16,384` `-∞`, `vis=32641..32768`) | `{2: 14, 3: 114}` | **2.8906** | 3.8906 | **62.250 MiB** (`498.0 KiB`/row) | **37,464.7** | **16,387.5** (`hist[0]`) | 18,310.1 |
| **(b′) `applyPool`-masked 100K** (`16 × 100,000, k=512`, `16,384` finite + `83,616` `-∞`) | `{2: 2, 3: 14}` | **2.8750** | 3.8750 | **23.651 MiB** (`1,513.7 KiB`/row) | **104,690.1** | **83,616.6** (`hist[0]`) | 85,233.4 |
| **(c1) Hypothetical causal `vis < 512` at 32K** (`128 × 32,768, k=512`, `vis=0..511`, post-`vis` `-∞`) | `{5: 128}` | **5.0000** | 6.0000 | **96.000 MiB** (`768.0 KiB`/row) | **162,820.0** | **32,513.0** (`hist[0]`) | 162,648.2 |
| **(c2) Actual served `vis < 512` (`L = 513`)** (`128 × 513, k=512`, `vis=386..513`) | `{1: 1, 2: 1, 5: 126}` | **4.9453** | 5.9453 | **1.489 MiB** (`11.9 KiB`/row) | **767.0** | 134.8 | 466.6 |
| **(c3) Exhaustive `vis = 0..511`** (`512 × 2,048, k=512`, post-`vis` `-∞`) | `{5: 512}` | **5.0000** | 6.0000 | **24.000 MiB** (`48.0 KiB`/row) | **9,218.0** | 1,792.5 (`hist[0]`) | 9,046.0 |
| **(c4) Unmasked post-`vis` (`vis < 512`, finite post-`vis`)** (`128 × 32,768, k=512`) | `{2: 3, 3: 120, 4: 5}` | **3.0156** | 4.0156 | **64.250 MiB** (`514.0 KiB`/row) | **33,534.8** | 9,420.4 | 11,731.3 |
| **(d1) Deep decode fallback, homogeneous deep** (`7 × 32,768, k=2,048`, `262K` bucket, `pos ~ 150K`) | `{3: 6, 4: 1}` | **3.1429** | 4.1429 | **3.625 MiB** (`530.3 KiB`/row) | **36,780.0** | **14,014.0** (`hist[0]`) | 25,017.6 |
| **(d2) Deep decode fallback, mixed `150K + 3K`** (`14 × 32,768, k=2,048`: 7 deep + 7 short `vis_nbk=375`) | `{3: 7, 5: 7}` | **4.0000** (`5` on short) | 5.0000 | **8.750 MiB** (`640.0 KiB`/row) | **99,582.5** (`162,343` short) | **23,205.0** (`32,393` short) | 93,600.2 |
| **(d3) Deep decode fallback, `1M` bucket mixed** (`8 × 131,072, k=2,048`: 7 deep + 1 short `vis_nbk=375`) | `{3: 6, 4: 1, 6: 1}` | **3.5000** (`6` on short) | 4.5000 | **18.000 MiB** (`2,304.0 KiB`/row) | **218,590.9** (`784,548` short) | **65,399.2** (`130,697` short) | 177,363.5 |

**Key structural invariant discovered for `k = 512` (and `k = 2,048`) on causal `-∞` ties:**
Whenever `vis < k` and `k % 256 == 0` (`512` or `2,048`), every column `j >= k` is causally `-∞`. If `F <= vis < k` columns in `[0, k)` have score `> -∞`, then `want_s` entering the column-byte passes is `k - F`, and the number of `-∞` columns in `[0, k)` (`d = 1` bins `0xFF .. (0x100 - k/256)`) is **identically `k - F`**. Thus `rank == hist[b]` (`want_s = 0`) triggers **100% of the time at `d = 1`**, skipping `d = 0` (`5` passes for `n <= 65,536`, `6` passes for `n > 65,536`).

### 1.3 Quantitative Breakdown: Why All-`-∞` Takes `138.788 µs` vs `100.717 µs` Random

From `runs/suite-fused-next-20261121/kernel-review/{index-timing.json,sass.json}` on GB10 (`sm_121`, `48` SMs, `2,392 MHz`, `REG=36`, `SHARED=18,464 B`, `blocks_per_sm = 1` at 1,024 threads -> `128` rows execute in **3 waves** of `48 + 48 + 32` CTAs, empty-launch floor = `0.807 µs`):

1. **Proof that intra-warp `atomicAdd` is hardware/compiler warp-coalesced on `sm_121`:**
   - In `all -∞`, every CTA executes `163,840` shared `atomicAdd`s (`32,768` per pass across 5 passes: `hist[0]` at `d=7`, `hist[127]` at `d=6`, `hist[255]` at `d=5,4`, and one `hist[0x80..0xFF]` per 32-lane warp at `d=1`, because `i = 1024*t + 32*w + lane` shares `(0xFFFF - i) >> 8` across all 32 lanes of a warp).
   - If 32 lanes serialized into 32 sequential shared-memory cycles per warp, 3 CTA waves would require `3 × 163,840 = 491,520` cycles = **`205.5 µs` at `2.392 GHz`** on `new flat` (and `3 × 229,632 = 688,896` cycles = **`288.0 µs`** on `old flat`), strictly exceeding the measured total kernel durations (`138.788 µs` and `227.199 µs`).
   - Furthermore, at an **identical 8 passes (`9` row scans = `144.0 MiB`)**, `old flat` (`227.199 µs`, `229,632` single-bin atomics/row) was **`9.823 µs` faster** than `old random` (`237.022 µs`, `33,537` atomics/row), because `nvcc -O3 -arch=sm_121` + Blackwell SM hardware coalesces uniform-address intra-warp `ATOMS.ADD` into 1 warp-level update, while constant `0xFF800000` cachelines benefit from zero warp branch divergence and Blackwell L2 sector compression.

2. **Exact cost attribution for `128 × 32,768 (k=512)`:**
   - **Per-pass cost calibration from `index-timing.json`:**
     - `old random -> new random` (`237.022 -> 100.717 µs`): removes `5.0078` zero-atomic passes + old lane-0 serial 256-bin loop (`~1.5 µs`/pass) = **`27.22 µs`/pass** (`~25.7 µs` per random `16 MiB` L2/DRAM row scan + barriers).
     - `old flat -> new flat` (`227.199 -> 138.788 µs`): removes `3.0000` passes (`d=3, 2` with `32,768` single-bin atomics + `d=0` with `256` atomics) = **`29.47 µs`/pass** (`~22.3 µs` compressed row scan + barriers + `~5–7 µs` cross-warp single-bank contention among the CTA's 32 warps hitting `hist[0]`/`hist[127]`/`hist[255]`).
   - **Component table (`new random` vs `new flat`):**
     - Fixed non-radix work (launch floor `0.81 µs` + final `16 MiB` collection scan `~22.5 µs` + 512-element bitonic sort & `vis` mask store `~4.0 µs`): **`~27.3 µs`** in both arms (`~27.8 µs` on `flat` where all 512 `count_s` atomics concentrate in `i < 512`).
     - Radix loop on `random` (`2.9922` passes, `47.875 MiB` radix reads, `33,531` atomics concentrated in Pass 1): **`~73.4 µs`** -> total **`100.72 µs`**.
     - Radix loop on `all -∞` (`5.0000` passes, `80.000 MiB` radix reads, `163,840` single-bin warp-coalesced atomics across 5 passes): **`~111.0 µs`** (`~73.4 µs` baseline + **`~31.5 µs` from `+2.0078` extra row-scan passes (`+32.125 MiB` + `6` `__syncthreads()`)** + **`~6.1 µs` net cross-warp single-bank serialization** on `hist[0]`/`hist[127]`/`hist[255]`) -> total **`138.79 µs`**.

### 1.4 Exact Served Frequency Across All Engine Call Sites (`TF_DS_IDX_FUSED=1`)

Inspecting every caller of `Ops.topkIndices` across `zig/src/families/dsv41/{prompt.zig,round.zig}`:

| Call Site | Layers & Condition (`TF_DS_IDX_FUSED=1`) | Served Rows & Width | Cutoff Straddles `-∞` (`5–6` passes)? | Pass-1 `hist[0]` `-∞` Contention? | Served Wall-Time Share |
|---|---|---|---|---|---|
| `prompt.zig:1470` (`0253` bypass) | `li >= 20` or non-pow2 `li < 20`, `n_comp_end <= 512` | Prompts `len <= 512` | **`topkIndices` NOT called** (`iotaVis` runs) | None | `0%` |
| *(Keyed prompt)* `prompt.zig:1542` | `li in {2, 8, 14}` (`len > 512` or pow2 `<= 512`) | All encoder chunks | **`topkIndices` NOT called** (`topkI64` runs) | None | `0%` |
| `prompt.zig:1561` (early causal edge) | `li in {20, 24, 28, 32, 36}`, **`513 <= len <= 638`** | `128` replay rows (`639 - len` rows have `vis < 512`), `n = 513..638, k = 512` | **Yes, on `639 - len` rows** (`5` passes, tiny `n <= 638`) | Minor (`<= 252` `-∞`/row) | `< 25 µs` / prompt (`< 0.01%`) |
| `prompt.zig:1561` (unpooled mid) | `li in {20, 24, 28, 32, 36}`, **`639 <= len <= 16,384`** | `128` replay rows (`replay = len - 128`), `vis >= 512, k = 512` | **0% of rows** (`~3.0` passes) | Negligible (`<= 127` trailing `-∞`/row) | `< 0.01%` |
| `prompt.zig:1561` (pooled L20) | `li == 20`, **`len > 16,384`** (32K, 100K, 300K) | `128` replay rows total per prompt, `vis >= len - 127` | **0% of rows** (`~3.0` passes) | Negligible (`<= 127` trailing `-∞`/row) | `< 0.002%` |
| `prompt.zig:1561` (`applyPool` L24–36) | `li in {24, 28, 32, 36}`, **`len > 16,384`** | `128` replay rows total per prompt (`16,384` finite + `n - 16,384` `-∞`) | **0% of rows** (`2.89` passes; `16,384` finite `>> k = 512`) | **100% of rows in Pass 1 (`d=7`) only** (`16,384` at 32K; `83,616` at 100K) | `~0.4 ms` at 32K (`0.0025%`); `< 2.5 ms` at 300K (`< 0.002%`) |
| `prompt.zig:1554` (deep prefill pool) | `li == 20` only, **`len > 131,072`** (`nbk > 16,384`) | `128` replay rows, `nbk > 16,384, k = 2,048`, `vis_nbk > 16,368` | **0% of rows** (`~3.1` passes) | Negligible (`<= 16` `-∞` blocks) | `< 0.001%` |
| `round.zig:1289` (decode `<= 131K`) | `li == 20`, `rd.bucket <= 131,072` (`nbk <= 16,384`) | All `<= 100K` decode buckets | **`topkIndices` NOT called** (`idx_small` / `idx_fused` runs) | None | `0%` |
| `round.zig:1289` (deep decode `> 131K`) | `li == 20` only, **`rd.bucket in {262K, 524K, 1M}`** (`nbk in {32K, 64K, 128K}, k = 2,048`) | `R` round rows (`1..48`), once per round at layer 20 | **0%** if all streams have `pos >= 16,384` (`3.14` passes); **`5` passes (`6` at 1M) ONLY on rows of a co-batched short stream with `pos < 16,384`** | Pass 1 sees `nbk - vis_nbk` bucket-padding `-∞`s | `37.6–48.5 µs` homogeneous (`~0.05%`); `+< 25 µs` mixed (`< 0.03%` of `~80 ms` round) |

*(Note on `todo.md` optimization ideas: manual `__match_any_sync` histogram warp-aggregation would not remove the 2 extra passes and risks slowing the dominant 3-pass random/pooled case because `sm_121` already warp-coalesces uniform shared atomics. If deep mixed-batch or `applyPool` Pass-1 cost ever mattered, stopping radix selection at `d = 4` and resolving cutoff ties by ascending column scan — as `0214` `tf_ds_pool_cut` already does — or skipping `x == -inf` in `d = 7` when `vis >= k` would address the actual mechanism, **neither of which is warranted at `< 0.03%` served share**.)*

---

## Part 2: `zig/patches/` 0213–0257 Patch-Series Lineage Audit

### 2.1 Scratch-Worktree `git apply --check` Verification onto `fbbc971`

Tested in a temporary detached git worktree `/tmp/pre-bench-20261121/topk-patches/scratch-apply` from `~/Workspace/tfref-fused-next` at `fbbc97132cd004237d9887b8d3185896efaedc5c` (worktree cleanly removed and verified via `git worktree list`):

- **Patch inventory in `~/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/zig/patches/`:**
  - `134` tracked patches (`0001-c7-indexer-split.patch` .. `0211-dsv41-tf-ds-prune-prefix-early-out.patch`): `116` in `0001..0171` + `12` in `0182..0190zc` + `6` in `0192..0211` (`0192`, collapsed `0195`, `0203`, `0204`, `0205`, `0211`; `0206–0208` are parked in `zig/patches-parked/`).
  - `2` untracked patches (`??`): `0254-dsv41-topk-cutoff.patch` and `0255-dsv41-candidate-mask-loads.patch` (plus modified `M zig/patches/README.md`).
- **Sequential `git apply --check && git apply` result:**
  - **136 / 136 patches (`0001..0211` + `0254..0255`) pass `git apply --check` and apply with 0 failures** in both `LC_ALL=C` byte order (used by `port-patches.sh` and container `C`/`C.UTF-8` globs) and `LC_ALL=en_US.UTF-8` collation order (where `0190z-` sorts after `0190zc-`). Only two pre-existing trailing-blank-line warnings occur (`0065:3454`, `0142:509`).
  - Tree hash after `0001..0211` (`134` patches): `e449cb68f56cc592c7972d0ab5201f62518bc2e9` — **byte-for-byte identical** to `remotes/recipe/recipe-refold` (`c3b3859`).
  - Tree hash after `+ 0254 + 0255` (`136` patches): `5ccb9b6e614172c898535823944d9fcd54709228`.

### 2.2 Why `0254` and `0255` Apply Cleanly onto `0211` Without `0213–0253`

Comparing `git -C ~/Workspace/tfref-fused-next show b4f6d18` / `d0c9fb1` against `zig/patches/0254-dsv41-topk-cutoff.patch` / `0255-dsv41-candidate-mask-loads.patch` shows that the two untracked patch files were **purpose-built against the recipe tree (`c3b3859` = `fbbc971 + 0001..0211`)**, not raw `git format-patch` exports from `fused-next`:
- `zig/patches/0254-dsv41-topk-cutoff.patch` carries `0182`'s surrounding context (`// TF_DS_TOPK_FAST (tf_ds_topk_indices_fast_kernel)` in `dsv41_ops.cu` and `.topk_indices_fast = try fm.function(...)` in `ops.zig:177`), neither of which exists in `tfref-fused-next`.
- `zig/patches/0255-dsv41-candidate-mask-loads.patch` targets `dsv41_ops.cu:885` and `ops.zig:111,177,361` (matching `0211 + 0254` with `0182`/`0186` present), whereas `d0c9fb1` on `fused-next` targets `dsv41_ops.cu:680` and `ops.zig:106,168,325` (where `0213–0253` are present and `0182–0190zc` are absent).
- **Consequence:** Building the Docker recipe right now produces `0001..0211 + 0254..0255` **missing the entire `0213–0253` + `0256–0257` fused-next optimization & review-fix series** (`23 files, +959 / -1,095 lines` diff between `5ccb9b6e` and `d0c9fb1`). Furthermore, once `0213–0253` are exported into `zig/patches/`, `0254` and `0255` will have to be regenerated on top of `0253`.

### 2.3 Complete Lineage Map & Exact Commits Missing from `zig/patches/`

Comparing `git -C ~/Workspace/tfref-fused-next log --oneline fbbc971..9ece091` (`39` commits in `fbbc971..d0c9fb1`, plus `2` post-`d0c9fb1` commits `0256`–`0257` = `41` commits total) against `zig/patches/`:

#### A. Base & Pre-0213 Commits (`fbbc971..3c54a9d`, 23 commits -> represented in `0001..0211`)
- `864055d` (`TF_REF source...`): tree-identical to `fbbc971` (`0` file diffs).
- `a149718` (`pinned image 234f9080b326: the 117 patches...`): squashed `0001..0171` base (differs from current `fbbc971 + 0001..0171` by `4+ / 5-` lines across `cuda.zig`, `http_body.zig`, `reply_text.zig`, and **omits all 12 `0182..0190zc` patches**; see §2.4).
- `c5d9904`, `311f93a`, `4fadc54`, `2039a60` (4 commits): exported squashed as `0192-dsv41-tf-ds-draft-block.patch`.
- `3e1f002` (0193+0194), `0b0305a` (0195), `6cd5cac` (0197), `e3eb746` (0198), `ed8276f` (audit port), `d66e663` (0199) + `040880b` (0200), `10c1802` (0201), `a926d85` (0202), `cfe65df` (0209), `e46bd4e` (0210), `0c373a0` (0212) (12 commits): folded into `0195-dsv41-tf-ds-fused-paths.patch`.
- `ecd0ef0` (`0203`), `12b5d13` (`0204`), `9e79f6b` (`0205`), `3c54a9d` (`0211`) (4 commits): exported 1-for-1 as `0203`, `0204`, `0205`, `0211`.

#### B. Exact Commits Missing from `zig/patches/` (**16 commits total**: 14 in `0213–0253` + 2 in `0256–0257`)

*(Note: commit numbers `0218–0229`, `0232–0239`, and `0243–0249` were never assigned; WP1 used `0213–0217`, WP2 `0230–0231`, WP3 `0240–0242`, WP4/review `0250–0253`.)*

| # | Commit SHA | Patch # | Commit Subject | 3-Way Cherry-Pick onto `0211` (`c3b3859`) |
|---:|---|---|---|---|
| 1 | `7d32cab` | `0213` | `dsv41: 0195's served block size takes a static, vectorized phase 1 (items 4-5)` | Clean |
| 2 | `338be16` | `0214` | `dsv41: 0195's mask-only arm skips compaction and the cutoff's tied columns resolve during compaction (item 6)` | Clean |
| 3 | `0b78bec` | `0215` | `dsv41: 0195's small-row form spreads the pool over block_max's grid (small-row research item)` | Clean |
| 4 | `b7dcdbc` | `0216` | `dsv41: prompt chunks whose compressed keys all fit skip the indexer's projection, scoring and selection (item 6)` | Clean |
| 5 | `d01975d` | `0217` | `dsv41: 0195's pool covers 131,072 columns through the opt-in dynamic shared limit (item 7)` | **CONFLICT** in `ops.zig` (`0186`/`0190zb` `Ops.load` tail) |
| 6 | `90052a0` | `0230` | `dsv41: 0197's decode compressor is one kernel per served width, with the partner slots read ahead of the current row (item 1)` | Stacked |
| 7 | `b28310e` | `0231` | `dsv41: 0197's prompt source step folds the raw-ring tails and the position setup into the fused launches (item 5)` | Touches `prompt.zig` (check `0183` overlap context) |
| 8 | `cfa54ab` | `0240` | `dsv41: 0198's FP4 scale byte comes from the BF16 bits, applied by an exact reciprocal, with a warp integer maximum for the 32-element groups (todo items: E8M0 scale from BF16 bits, exact reciprocal multiply, warp integer maximum)` | Stacked |
| 9 | `50a70e1` | `0241` | `dsv41: the decode query's geometry follows the row count - one element a thread below decode's largest window, a warp a head with the rotary pair inside one lane from it up (todo item: the R=32/48 geometry regression)` | Stacked |
| 10 | `9d4232a` | `0242` | `dsv41: 0198's latent chain skips the rotary partner exchange on wholly unrotated warps (todo item: skip RoPE partner shuffles on unrotated warps)` | Stacked |
| 11 | `f013282` | `0250` | `dsv41: PDL entry splits around the fixed norm weights in the decode compressor and the indexer key (todo item: split PDL entry around proven independent work)` | Stacked |
| 12 | `59e71a8` | `0251` | `dsv41: an opt-in fused-arm trace at the launches, with the served matrix as a host test (todo item: prove engagement)` | Conflicts in `ops.zig` (`0182`/`0186`) |
| 13 | `5110596` | `0252` | `dsv41: 0241's pair query geometry rounds the odd rotary element in the kit's contraction, fma(xe, sn, xo*cs) (review B1)` | Stacked (**critical correctness fix for `0241`**) |
| 14 | `d5ecba0` | `0253` | `dsv41: 0216's prompt bypass rides TF_DS_IDX_FUSED, skips the keyed layers and refuses a coexisting pool (review S1-S3)` | Stacked (**critical gate fix for `0216`**) |
| — | *(`b4f6d18`)* | *(`0254`)* | *`dsv41: stop score top-k at a complete radix bin under TF_DS_IDX_FUSED`* | *Present untracked (re-contextualized onto `0211`)* |
| — | *(`d0c9fb1`)* | *(`0255`)* | *`dsv41: vectorize aligned eight-score decode candidate masks under TF_DS_IDX_FUSED`* | *Present untracked (re-contextualized onto `0211+0254`)* |
| 15 | `e5110a0` | `0256` | `dsv41: make fused_trace thread-safe, unify call-site arm dispatch and cover 0254/0255 (review S1, S4, S5)` | **CONFLICT + precedence bug** vs `0182` `TF_DS_TOPK_FAST` (see §2.4) |
| 16 | `9ece091` | `0257` | `dsv41: extract 0231 prompt source dispatch plan with multi-chunk tests and fix tile-prefix warp zeroing (review S9, S10, S12)` | Stacked on `0231`/`0256` |

### 2.4 The `0182–0190zc` Base Difference & Export Hazards

`zig/patches/` contains **12 patches between `0171` and `0192`** (`19 files, +914 / -166 lines` in `git diff 3c54a9d c3b3859`) that are **present in `zig/patches/` (`remotes/recipe/recipe-refold`) but absent from `tfref-fused-next` (`fused-next`)** (`0187` does not exist):
1. `0182-dsv41-tf-ds-topk-fast-off-by-default-selects-the.patch` (`TF_DS_TOPK_FAST`: `dsv41_ops.cu`, `ops.zig`, `native.zig`)
2. `0183-dsv41-tf-ds-prefill-gather-overlap-off-by-defaul.patch` (`TF_DS_PREFILL_GATHER_OVERLAP`: `prompt.zig`, `model.zig`, `native.zig`)
3. `0184-dsv41-tf-ds-idx-bucket-finer-off-by-default-size.patch` (`TF_DS_IDX_BUCKET_FINER`: `round.zig`, `model.zig`, `native.zig`)
4. `0185-dsv41-tf-ds-bf16-partials-off-by-default-lossy-r.patch` (`TF_DS_BF16_PARTIALS`: `prompt.zig`, `round.zig`, `tri_hc.zig`, `model.zig`)
5. `0186-dsv41-tf-ds-greedy-argmax-off-by-default-exchang.patch` (`TF_DS_GREEDY_ARGMAX`: `dsv41_ops.cu`, `ops.zig`, `round.zig`, `lanes.zig`, `sampling.zig`)
6. `0188-dsv41-tf-ds-head-last-chunk-off-by-default-skips.patch` (`TF_DS_HEAD_LAST_CHUNK`: `model.zig`, `lanes.zig`, `prompt.zig`)
7. `0189-core-a-test-pins-the-draft-chain-s-granted-slack.patch` (`core/lanes/depth.zig`, `lane_host_robust_test.zig`)
8. `0190-dsv41-tf-ds-snap-trim-off-by-default-keeps-no-wi.patch` (`TF_DS_SNAP_TRIM`: `spill_state.zig`, `prompt_cache.zig`, `model.zig`, `lanes.zig`)
9. `0190z-dsv41-build-rank-and-checkpoint-fixes.patch` (`cuda.zig`, `engram_io.zig`, `lanes_main.zig`, `layer_main.zig`, `parts.zig`)
10. `0190za-dsv41-host-test-fixtures.patch` (`prompt_cache_test.zig`, `lanes.zig`)
11. `0190zb-dsv41-greedy-rank-major.patch` (`dsv41_ops.cu`, `ops.zig`, `round.zig`, `lanes.zig`)
12. `0190zc-dsv41-bf16-dispatch-and-precision-agreement.patch` (`rank_identity.zig`, `lanes.zig`, `model.zig`, `prompt.zig`, `round.zig`)

**Critical hazards when rebasing/exporting `0213–0257` onto `c3b3859` (`zig/patches/`):**
1. **`0182` (`TF_DS_TOPK_FAST`) vs `0256` (`e5110a0`) Precedence & Trace Mismatch:**
   - In `zig/patches/` (`0182` + `0254`), `ops.zig` launches `const f = if (o.topk_fast) o.topk_indices_fast else o.topk_indices;` while `.topk_indices` holds `tf_ds_topk_indices_cut_kernel` under `TF_DS_IDX_FUSED=1`. Thus **`TF_DS_TOPK_FAST` has precedence over `TF_DS_IDX_FUSED`**, as explicitly documented in `zig/patches/README.md` and `kernel-review/RESULTS.md`.
   - In `tfref-fused-next` `0256` (`e5110a0`), `fused_trace.zig:124-127` defines `topkIndicesArm(idx_fused_on, topk_fast)` with **`if (idx_fused_on) return .topk_indices_cut;`** (reversing precedence!) and `ops.zig:169,303` passes literal `false` for `topk_fast` because `o.topk_fast` does not exist on `fused-next`.
   - When porting `0256` onto `c3b3859`, `topkIndicesArm` and `Ops.load`/`Ops.topkIndices` **must be reconciled** (`if (topk_fast) return .topk_indices_fast;` first, passing `o.topk_fast` in `ops.zig:303`) so `TF_DS_FUSED_TRACE` never logs `.topk_indices_cut` while `ops.zig` actually launches `tf_ds_topk_indices_fast_kernel`.
2. **`0217` (`d01975d`) & `0251` (`59e71a8`) `ops.zig` Hunks:** Must include `0182`'s `topk_indices_fast` / `topk_fast` and `0186`/`0190zb`'s `argmax_local` fields inside `Ops` and `Ops.load`.
3. **Regenerate `0254` & `0255` in sequence:** Once `0213–0253` are rebased onto `c3b3859` (recommended: create a branch `recipe-fused-next` from `c3b3859`, cherry-pick `7d32cab..9ece091` resolving `0217`/`0251`/`0256`, run `zig build test`, and export `0213..0257` in order), replace the standalone `0254`/`0255` patches so the full `0001..0257` chain reproduces the tested engine tree byte for byte.

## Validation Artifacts
- Simulation script: `/tmp/pre-bench-20261121/topk-patches/sim_0254.py`
- Simulation output JSON: `/tmp/pre-bench-20261121/topk-patches/sim_0254_results.json`
