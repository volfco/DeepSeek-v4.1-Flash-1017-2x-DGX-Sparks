# Review Blocker B2 Investigation Report (`runs/pre-bench-20261121/b2-investigation`)

## Executive Summary & Verdict

**Root cause established, reproduced on GB10 hardware, and fixed in `tools/fused_bench/bench.py`.**

The two one-off `new`-arm parity mismatches recorded once during WP2 (`runs/fused-next-20261121/wp2/wp2-full-parity.json`, 2026-11-21 11:02:44 UTC) —
1. `0195-idx-small` (`idx-small R=1 nb=8192 k=2048 random`): `new_vs_ref = 3,984` differing bytes (`first 4`), while `fused_vs_ref = 0` and `old_vs_ref = 0`;
2. `0197-compress-decode` (`compress-decode R=32 D=512 mixed`): `new_vs_ref = 3,707` differing bytes (`first 29952`), while `old_vs_ref = 0` —

were **not** kernel bugs in `0215` (`tf_ds_idx_pick_kernel`) or `0230` (`tf_ds_compress2x_norm512_kernel`). Both were caused by a **CUDA stream-ordering DMA race inside `tools/fused_bench/bench.py`**:

- `bench.py` creates its benchmark stream with `cuStreamCreate(ref(stream), 1)` (`CU_STREAM_NON_BLOCKING`), which explicitly disables implicit synchronization with the legacy default stream (`stream 0`).
- `Buffer.__init__` previously uploaded initial/poisoned host arrays (`out(...)` fills with `0x55`) via `cuMemcpyHtoD_v2` on **legacy stream 0** without synchronizing before `launch(..., stream)`. For pageable host transfers **`<= 64 KiB`**, the CUDA driver stages the payload into an internal pinned buffer and **returns to Python before the stream-0 copy-engine DMA completes**.
- Likewise, `Buffer.read` and `Buffer.readarr` previously issued `cuMemcpyDtoH_v2` on **legacy stream 0** without first synchronizing `stream`.
- When the GPU copy engine briefly lagged behind the compute queue (during WP2's overlapping `bench.py` processes while production served), stream 0's trailing `<= 64 KiB` poison-fill DMA (`32,768 B` for `idx-small` `bmax`; `33,792 B` for `compress-decode` `dst`) raced concurrently with or landed *after* the kernel executing on `CU_STREAM_NON_BLOCKING` `stream`.

**Production engine immunity:** The TensorFold Zig runtime (`zig/src/families/dsv41/prompt.zig:689`) stages all runtime host uploads and fills onto `e.s.handle` via `cuMemcpyHtoDAsync_v2` / `cuMemsetD8Async` / `cuMemsetD32Async` (explicitly noting in its doc comment that the non-blocking engine stream does not wait for legacy stream-0 copies). Only load-time weight initialization uses `DeviceBuffer.upload` (`cuMemcpyHtoD_v2`). Production serving was never exposed.

---

## (a) Forensic Byte-Count Analysis of Original `wp2-full-parity.json` Failures

Artifact inspected: `runs/fused-next-20261121/wp2/wp2-full-parity.json` (SHA-256 `316919f6b133f441849f45614b1f27089c808b3337034c1097dd9e2faaf53532`) and `wp2-full-parity.log` (SHA-256 `5255309860974e57e69e0caf58660bb0b9e8f2b1d520922af098f38fd73dcad6`), written 2026-11-21 11:02:44 UTC (`build.sh` completed 11:01:25 UTC, ruling out a mid-run cubin rebuild). Reconstructing the exact `np.random.default_rng(197)` state through all preceding harness cases (`reconstruct_wp2_seed197()` in `b2_probe.py`) accounts for both failure signatures down to the exact byte count and first differing offset:

| WP2 Row | Case Label | Failing Arm | Raced `out(...)` Buffer | Buffer Size (`<= 64 KiB`?) | Exact Race Mechanism | Reconstructed Byte Signature |
|---:|---|---|---|---:|---|---|
| `#31` | `idx-small R=1 nb=8192 k=2048 random` | `new` (`i=3`: `block_max` + `idx_pick`) | `bmax = out((1, 8192), "f4")` (`float32(85.0)`) | `32,768 B` (**Yes**) | `bmax` is the **last** `Buffer` allocated before `chain()` launches `tf_ds_block_max_kernel` (`grid=(32,1)`, `~1.5 us`) on `stream`. Stream 0's trailing `128 B`-cacheline DMA of `float32(85.0)` into `bmax[7136:8192]` (`33 x 128 B = 4,224 B`) overwrites `block_max`'s `N(0,1)` maxima (`max ~ 4.3`) before `idx_pick` reads `bmax`. (`vis = [0]` gives `last = -1`, so no block is pinned to `+inf`.) All `1,056` corrupted tail blocks `7136..8191` win selection, leaving `992` slots for genuine blocks `0..7135`. | Reference `cblk[:6] = [6, 8, 9, 18, 19, 23]`. Column `6` (`bmax[6] = 2.401343`, rank `521 < 992`) survives at `cblk[0]` (bytes `0..3` match); column `8` (rank `> 992`) is evicted at `cblk[1]` -> `[6, 18, 19, 45, 54, 58]`, yielding **exactly `3,984` differing bytes, first at byte `4`**. |
| `#57` | `compress-decode R=32 D=512 mixed` | `new` (`i=2`: `compress2x_norm512`) | `dst = out((32, 528), "u2")` (`uint16(0x0055)`) | `33,792 B` (**Yes**) | `dst` is the **last** `Buffer` allocated before `chain()` launches `tf_ds_compress2x_norm512_kernel` (`32` CTAs, `~1.95 us`) on `stream`. Stream 0's trailing HtoD DMA of `[0x55, 0x00]` lands **after** CTAs `28..31` write `dst`, overwriting the final `3,840 B` (`30 x 128 B`) from cacheline-aligned offset `29,952` (`234 x 128 B`, row `28` col `192`) to `33,792`. | Of the `3,840` overwritten tail bytes, `4 x 16 x 2 = 128 B` are unwritten stride-pad columns `[512:528]` (already `0x0055`) and `5` active BF16 bytes match `[0x55, 0x00]` by value (`3,840 - 128 - 5 = 3,707`), yielding **exactly `3,707` differing bytes, first at byte `29,952`**. |

Why the other arms and larger row counts did not fail in `wp2-full-parity.json`:
- In `case_idx_small`, arms `old` (`i=1`) and `fused` (`i=2`) launch single-CTA `tf_ds_idx_fused_kernel`, which **never touches `bmax`**, runs `~15-20 us` before writing `blocks`, and benefits from `bmax`'s subsequent `32 KiB` upload buffering time ahead of `chain()`. At `R=7, nb=8192`, `bmax` is `229,376 B > 64 KiB`, forcing `cuMemcpyHtoD_v2` onto the driver's synchronous pageable path.
- In `case_compress_decode` `R=32, D=512`, `rk` and `rg` are `131,072 B > 64 KiB` (synchronous), whereas `dst` is `33,792 B <= 64 KiB` (asynchronous on stream 0) and is allocated last right before `chain()`.

---

## (b) Static SASS Audit & `compute-sanitizer` Verification (`initcheck`, `racecheck`, `memcheck`)

### Static Source & `sm_121` SASS Audit (`sass_res_usage.txt`)

Cubins compiled on GB10 (`100.95.49.21`) with `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false` across WP1 (`d01975d`), WP2 (`b28310e`), and tip (`d0c9fb1`):

| Kernel | WP2 (`b28310e`) vs Tip (`d0c9fb1`) | Registers | Stack / Spills | Static Shared | Barriers (`BAR.SYNC`) | Audit Findings |
|---|---|---:|---:|---:|---:|---|
| `tf_ds_block_max_kernel` | **Byte-identical SASS** (`464` insns) | 22 | `0 / 0` | `1,024 B` | 0 | Pure per-block reduction; writes all `rows * nb` floats. |
| `tf_ds_idx_pick_kernel` | **Byte-identical SASS** (`1,328` insns) | 38 | `0 / 0` | `2,208 B` (+ `nb*4` dyn) | 15 | Stages all `nbk` floats before `__syncthreads()`; initializes `prefix_s`, `mask_s`, `want_s`, `hist[256]`, and `warp_end[32]` (`block=1024`); writes exactly `k` `cblk` entries. *(Latent non-multiple-of-32 CTA `warp_end` guard `S12` fixed in `0257`; unreachable at `block=1024`.)* |
| `tf_ds_compress2x_norm512_kernel` | `880` insns (WP2) / `1,520` insns (tip, 2-arm PDL weight preload) | 30 (WP2) / 32 (tip) | `0 / 0` | `3,088 B` | 2 per arm | `row[512]` written by all 512 threads before `BAR.SYNC #1`; `warp_sum[0..3]` written by warp leaders `tid in {0,32,64,96}` before `BAR.SYNC #2`; writes `out[g*ldo + 0..511]` while leaving pad `[512..527]` untouched (matching kit `_rmsnorm`). |

### `compute-sanitizer` Results on GB10

1. **Historical WP2 `verify/cs-*.log` forensic caveat discovered:** Inspecting `100.95.49.21:~/tf-bench/verify/cs-{initcheck,racecheck,memcheck}-{compress-decode,idx}.log` (and `runs/fused-next-20261121/wp2/supervisor-cs-*.log`) confirmed `0 errors / 0 hazards` on `compress-decode R=32 D=512 mixed`, **but revealed that the three WP2 `cs-*-idx.log` runs had an empty `== parity` section** because `--kernel idx` was passed instead of `--kernel idx-small`, matching zero shapes.
2. **Standalone C++ Driver API sanitizer suite (`b2_cs_driver.cu` -> `cs_{initcheck,racecheck,memcheck}.log`):** Executed under `flock ~/tf-bench/gpu.lock` on GB10 (`< 2.2 MiB` peak device memory) directly launching **both** WP2 (`b28310e`) and tip (`d0c9fb1`) cubins with `pdl=0` and `pdl=1` across:
   - `tf_ds_block_max_kernel` + `tf_ds_idx_pick_kernel` at `R={1, 7}, nb=8192, k=2048` over random, tied, and special (`+-0, +-inf, qNaN`) distributions (with `d_bmax` and `d_cblk` deliberately left uninitialized before `dist=0` so `initcheck` verifies no uninitialized reads); and
   - `tf_ds_compress2x_norm512_kernel` at `R={1, 7, 32, 48}, D=512` (copying all `rows * 512` active BF16 channels back to host under `initcheck`).

| Tool | Log Artifact | `tf_ds_block_max_kernel` | `tf_ds_idx_pick_kernel` | `tf_ds_compress2x_norm512_kernel` | Summary |
|---|---|---|---|---|---|
| `initcheck` | `cs_initcheck.log` | **0 errors** | **0 errors** | **0 errors** | `ERROR SUMMARY: 0 errors` (all 12 suites `OK`) |
| `memcheck` | `cs_memcheck.log` | **0 errors** | **0 errors** | **0 errors** | `ERROR SUMMARY: 0 errors` (all 12 suites `OK`) |
| `racecheck` | `cs_racecheck.log` | **0 errors, 0 warnings** | **0 errors** *(24 benign intra-warp WAR warnings at `+0xbb0`/`+0xec0`)* | **0 errors, 0 warnings** | `RACECHECK SUMMARY: 0 errors` (all 12 suites `OK`) |

*Note on the `racecheck` intra-warp warning in `tf_ds_idx_pick_kernel` (`0 errors, 24 warnings`):* Disassembly of `sass_tip_idx_pick.txt` at `+0xbb0`..`+0xec0` (`tf_ds_pool_cut` warp-0 bin search, `if (threadIdx.x < 32)`) shows all 32 lanes of warp 0 read `want_s` via `LDS R32, [R27+0x410]` at `+0xbb0`, execute five full-mask `SHFL.UP PT, ..., 0xffffffff` instructions (`+0xc10`..`+0xcd0`), and consume `R32` in `IADD3 R31, R32, R31, RZ` at `+0xd00` **before** the single lane satisfying `0 < rank <= count` branches past `+0xd30` and updates `want_s` via `STS [R27+0x410], R4` at `+0xec0` (followed by `WARPSYNC.ALL` at `+0xf20` and `BAR.SYNC` at `+0xf40`). Because `racecheck` does not model intra-warp `SHFL.UP` + register-dataflow ordering, it logs a warning (`0 errors`), while hardware execution strictly orders every lane's `LDS R32` before the winning lane's `STS`.

---

## (c) Instrumented GB10 Diagnostic Sweeps (`b2_probe.py`) — 14,400 Kernel Executions, 0 Mismatches

`b2_probe.py` ran per-PID isolated suites on GB10 using stream-ordered `cuMemcpyHtoDAsync_v2(..., stream)` across **3 cubin trees** (`old_wp1` = `d01975d`, `wp2` = `b28310e`, `new_tip` = `d0c9fb1`), **4 execution modes** (`eager_pdl0`, `eager_pdl1`, `graph_pdl0`, `graph_pdl1`) × **100 iterations each** (`400` runs/arm, `1,200` runs/case, **`14,400` total runs**):
- Pre-poisoned the CUDA freed-memory pool and every scratch/output buffer before every launch cycling `0xFF`, `0x7FC00000` (`qNaN`), `0xDEADBEEF`, and random non-zero bytes;
- Rotated input, scratch, ring, and output pointers across non-256B alignments (`+0, +2, +4, +6, +8, +12, +14, +16, +20, +24, ..., +264` bytes) inside 64-byte head/tail guarded arenas;
- Verified SHA-256 of every device input buffer **before** and **after** all 400 launches per arm (`all_input_sha256_and_guards_clean: true`, `0` guard-byte corruptions).

### 1. Indexer Small-Row Sweep (`b2_probe_idx_pid2564496.json`, peak VRAM `2.573 MiB`, `16.02 s`)

Cubin SHA-256s: `old-ops.cubin` = `wp2-ops.cubin` = `927b41ab84dda9bc...`, `new-ops.cubin` = `f00e5a1e49b3493a...` (`tf_ds_block_max_kernel` + `tf_ds_idx_pick_kernel`).

| Case | Score Input SHA-256 | Sample `new_tip` Ptrs (`score` / `bmax` / `cblk`) | `old_wp1` (`4x100`) | `wp2` (`4x100`) | `new_tip` (`4x100`) | Input SHA & Guards |
|---|---|---|---:|---:|---:|---|
| `idx-small R=1 nb=8192 k=2048 wp2_seed197_random` | `a207ef699d44...` | `0x32ee0004c` / `0x32ee4ca40` / `0x32ee54c44` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `idx-small R=1 nb=8192 k=2048 random` | `6f357ff60f7d...` | `0x32ee0004c` / `0x32ee4ca40` / `0x32ee54c44` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `idx-small R=1 nb=8192 k=2048 ties` | `0b52c96fd817...` | `0x32ee0004c` / `0x32ee4ca40` / `0x32ee54c44` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `idx-small R=1 nb=8192 k=2048 flat` | `6470bc26e7cc...` | `0x32ee0004c` / `0x32ee4ca40` / `0x32ee54c44` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `idx-small R=1 nb=8192 k=2048 adversarial` | `83caaede093e...` | `0x32ee0004c` / `0x32ee4ca40` / `0x32ee54c44` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `idx-small R=7 nb=8192 k=2048 random` | `3e29e1a693b4...` | `0x32ee0004c` / `0x32f41c240` / `0x32f454444` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |

### 2. Decode Compressor Sweep (`b2_probe_compress_pid2564807.json`, peak VRAM `0.899 MiB`, `15.93 s`)

Cubin SHA-256s: `old-torch.cubin` = `fafb487170698645...` (`tf_ds_compress2x_norm_kernel`), `wp2-torch.cubin` = `079c965abddcf0a5...` (`tf_ds_compress2x_norm512_kernel`), `new-torch.cubin` = `a55c9fee2d3bd3c3...` (`tf_ds_compress2x_norm512_kernel`), compared byte-for-byte on `dst` (active + untouched pad), `ring_kv`, `ring_score`, and `invalid` flag against `tf_ds_compress2x_kernel` + kit `ee9e02` `_rmsnorm`.

| Case | `ckv` Input SHA-256 | Sample `new_tip` Ptrs (`ckv` / `w` / `dst`) | `old_wp1` (`4x100`) | `wp2` (`4x100`) | `new_tip` (`4x100`) | Input SHA & Guards |
|---|---|---|---:|---:|---:|---|
| `compress-decode R=32 D=512 wp2_seed197_mixed` | `c79a992c7bd6...` | `0x32ee0004c` / `0x32ee20c46` / `0x32ee69840` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `compress-decode R=1 D=512 mixed` | `afcbe2992266...` | `0x32ee0004c` / `0x32ee01846` / `0x32ee04840` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `compress-decode R=7 D=512 mixed` | `d4c1a587c53e...` | `0x32ee0004c` / `0x32ee07846` / `0x32ee18040` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `compress-decode R=32 D=512 mixed` | `0b66b36718e7...` | `0x32ee0004c` / `0x32ee20c46` / `0x32ee69840` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `compress-decode R=48 D=512 mixed` | `7ba73c483519...` | `0x32ee0004c` / `0x32ee30e46` / `0x32ee9da40` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |
| `compress-decode R=32 D=512 adversarial` | `dfbc65babed3...` | `0x32ee0004c` / `0x32ee21046` / `0x32ee69c40` | `0 / 400` | `0 / 400` | `0 / 400` | Clean (`0` corrupt) |

---

## (d) Direct Hardware Reproduction of the Stream-0 DMA Race on GB10

`b2_probe.py --suite stream0-repro` replayed the exact unmodified `bench.py` upload sequence (`4` pageable `<= 64 KiB` `cuMemcpyHtoD_v2` calls on legacy stream 0 immediately followed by kernel launch on `CU_STREAM_NON_BLOCKING` `stream`) back-to-back against the fixed stream-ordered `cuMemcpyHtoDAsync_v2(..., stream)` path on GB10:

| Run Artifact | Trials / Mode | `bench_py_stream0_cuMemcpyHtoD_v2` (`idx-small R=1 nb=8192`) | `fixed_stream_cuMemcpyHtoDAsync_v2` (`idx-small R=1 nb=8192`) | `fixed_stream_cuMemcpyHtoDAsync_v2` (`compress-decode R=32 D=512`) |
|---|---:|---|---|---|
| `b2_probe_stream0_repro_pid2565118.json` | `20` | **1 failure** (`trial 7`: **`4,012` differing bytes**, `first_offset: 0`) | **0 / 20 failures** | **0 / 20 failures** |
| `b2_probe_stream0_repro_pid2568086.json` | `200` | **1 failure** (`trial 68`: **`4,009` differing bytes**, `first_offset: 0`) | **0 / 200 failures** | **0 / 200 failures** |
| **Combined** | **`220`** | **2 hardware race reproductions (`~0.9%` per-launch rate; `~4,000` byte signature)** | **0 / 220 failures** | **0 / 220 failures** |

Both hardware reproductions (`4,012` and `4,009` differing bytes out of `8,192`) match the exact `~3,970–4,012`-byte envelope created when stream 0's `0x55` HtoD DMA overwrites a suffix of `bmax` or `blocks` concurrently with `CU_STREAM_NON_BLOCKING` execution, while `cuMemcpyHtoDAsync_v2(..., stream)` achieved **0 / 220** failures (plus **0 / 14,400** in the main diagnostic sweeps).

---

## Harness Fix & CPU Unit-Test Validation

- **`tools/fused_bench/bench.py` (`Buffer.__init__`, `Buffer.read`, `Buffer.readarr`):**
  - When `stream.value` is initialized (`CU_STREAM_NON_BLOCKING`), `Buffer.__init__` uploads contiguous `data` onto `stream` with `cuMemcpyHtoDAsync_v2(self.addr, P(data.ctypes.data), ct.c_size_t(data.nbytes), stream)` and immediately calls `cuStreamSynchronize(stream)` while holding `data` alive across the synchronization; when `stream.value` is null, it calls `cuMemcpyHtoD_v2` + `cuCtxSynchronize`.
  - `Buffer.read` and `Buffer.readarr` call `if stream.value: cu("cuStreamSynchronize", stream)` before `cuMemcpyDtoH_v2`.
- **`tools/fused_bench/test_bench_guard.py`:** Added `test_f_buffer_stream_ordered_upload_and_read_sync_when_stream_non_null` (verifying `cuMemcpyHtoDAsync_v2` + `cuStreamSynchronize` ordering, `weakref`-proven host-array liveness across `gc.collect()` during `cuStreamSynchronize`, and deferred-stream-write visibility before `cuMemcpyDtoH_v2` in `read()` / `readarr()`) and `test_g_buffer_null_stream_fallback_uses_ctx_sync`. All **9/9** unit tests pass (`python3 -m unittest discover -s tools/fused_bench -p 'test_*.py' -v`).

## Artifacts in This Directory

- `b2_probe.py` — Instrumented Python/ctypes CUDA Driver API diagnostic & stream-0 race reproduction harness.
- `b2_cs_driver.cu` — Minimal C++ CUDA Driver API harness used for `compute-sanitizer`.
- `b2_probe_idx_pid2564496.{json,log}` — 7,200-run `0195` `idx-small` poisoned/misaligned sweep (`0` mismatches).
- `b2_probe_compress_pid2564807.{json,log}` — 7,200-run `0197` `compress-decode` poisoned/misaligned sweep (`0` mismatches).
- `b2_probe_stream0_repro_pid2565118.{json,log}`, `b2_probe_stream0_repro_pid2568086.{json,log}` — Hardware reproductions (`4,012` and `4,009` differing bytes on stream 0 vs `0 / 220` on `stream`).
- `cs_initcheck.log`, `cs_racecheck.log`, `cs_memcheck.log` — `compute-sanitizer` outputs from `b2_cs_driver` on GB10 (`0` errors across all tools).
- `sass_res_usage.txt` — `cuobjdump --dump-resource-usage` for WP1, WP2, and tip cubins on `sm_121`.
