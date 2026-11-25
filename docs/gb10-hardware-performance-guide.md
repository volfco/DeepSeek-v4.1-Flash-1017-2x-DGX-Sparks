# GB10 (`sm_121`) Hardware Performance & Kernel Design Guide

Empirical hardware performance ceilings, instruction/opcode behaviors, measurement traps, and bit-exactness contracts discovered on the 2× NVIDIA DGX Spark (`gb10`, `sm_121`) DeepSeek-V4.1-Flash stack.

**Companion process:** [CUDA Kernel Compilation, Block Microbenchmarking, and Opcode Evaluation Process](cuda-kernel-microbenchmark-process.md) (use this guide as the hardware reference when triaging bounds, interpreting SASS, and applying optimization patterns).

All measurements below were collected on the Spark pair (`gx10-eb0e` head, `gx10-235f` worker) with CUDA 13.0 and the engine's pinned compiler flags:

```bash
nvcc -O3 -arch=sm_121 --fmad=false --ftz=false
```

Never compile repo kernels with `--use_fast_math`.

---

## 1. Hardware Topology & Driver-Reported Limits

Each DGX Spark is an independent unified-memory board (256-bit LPDDR5X-8533 shared by CPU and GPU, connected across ranks over dual ConnectX RDMA rails).

| Parameter | Measured / Reported Value | Source |
|---|---|---|
| Architecture | NVIDIA GB10 Blackwell (`sm_121`) | `tools/fused_bench/build.sh`, `continue.md` |
| Streaming Multiprocessors (`sm_count`) | **48 SMs** | `runs/fused-next-20261121/wp0/sass.json` |
| Max threads / SM (`max_threads_per_sm`) | **1,536** (**48 warps / SM**, `warp_size = 32`) | `runs/fused-next-20261121/wp0/sass.json` |
| Register file / SM (`regs_per_sm`) | **65,536** 32-bit registers | `runs/fused-next-20261121/wp0/sass.json` |
| Shared memory / SM (`smem_per_sm`) | **102,400 B** (100 KiB; opt-in dynamic shared required for wide pools like 0217's 131,072-col path) | `runs/fused-next-20261121/wp0/sass.json`, `continue.md` |
| GPU L2 cache / System-Level Cache (SLC) | **24 MB GPU L2**, **16 MB SLC** ahead of LPDDR5X | `runs/overnight-opt/RESULTS.md` |
| Memory interface (nominal) | **273 GB/s** (256-bit LPDDR5X-8533) | `runs/overnight-opt/RESULTS.md` |
| Sustained GPU clock under load | **2,359–2,392 MHz** (under 2,400 MHz cap; 62–87 °C, no throttle) | `runs/w4/RESULTS.md`, `continue.md` |

### Occupancy Arithmetic on GB10 (`sm_121`)

Because GB10 caps each SM at **1,536 threads (48 warps)** and **65,536 registers**, theoretical occupancy is:

$$\text{blocks/SM} = \min\!\left(\left\lfloor\frac{1536}{\text{threads}}\right\rfloor,\ \left\lfloor\frac{65536}{\text{regs} \times \text{threads}}\right\rfloor,\ \left\lfloor\frac{102400}{\text{smem\_bytes}}\right\rfloor\right)$$

Practical consequences observed in repo kernels:
- **1,024-thread CTA** (`idx_fused`, `idx_pick`, `topk_indices_cut`): capped at **1 block/SM (32 warps, 66.7% theoretical occupancy)** by `max_threads_per_sm = 1536` regardless of register count (as long as `REG <= 64`). Shrinking the CTA to 512 or 256 threads increases occupancy on paper but **slows radix scans by 25–45%** because each CTA needs more grid-stride passes (`runs/fused-next-20261121/wp1/RESULTS.md`, Item 3).
- **512-thread CTA** (`compress2x_norm512`): **3 blocks/SM (48 warps, 100% occupancy)** while `REG <= 42` (`65536 / (32 * 512) = 4`, thread-capped at 3). Moving 30 → 32 registers in `0250` kept occupancy at 100% (`runs/fused-next-20261121/wp4/RESULTS.md`).
- **256-thread CTA** (`fused_rope_store`, `fused_rope_qd_pair`): **6 blocks/SM (100%)** while `REG <= 42`; at `REG = 66` (`compress2_norm` prompt), register-bound at **3 blocks/SM** (`runs/fused-next-20261121/wp2/RESULTS.md`, Item 6).
- **128-thread CTA** (`compress2x_norm128`, `fused_idx_key`, `fused_rope_qd`): **12 blocks/SM (100%)** while `REG <= 42`; at `REG = 66`, register-bound at **7 blocks/SM**.
- **Small-grid reality:** With 48 SMs, a 1-CTA-per-row decode kernel at `R = 1..48` puts **at most 1 CTA per SM**. At decode row counts, occupancy tricks do not hide single-CTA critical-path latency (`runs/fused-next-20261121/wp3/RESULTS.md`, Item 4).

---

## 2. Measured Bandwidth & Latency Ceilings (Rooflines)

Always price a kernel against the **measured** GB10 ceiling for its actual working-set size and access pattern, never the 273 GB/s datasheet figure.

### Memory Bandwidth Ceilings

| Operation / Working Set | Measured GB/s (per board) | Basis / Notes | Source |
|---|---:|---|---|
| Repeated L2-resident read-only scan | **1,441–2,204 GB/s** | Fits inside 24 MB L2; never use for DRAM sizing | `runs/w4/RESULTS.md`, `runs/overnight-opt/RESULTS.md` |
| L2-resident 16B copy (`bw.py`, 8 MiB R+W) | **954 GB/s** (1 MiB: 863; up to 1,510 at peak clocks) | Conservative **cache-basis roofline** used by `report.py` for working sets `<= 32 MB` | `runs/fused-next-20261121/wp0/bw.json`, `BASELINE.md` |
| DRAM streaming read (`uint4`, 256 MiB) | **240.4 GB/s** head / **250.3 GB/s** worker (paired 6-iter: 252–253) | Contiguous read ceiling (~88–92% of 273 GB/s) | `runs/overnight-opt/RESULTS.md` |
| DRAM scattered / gapped weight read (1 GiB or 4× stride) | **231–236 GB/s** (`ldg`); **206 GB/s** (`cp.async` gapped) | **True DRAM ceiling for routed MoE weights**; `ldg` beats `cp.async` on gapped ranges | `runs/w4/RESULTS.md` |
| DRAM 16B copy (`bw.py`, 64 / 256 MiB R+W) | **224 / 222 GB/s** | **DRAM-basis copy roofline** used by `report.py` for working sets `> 32 MB` | `runs/fused-next-20261121/wp0/bw.json` |
| GPU reads during CPU contention | **231 GB/s** | Unified LPDDR5X shared with CPU | `runs/overnight-opt/RESULTS.md` |
| GPU reads alongside concurrent L2 prefetch stream | **13–30% lower** (179–226 vs 251–262 GB/s) | Concurrent prefetch competes for the same DRAM bus; net serving win comes from hiding latency between layers, not raising peak bus rate | `runs/w4/RESULTS.md` |
| Cache-bypassing UMA read / write | **165.7 / 115.9 GB/s** | Uncached unified-memory access | `runs/overnight-opt/RESULTS.md` |
| Routed MoE decode (achieved in serving) | **221–222 GB/s** (~94–96% of scattered roof; 27.2 µs / routed place) | Bus-bound; leaves only 0.62–0.95 ms removable at fixed bytes | `runs/w4/RESULTS.md`, `runs/overnight-opt/RESULTS.md` |

### Launch & Chain Latency Floors (CUDA Graphs)

| Measurement | Measured Latency (µs) | Notes | Source |
|---|---:|---|---|
| Empty kernel (`empty_kernel`), single/small CTA | **0.456–0.459 µs** | In-graph node floor (`R = 1..48`, grid 1..48) | `runs/fused-next-20261121/wp2/RESULTS.md`, `wp3/RESULTS.md` |
| Empty kernel with 2 KiB static shared memory | **+0.020 to +0.038 µs** (0.496 vs 0.458 µs at grid 1, block 512) | Dropping unused static shared memory pays immediately on micro-kernels | `runs/fused-next-20261121/wp2/RESULTS.md`, Item 1 |
| Empty kernel at larger prompt grids (`R = 128..2048`) | **0.497–1.215 µs** (`R=128..1024`), up to **7.22 µs** (`R=2048, block=1024`) | Always match the exact grid/block/shared shape for the floor | `runs/fused-next-20261121/wp0/BASELINE.md`, `wp1/RESULTS.md` |
| 6-node decode fused chain empty floor (no PDL / PDL attr) | **2.69–2.95 µs** / **2.32–2.71 µs** | PDL removes ~0.35–0.37 µs of pure floor and ~1.5–1.9 µs across real bodies | `runs/fused-next-20261121/wp4/RESULTS.md`, Item 1 |
| Standalone RDMA ring gather (width 5120, GID 3, PDL) | **9.4 µs + 1.16 µs/row** (9.9 µs at 1 row, 17.3 µs at 7 rows) | Cross-rank collective floor | `continue.md` ("Scheduling round A") |
| Markov bias step (hit vs miss) | **9.8 µs** hit / **142–162 µs** miss (+133–153 µs/miss) | Miss reads 33.1 MB FP16 head at bus speed | `continue.md` ("Trace review results") |

### Roofline Triage Thresholds (`tools/fused_bench/report.py`)

- **Working-set basis:** `bytes > 32 MB` → DRAM basis (`224 GB/s` copy / `231–236 GB/s` scattered read); `bytes <= 32 MB` → cache basis (`954 GB/s`).
- **Launch-bound:** `measured <= 1.10 × empty_floor` (or within ~0.2 µs of floor) — body edits cannot win; only fusing launches or skipping the call helps.
- **Bandwidth-bound:** `roofline_us >= 0.85 × measured` — compute/opcode edits cannot win; only moving fewer bytes, changing precision/layout, or improving cache residency helps.
- **Issue / latency-bound:** everything else (`headroom = measured - max(floor, roofline)`) — target for block-level microbenchmarking and SASS optimization.

---

## 3. Instruction & Opcode Performance on GB10 (`sm_121`)

Every rule below was directly measured on GB10 hardware via `tools/fused_bench`, `cuobjdump -sass`, and paired graph timings.

### 3.1 Global Memory Access Patterns (`LDG` / `STG`)

1. **Vectorized aligned 16-byte loads (`LDG.E.128` via `float4`) with static trip counts win reliably.**
   - Replacing 8 scalar loads with 2× `LDG.E.128` + fully unrolled comparisons (`0213`, `0255`) cut runtime **−6% to −20%** on `idx_fused` and **−15% to −23%** on `cand_fast8`, while dropping `cand_fast8` register usage from **31 → 22** (`runs/fused-next-20261121/wp1/RESULTS.md` Item 2; `runs/kernel-sweep-20261123/PATTERNS.md` P5).
   - Guard alignment (`stride % 4 == 0`, 16-byte base) and keep a scalar fallback; avoid whole-kernel templating when only a hot inner loop needs `<BS>` specialization (whole-body templating inflated code size and regressed small shapes by 0.2 µs in `0213`).
2. **Strided per-lane stores (`STG` 64 B apart) are catastrophic.**
   - Writing 8× `int64` keys per block from one lane (adjacent lanes 64 bytes apart) measured **41.7 µs** per CTA-row in `storepat.cu` versus **7.5 µs** for coalesced 8-byte stores (5.5× slower) and **21.2 µs** for 16-byte `ulonglong2` stores (which also spilled `STACK=64`).
   - Folding a 224-CTA streaming writer (`_score_keys`) into a 7-CTA per-row kernel lost at every decode shape (`0206`: 316.6 vs 38.5 µs at `R=128, w=32,768`; `runs/fused-next-20261121/wp1/RESULTS.md` Item 1).
3. **Instruction scheduling of cold loads matters even when the opcode histogram is identical.**
   - In `compress2x_norm128`, hoisting partner slot loads (`gi[2g]`, `gi[2g+1]`) ahead of the ring scatters produced the **exact same 392-instruction histogram**, yet improved runtime from **1.460 → 1.316 µs** at `R=1` and **1.668 → 1.521 µs** at `R=7` by letting ptxas hide load latency behind stores and ballots (`runs/fused-next-20261121/wp2/RESULTS.md` Item 1).
4. **Eliminate runtime integer division in block prologues.**
   - Passing grid `(rows, heads)` instead of `(rows * heads)` with `blockIdx.x / heads` removed ~15 prologue instructions and a serial divide before the first load in `0241` (`runs/fused-next-20261121/wp3/RESULTS.md` Item 0).

### 3.2 Warp Reductions, Shuffles, Atomics & Barriers (`REDUX`, `SHFL`, `ATOMS`, `BAR.SYNC`)

1. **Whole-warp `REDUX` beats 4-stage `SHFL` — half-warp masked `REDUX` loses.**
   - Full-mask `__reduce_max_sync(0xffffffffu, ...)` over sign-cleared BF16 bits replaced 4 `SHFL`/`FMNMX` stages with 1 `REDUX` and contributed to −4% to −8% decode wins (`0240`).
   - **Anti-pattern:** Half-warp masked `__reduce_max_sync` (16-lane groups) made `ptxas` duplicate the reduction into 3 code paths (**6 static `REDUX`**) and **regressed latent by +0.10 µs (1.541 → 1.644 µs)** at every decode row count. Keep 4 width-16 `SHFL.BFLY` stages for half-warp groups (`runs/fused-next-20261121/wp3/RESULTS.md` Item 2).
2. **`sm_121` hardware already coalesces intra-warp uniform shared `ATOMS`.**
   - Replacing per-element shared `atomicAdd` in radix histograms with `__match_any_sync` + `POPC` + leader election (`MATCH` + `ATOMS`) was **2.9–4.4 µs slower** at 1,024 threads (`11.33 vs 8.42 µs` random, `21.51 vs 17.15 µs` ties; `runs/fused-next-20261121/wp1/RESULTS.md` Item 3).
   - Even on heavy tie rows (`0254`), 82–86% of extra cost is extra global scan passes and barriers, not shared-atomic bank contention (`continue.md` register).
3. **A single `__syncthreads()` (`BAR.SYNC`) costs more than redundant L1-resident per-warp work.**
   - Resolving partner slots once per block + `__syncthreads()` regressed `compress2x_norm` by **+0.025 to +0.058 µs** at 16 warps and **+0.164 to +0.183 µs** at 4 warps compared to letting every warp run its own L1-cached ballot/load (`runs/fused-next-20261121/wp2/RESULTS.md` Item 3).
4. **Warp-uniform skip guards only win when the skipped work dominates the guard latency.**
   - Skipping `SHFL.BFLY` on wholly unrotated warps won **−9% on decode latent** (14/16 warps skipped, `1.540 → 1.401 µs`) and **−60% on prompt latent `R=128`** (`4.209 → 1.664 µs`), but **regressed key** (`1.623 → 1.686 µs`, only 2/4 warps skipped) and was neutral on 1-element query (`runs/fused-next-20261121/wp3/RESULTS.md` Item 1).

### 3.3 Arithmetic, Transcendentals (`MUFU`), FP4 & Tensor Cores

1. **Avoid per-element `__fdiv_rn` when exact bit-level or power-of-two reciprocal math exists.**
   - `__fdiv_rn` expands to `MUFU.RCP` + 4–6 dependent Newton-Raphson `FFMA`/`FMUL` instructions. Replacing E8M0 scale derivation + per-element `__fdiv_rn(x, scale)` with bit-extracted BF16 exponent (`tf_ds_scale_byte_bits`) + exact power-of-two `__fmul_rn(x, tf_ds_recip_bits(byte))` cut `prompt_idx_key` by **−50% to −55%** (`4.270 → 1.909 µs` at `R=128`; `6.174 → 3.087 µs` at `R=1024`) and `idx_key` by **−8%** (`runs/fused-next-20261121/wp3/RESULTS.md` Item 2).
2. **NVCC already CSEs duplicate `expf` calls on identical operands.**
   - Always check `cuobjdump -sass` before hand-caching transcendentals: 4 source `expf` calls on 2 operands already compiled to 2 `MUFU.EX2` (`runs/fused-next-20261121/wp2/RESULTS.md` Item 2; `continue.md` register).
   - Hand-rewriting a 2-`expf` pair into 1 `expf` with an `isfinite` branch regressed decode compressor by **13–17%** (`0207`); a branchless 1-`expf` form was exact and **0.000 µs faster** because pair math was only 0.12 µs of a load-latency-bound body.
3. **FP4 conversion intrinsics depend critically on `-arch=sm_121` vs `-arch=sm_121a`.**
   - On the repo's pinned **`-arch=sm_121`**, `__nv_cvt_float2_to_fp4x2` / `__nv_cvt_float_to_fp4` are **software-emulated** (168 → 184 instructions) and run **5–22% slower** (`1.297 → 1.362 µs` at `R=1`, `3.386 → 4.123 µs` at `R=48`).
   - On **`-arch=sm_121a`** (family-specific ISA), the same intrinsics emit native `CVT`/`F2FP` hardware instructions and run **10–15% faster** (`1.296 → 1.173 µs` at `R=1`, `3.385 → 2.975 µs` at `R=48`). Because production builds pin `sm_121`, do not introduce FP4 intrinsics unless the build target changes (`runs/fused-next-20261121/wp3/RESULTS.md` Item 3).
4. **Never replace a Tensor-Core GEMM/dot with scalar CUDA-core FP32 math.**
   - `0267` (`TF_DS_MARKOV_FUSED`) saved 252.5 KiB of FP16 staging + 1 launch per greedy Markov miss, but replaced the kit's Tensor-Core dot with scalar CUDA-core accumulation (with 512 B strided head reads across lanes). Result: **−5.9% to −10.2% greedy tok/s**, drafter pass **4.20 → 6.33 ms** (`runs/overnight-20261122/k267/RESULTS.md`, `docs/fused-kernel-rebenchmark-audit.md`).

### 3.4 Geometry, Parallelism, Fusion & PDL

1. **Spread wide work across the grid; never serialize wide inputs in one CTA per row.**
   - One CTA/row over a 384×5,120 gate (`0193`/`0194`) slowed serving **11–19%**.
   - Splitting wide indexer selection (`nbk >= 6,144` at `R <= 7`) into grid-wide `block_max` + `idx_pick` (`0215`) beat single-launch `idx_fused` at `nbk = 8,192` (`12.14 → 10.53 µs` at `R=1`), while single-launch still won at `nbk = 4,096` (`8.46 vs 8.67 µs` at `R=7`). Always measure the exact crossover (`runs/fused-next-20261121/wp1/RESULTS.md` Item 5).
2. **Thread coarsening (fewer threads, more channels/thread) hurts latency-bound decode kernels.**
   - At `D=512`, moving from 512 threads × 1 channel (with shared exchange + barrier) to 128 threads × 4 channels (no exchange, no barrier, 16B loads) was **4–7% slower** (`1.932 → 2.033 µs` at `R=7`) because quadrupling each thread's dependent chain outweighed removing the barrier (`runs/fused-next-20261121/wp2/RESULTS.md` Item 1).
   - Conversely, packing 2 rotary elements per lane (`fused_rope_qd_pair`, 1 warp/head, zero `SHFL`) beat 1-element/thread from `R >= 32` onward (`4.020 → 2.812 µs` at `R=48`, `68.9 → 36.9 µs` at `R=1024`; `runs/fused-next-20261121/wp3/RESULTS.md` Item 0).
3. **Fusion vs split depends on working-set residency, not just launch count.**
   - At `R=128` prompt query, the 2-launch kit chain beat fused `query_pair` by **2.56–3.30 µs** for 1–4 distinct buffers and for 8 reuses of 1 buffer (`+1.07 µs`), but **fused won by 0.98–1.14 µs** when 8 distinct buffers exceeded L2-resident reuse (`runs/suite-fused-next2-20261121/query128/RESULTS.md`).
4. **PDL (`PREEXIT` / `ACQBULK`) must place proven-independent loads between `launch_dependents` and `wait`.**
   - Issuing `launch_dependents` before `wait` with no independent work **slowed** the decode chain by up to **+0.6 µs** (occupying SM slots while blocked at `ACQBULK`).
   - Placing immutable model-weight loads (`LDG.E.U16` of norm weights) between `PREEXIT` and `ACQBULK` won **0.17–0.35 µs** across the 6-node chain (`0250`), provided the default non-PDL path uses a separate branch/instantiation so it does not pay for the held register (`runs/fused-next-20261121/wp4/RESULTS.md` Item 2).
   - Always validate PDL splits with a **delayed-producer canary** (`delay_copy` + negative race control) to prove no dependent byte is read before `ACQBULK`.
5. **Radix early-exit (`0254` / `0290`) and dead-work skipping (`0214`) deliver the largest algorithmic wins.**
   - Stopping a radix select when `remaining_rank == bin_count` plus warp-parallel cutoff search cut `topk_indices` **−39% to −60%** (`237 → 101 µs` at `128 × 32,768`; `runs/kernel-sweep-20261123/PATTERNS.md` P1).
   - Skipping ballot/prefix/compaction on mask-only prompt arms (`0214`) saved **−10% to −13%** (`runs/fused-next-20261121/wp1/RESULTS.md` Item 4).
6. **Zero spills (`STACK=0 LOCAL=0`) is a hard baseline invariant.**
   - Every shipped kernel in `dsv41_ops.cu` and `dsv41_torch.cu` compiles with `STACK=0 LOCAL=0`. Any per-thread array or register pressure spike that creates non-zero `STACK`/`LOCAL` must be treated as a red flag (`runs/fused-next-20261121/wp1/RESULTS.md` Item 1; `continue.md` register).

---

## 4. Hardware Counters, Profiling Quirks & Measurement Traps on GB10

| Mechanism / Tool | Verified GB10 Behavior | What To Do Instead | Source |
|---|---|---|---|
| `lts__t_sectors_op_read_lookup_miss_realtime.sum` | **Calibrated and accurate** (× 32 B/sector recovers 100.003% of a 256 MiB scan). Counts GPU read misses leaving L2 (SLC may still satisfy some). | Use for read-miss bandwidth attribution; collect **separately from timing runs**. | `runs/overnight-opt/RESULTS.md` |
| DRAM / MC byte counters & `lts__d_sectors_fill_device.sum` | **Unavailable / returns 0** in GB10 CUPTI PM catalog; driver also reports memory clock unavailable. | Estimate DRAM traffic from L2 read-miss sectors or calibrated copy/read probes (`bw.py`, `calibrate.cu`). | `runs/overnight-opt/RESULTS.md` |
| NVML memory utilization (`nvmlUtilization_t.memory`) | **Returns 0%** even during sustained 253 GB/s reads. | Never use NVML memory % on GB10 to judge bandwidth utilization. | `runs/overnight-opt/RESULTS.md` |
| `nsys` CUDA tracing on the Zig engine | **CUDA injection does not attach** (captures OS-runtime calls only). | Use `TF_DS_ROUND_PROFILE=2 GRAPHS=0`, `TF_DS_RING_TRACE`, or standalone `tools/fused_bench` + `ncu`. | `continue.md` ("Round B2") |
| `ncu` (Nsight Compute) | **Requires root** on the Sparks; hardware replay perturbs wall time. | Use `ncu` only for targeted SASS/stall/counter verification (single kernel, `--launch-count 1 --launch-skip 20`, graphs off), never for latency A/B. | `runs/fused-next-20261121/wp3/NCU_REQUESTS.md`, `wp4/NCU_REQUESTS.md` |
| Legacy stream-0 `cuMemcpy*_v2` vs `CU_STREAM_NON_BLOCKING` | Pageable HtoD transfers **`<= 64 KiB`** stage asynchronously on legacy stream 0 and **race** kernels on `CU_STREAM_NON_BLOCKING`, causing sporadic false parity mismatches (Blocker B2). | Always use `cuMemcpyHtoDAsync_v2(..., stream)` + `cuStreamSynchronize(stream)` before launch, and `cuStreamSynchronize(stream)` before `cuMemcpyDtoH_v2`. | `runs/pre-bench-20261121/b2-investigation/RESULTS.md` |
| `compute-sanitizer racecheck` intra-warp WAR | Reports benign warnings when warp lanes read shared memory (`LDS`), synchronize via full-mask `SHFL.UP` dataflow, then one lane writes (`STS`). | Inspect SASS around reported offsets to distinguish true cross-warp races from intra-warp `SHFL`-ordered WAR; require **0 errors** in `memcheck`, `initcheck`, `racecheck`. | `runs/pre-bench-20261121/b2-investigation/RESULTS.md` |
| Served `TF_DS_CLOCK_MARKS` & host `forward_ms_per_round` | `TF_DS_CLOCK_MARKS` is broken (`0xAAAA...` uninitialized fill); host `forward_ms` diffs 0.01 ms quantized means (up to ±0.6 ms artifact per cell). | For serving attribution, read GPU-clock stretch `per_round.tail[0]` (`table_gpu.py`) or eager `TF_DS_ROUND_PROFILE=2 GRAPHS=0` (and set `TF_DS_ROUND_PROFILE=0` when measuring cross-rank skew). | `continue.md` ("Scoring artifacts", "Round B2") |
| Production interference during standalone timing | Serving traffic or `< 3 GiB MemAvailable` contaminates timings or triggers `memguard` (`2.5 GiB` floor). | Hold `~/tf-bench/gpu.lock`, require `MemAvailable >= 3 GiB`, verify `/health` `live.connections == 0` and unchanged `(prompt_tokens_total, generation_tokens_total)` before and after every timed block (`timing_clean == true`). | `tools/fused_bench/README.md`, `runs/fused-next-20261121/wp0/RESULTS.md` |

---

## 5. Bit-Exactness Contracts (Must Hold Before Timing)

Every kernel optimization in this repo must preserve bit-for-bit parity against the unfused/kit reference (`0 differing bytes`) before any timing counts:

1. **Compiler arithmetic flags:** `-O3 -arch=sm_121 --fmad=false --ftz=false`. Fast math or unexpected FMA contraction breaks exactness.
2. **Decode vs. Prompt RMSNorm (`0197`):**
   - Decode RMSNorm must match the pinned kit cubin's approximate PTX sequence (**`MUFU.SQRT` then `MUFU.RCP` with range handling**), not `sqrtf` + `__fdiv_rn` (which disagreed on 9 BF16 bytes across 4,096 × 512 rows).
   - Prompt RMSNorm must match torch's `rsqrt` (`MUFU.RSQ`) and reduction order (`continue.md` register; `runs/fused-kernel-audit-20261120/RESULTS.md`).
3. **RoPE contraction (`0198`):** Triton contracts `re = fma(xe, cs, -(xo * sn))`; preserve the exact FMA form and odd-element convention (`0252`).
4. **FP4 E2M1 / E8M0 / E4M3 quantization (`0198`):**
   - E2M1 uses ties-to-even, NaN clamps to `-6`, and negative zero (`0x8`) must be masked to `+0` (`0x0`).
   - Verify latent changes exhaustively across **all 65,536 BF16 encodings** (and for intrinsic tests, all `65,536 × 255 = 16,711,680` `(bf16, E8M0)` pairs).
5. **Score keys & radix selection (`0195`, `0254`, `0261`, `0290`):**
   - `-0.0` maps to `+0.0` in all score keys; ties break to the lower column/block ID.
   - Kit `_score_keys` stores **signed** int64 (`scoreKey ^ 0x8000000000000000`); engine radix select operates on **unsigned** `scoreKey`.
   - NaN ordering differs by site (`tf_ds_score_key` orders NaN payload bits; `tf_ds_draw_key` places NaN lowest).
   - Early-exit complete-bin cutoff is exact **only as a set** (never when a caller depends on the exact k-th key value or full order).
   - Skipping constant high column bytes (`0254`) holds **only for local columns `< 65,536`** — rank-1 global IDs starting at `64,640` straddle `65,536` (`docs/fused-kernel-rebenchmark-audit.md`).
6. **Markov bias (`0267`) & drafter guards (`0268`):** Preserve FP16 rounding of bias before adding logits; preserve `slot == -1` ring-write guard and untouched stride-padding bytes.

---

## 6. Serving-Budget Reality Check (Price Before Optimizing)

Before investing in micro-optimizing a kernel, multiply its standalone headroom by its **actual served call count per decode cycle** (`continue.md`, "Audit review" & "Kernel register"):

- One solo decode cycle is **~42.5–42.8 ms** (`~38.3 ms` forward + `~4.1–4.2 ms` drafter pass + `~0.3 ms` absorb/logits).
- **Kit-owned (do not touch without full rebuild & 39-case exact gate):** `moe` (40–45% of forward, already 93–96% of scattered-read DRAM roof), `attn` core (4%), `head` (2.5%).
- **Repo-owned decode kernels** (`a_idx` 2%, `a_in`/`a_q` glue, compressor, route) total **~8% of a decode round** (outside `sh_rows`/`sh_fetch` collective waits at 3.7–9.8% and NCCL gathers at 6.6–13.4%).
- **Rule of thumb:** A **2.0 µs** per-layer kernel win across 40 layers equals **0.08 ms** — roughly **0.19%** of a 42.8 ms cycle, which is below the pair-gate noise floor (~0.28–0.64% GPU-clock median/p90). Only pursue body micro-optimizations when either (a) the kernel sits on a multi-hundred-microsecond path (e.g. prompt `topk_indices`, deep indexer, wide prefill chunks, Markov miss), or (b) multiple clean wins stack on a single touch of a file.

---

## 7. Quick-Reference: Proven Wins vs. Verified Anti-Patterns

| Do (Measured Wins on GB10) | Avoid (Measured Regressions on GB10) |
|---|---|
| Vectorize aligned loads (`LDG.E.128` / `float4`) with static trip counts (`-6%` to `-23%`) | Strided 64B-apart per-lane stores (`5.5×` slower) or folding streaming writers into 1-CTA/row kernels |
| Use whole-warp `REDUX` (`0xffffffffu`) for 32-lane max/min | Half-warp masked `__reduce_max_sync` (`+0.10 µs`, 6 static `REDUX`) |
| Replace power-of-two `__fdiv_rn` with bit-derived exact `__fmul_rn` (`-50%` to `-55%` prompt key) | Hand-adding `isfinite` branches in hot loops (`+13%` to `+17%` slower) or CSE'ing `expf` NVCC already merged |
| Keep shared `atomicAdd` (`ATOMS`) simple on `sm_121` | Warp-aggregated `__match_any_sync` histogram (`+2.9` to `+4.4 µs` slower) |
| Let each warp do cheap L1-hit ballots/loads | Adding `__syncthreads()` (`BAR.SYNC`) to deduplicate trivial per-warp work (`+0.03` to `+0.18 µs`) |
| Spread wide work over the grid (`block_max` + `idx_pick` past crossover) | Serializing wide inputs (e.g., 384×5,120 gate or 65K columns) in 1 CTA/row |
| Keep kit Tensor-Core dots/GEMMs | Replacing Tensor-Core work with scalar CUDA-core FP32 math (`-6%` to `-10%` tok/s) |
| Split PDL `PREEXIT` → independent weight load → `ACQBULK`, specialized off the non-PDL path | `launch_dependents` before `wait` with no independent work (`+0.6 µs`) or holding extra regs on non-PDL path |
| Keep `STACK=0 LOCAL=0` and verify working-set reuse in graphs | Assuming 1-buffer graph timing predicts multi-buffer or cold-cache serving (`query128` reversal) |
| Stay off CUDA FP4 intrinsics on pinned `-arch=sm_121` | Emitting emulated FP4 intrinsics on `-arch=sm_121` (`+5%` to `+22%` slower; requires `-arch=sm_121a`) |
