# CUDA Kernel Compilation, Block Microbenchmarking & Opcode Evaluation Process (GB10 / `sm_121`)

End-to-end engineering process for compiling CUDA kernels for this repository's 2× NVIDIA DGX Spark (`gb10`, `sm_121`) stack, microbenchmarking each block of instructions with a three-tier attribution workflow, and evaluating the generated SASS opcodes for optimality.

**Required hardware reference:** [GB10 (`sm_121`) Hardware Performance & Kernel Design Guide](gb10-hardware-performance-guide.md) — contains the measured GB10 ceilings, instruction/opcode behaviors, counter quirks, and bit-exactness rules referenced throughout this process.

**Related docs & harness:**
- Standalone harness: [`tools/fused_bench/README.md`](../tools/fused_bench/README.md) (`build.sh`, `bench.py`, `bw.py`, `sass.sh`, `report.py`, `chain.py`)
- Audit & serving-pricing lessons: [`docs/fused-kernel-rebenchmark-audit.md`](fused-kernel-rebenchmark-audit.md)
- Serving measurement definitions: [`docs/measurements.md`](measurements.md)

---

## Process Overview

```text
  Stage 0: Price & Roofline Triage
     (served calls/cycle × headroom over empty floor & GB10 roofline)
                        │
                        ▼
  Stage 1: Exact sm_121 Compilation & Disassembly
     (nvcc -O3 -arch=sm_121 --fmad=false --ftz=false → .cubin, -res-usage, -sass)
                        │
                        ▼
  Stage 2: Bit-Exact Parity & Safety Gate (MUST PASS BEFORE TIMING)
     (0 differing bytes vs unfused/kit + compute-sanitizer + PDL canary)
                        │
                        ▼
  Stage 3: Tiered Instruction-Block Microbenchmarking
     ├── Tier 1: Unperturbed CUDA-Graph Whole-Kernel & Block Ablations  ◄── keep/drop truth
     ├── Tier 2: In-Kernel clock64() / %globaltimer Block Probes        ◄── per-stage cycle split
     └── Tier 3: Nsight Compute (ncu) SASS PC Sampling & Stall Counters ◄── hardware stall proof
                        │
                        ▼
  Stage 4: SASS Opcode & Optimality Evaluation
     (spills/occupancy → opcode checklist → stall-to-action matrix → stopping criteria)
                        │
                        ▼
  Stage 5: Evidence Bundle & Serving-Gate Handoff
     (opt-in TF_DS_* switch + engagement banner → 39-case exact + GPU-clock pair A/B)
```

### Environment & Safety Pre-Flight (Spark Head)

The workstation has **no `nvcc` and no GPU**. All compilation, parity checks, timing, and SASS dumps run on the GB10 head (`gx10-eb0e`, `Justfile` `head := "colum@100.95.49.21"`):

1. **Sync safely:** Use `just fused-bench` or single-file `rsync` into `~/tf-bench/...`. Do **not** run `just sync-head` while local `zig/patches/` diverges from the head's deployed image (see [`continue.md`](../continue.md)).
2. **Memory & lock guards:** Production holds ~115 of 121 GiB on the head and `memguard` kills at `2.5 GiB`. Every standalone timing script must:
   - Require `MemAvailable >= 3.0 GiB`;
   - Acquire `~/tf-bench/gpu.lock` (`fcntl.flock(..., LOCK_EX)`);
   - Verify `http://127.0.0.1:8000/health` reports `live.connections == 0` and unchanged `(prompt_tokens_total, generation_tokens_total)` across every timed block (`timing_clean: true`).
3. **Parity-only exception:** `bench.py --parity-only` (or `PARITY=1 just fused-bench`) skips the idle-serving wait and may run while production serves, provided `MemAvailable >= 3.0 GiB` and peak test allocations stay well under 1 GiB.
4. **Thermal & clock sanity:** Confirm clocks sit near the nominal sustained band (**2,359–2,392 MHz**, under the 2,400 MHz cap) with no throttling before trusting sub-microsecond deltas.

---

## Stage 0 — Price the Target & Classify Its Bound

Before editing a kernel, establish both its **serving ceiling** and its **hardware bound**:

1. **Serving budget check:** Multiply estimated per-launch savings by the kernel's **actual served call count per decode round or prompt chunk** (verify engagement with `TF_DS_FUSED_TRACE=1`, not assumptions). See [Hardware Guide §6](gb10-hardware-performance-guide.md#6-serving-budget-reality-check-price-before-optimizing).
   - A 2 µs saving × 40 layers = `0.08 ms` of a `42.8 ms` solo decode cycle (`0.19%`), below the `~0.28–0.64%` GPU-clock pair-gate spread.
   - Only spend deep instruction-tuning effort where `Δµs × calls` matters at serving scale, or where multiple clean wins stack on a single touch.
2. **Measure the two hardware ceilings on the same board:**
   ```bash
   # On the head, from ~/tf-bench/fused-next
   python3 tools/bw.py bin --json bw.json
   python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
       --kernel <family> --json baseline.json
   bash tools/sass.sh bin --json sass.json
   python3 tools/report.py baseline.json bw.json sass.json --md BASELINE.md
   ```
3. **Classify the bound** (using `tools/fused_bench/report.py` and [Hardware Guide §2](gb10-hardware-performance-guide.md#2-measured-bandwidth--latency-ceilings-rooflines)):
   - `roofline_us = live_bytes / (224 GB/s if live_bytes > 32 MB else 954 GB/s)` (use `231–236 GB/s` for scattered read-only weights).
   - `headroom_us = measured_us - max(empty_floor_us, roofline_us)`.
   - **Launch-bound** (`measured <= 1.10 × empty_floor`): body opcodes will not move wall time; only fusion, PDL overlap, or skipping the launch can help.
   - **Bandwidth-bound** (`roofline >= 0.85 × measured`): arithmetic/opcode changes will not help; only reducing live bytes, improving cache residency, or changing layout can help.
   - **Issue / latency-bound** (everything else with meaningful `headroom_us`): proceed to Stages 1–4.

---

## Stage 1 — Compiling CUDA Kernels & Extracting Artifacts

### 1.1 Pinned Compilation Flags (Never Fast Math)

Every production and benchmark cubin in this repo uses:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 --fmad=false --ftz=false -cubin <src.cu> -o <out.cubin>
```

| Flag | Why It Is Mandatory |
|---|---|
| `-O3` | Full ptxas optimization matching `zig/Dockerfile` release builds. |
| `-arch=sm_121` | Matches the engine's pinned GB10 build target (`SM=121`). **Do not substitute `-arch=sm_121a`** unless explicitly evaluating a build-flag change: FP4 intrinsics emit hardware `CVT`/`F2FP` on `sm_121a` (+10–15% faster) but are software-emulated on `sm_121` (−5 to −22% slower; [Hardware Guide §3.3](gb10-hardware-performance-guide.md#33-arithmetic-transcendentals-mufu-fp4--tensor-cores)). |
| `--fmad=false` | Prevents unintended FMA contraction that breaks bit-exactness against the pinned kit cubins. Explicit `fmaf(...)` / `__fmaf_rn(...)` still emits `FFMA` where the kit contracts (e.g., RoPE). |
| `--ftz=false` | Preserves subnormal IEEE-754 behavior required by parity suites (`NaN`, `±inf`, `±0`, denormals). |

### 1.2 Standard Build, Diagnostic Build & Disassembly Commands

Use `tools/fused_bench/build.sh` for side-by-side old/new builds (also builds `floor.cubin` containing `empty_kernel`, `copy_kernel`, and `delay_copy`):

```bash
# 1. Production-flag cubins (use THESE for all parity & timing measurements)
bash tools/build.sh src-old src-new bin

# 2. Compiler resource report at build time (registers, spills, smem, cmem)
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 --fmad=false --ftz=false \
    -Xptxas -v -cubin src-new/dsv41_torch.cu -o /tmp/check.cubin

# 3. Diagnostic cubin with source line mapping (for nvdisasm & ncu source-level SASS ONLY)
#    NEVER pass -G (disables optimization). Verify -lineinfo produces identical reg/spill counts.
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 --fmad=false --ftz=false \
    -lineinfo -cubin src-new/dsv41_torch.cu -o /tmp/diag-torch.cubin

# 4. Intermediate PTX (to distinguish CUDA frontend codegen from ptxas lowering)
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 --fmad=false --ftz=false \
    -ptx src-new/dsv41_torch.cu -o /tmp/new-torch.ptx
```

### 1.3 Disassembly & Resource Extraction

```bash
# Structured per-kernel opcode histogram + registers/spills/shared + GB10 theoretical occupancy
bash tools/sass.sh bin --json sass.json

# Full SASS disassembly for a single kernel (with byte offsets for scheduling & PDL checks)
/usr/local/cuda/bin/cuobjdump -sass -fun tf_ds_compress2x_norm512_kernel bin/new-torch.cubin > norm512.sass

# Resource summary (REG, STACK, SHARED, LOCAL)
/usr/local/cuda/bin/cuobjdump -res-usage bin/new-torch.cubin

# Control-flow graph + live register pressure per SASS instruction (requires -lineinfo for source interleave)
/usr/local/cuda/bin/nvdisasm -c -g -plr -ndf /tmp/diag-torch.cubin > diag-torch.nvdisasm
```

**Provenance rule:** When benchmarking against a deployed image's kernels (as in `runs/suite-fused-next2-20261121/query128/`), extract the embedded fatbin directly from the image's `tf-dsv41-lanes` executable, record its SHA-256 alongside the kit cubin hashes, and verify parameter counts and PDL launch attributes match serving dispatch before timing.

---

## Stage 2 — Bit-Exact Parity & Safety Gate (Before Any Timing)

Never time a variant until it passes **0 differing bytes** on every live output and **0 errors** under `compute-sanitizer`.

### 2.1 Byte-Parity Matrix

```bash
# Quick parity check for one family (runs even while production serves if MemAvailable >= 3 GiB)
python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
    --kernel <family> --parity-only --json parity.json

# Durable all-kernel + fallback parity check from workstation
PARITY=1 OLD=<old-tree> NEW=<new-tree> OUT=<artifact-dir> JSON=parity.json just fused-bench
```

Every new kernel or variant must include adversarial test inputs covering:
- **Distributions:** random normal, heavy ties, all-equal/flat, all-`−∞`, mixed signs.
- **IEEE-754 edge cases:** `+0.0`, `−0.0`, `±inf`, canonical and payload `NaN`, subnormals, `±FLT_MAX`.
- **Exhaustive small-domain sweeps:** all **65,536 BF16 encodings** for quantization/norm paths (and all `65,536 × 255` `(bf16, E8M0)` pairs when touching FP4 conversion logic).
- **Geometry & alignment edges:** partial last tile/block, odd row/column counts, misaligned strides (`pad=1, 3`), untouched stride-padding bytes (pre-poison outputs with `0x55` / `0xDEADBEEF` before launch), and fallback shapes outside the fast-path gate.

### 2.2 Stream-Ordering & Sanitizer Hygiene

1. **No legacy stream-0 DMA races (Blocker B2):** `tools/fused_bench/bench.py` executes kernels on a `CU_STREAM_NON_BLOCKING` stream. Pageable `cuMemcpyHtoD_v2` transfers `<= 64 KiB` execute asynchronously on legacy stream 0 and **will race** non-blocking streams. Always upload with `cuMemcpyHtoDAsync_v2(..., stream)` + `cuStreamSynchronize(stream)`, and synchronize `stream` before `cuMemcpyDtoH_v2` ([Hardware Guide §4](gb10-hardware-performance-guide.md#4-hardware-counters-profiling-quirks--measurement-traps-on-gb10)).
2. **Sanitizer checks (0 errors mandatory):**
   ```bash
   for tool in memcheck initcheck racecheck; do
       /usr/local/cuda/bin/compute-sanitizer --tool "$tool" \
           python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
           --kernel <exact-family> --shape "<exact-shape>" --parity-only
   done
   ```
   - Verify the output actually printed `PASS` lines (a mismatched `--kernel`/`--shape` filter can silently execute 0 cases and report 0 errors — see B2 report).
   - Require `0 errors` across all three tools. If `racecheck` flags a **warning** on an intra-warp shared read → full-mask `SHFL` → single-lane write, confirm dataflow ordering in the SASS before waiving the warning.
3. **PDL delayed-producer canary (mandatory for any `griddepcontrol` change):**
   ```bash
   python3 tools/chain.py --old bin --new bin-cand \
       --kit /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
       --mode canary --shape "R=48 nb=8192 k=2048" --json canary.json
   ```
   Require `canary_diff == 0` (200 µs `delay_copy` producers before every dependent input) **and** `race_control_diff > 0` (negative control on an independent stream proves the canary catches premature reads).

---

## Stage 3 — Tiered Microbenchmarking of Each Block of Instructions

Measuring the cost of individual instruction blocks inside a GPU kernel is tricky because **any instrumentation can change register allocation, warp occupancy, instruction scheduling, and memory latency hiding**. Use the three tiers below in order: **Tier 1** establishes unperturbed ground truth and block-level ceilings, **Tier 2** splits sequential stages inside a single CTA into cycle budgets, and **Tier 3** pins down the exact hardware stall reason per SASS line.

---

### Tier 1 — Unperturbed CUDA-Graph Whole-Kernel & Block-Ablation Timing (Primary Truth)

Tier 1 is the **only** tier whose wall-clock timings decide keep vs. drop.

#### 1.1 Standard Whole-Kernel Graph Protocol (`tools/fused_bench/bench.py`)

- **Graph batching:** Warm up once, capture `--batch 100` consecutive replays inside a CUDA graph, launch `--reps 20` timed repetitions, and report **median + IQR** in `µs / launch`.
- **Interleaving:** Always interleave `old`, `new`, `ref`, and `floor` (`A B A B ...`) inside one process and one CUDA context to cancel clock/thermal drift. Verify identical-code noise floor is `~0.001 µs` median on cache-resident shapes and `<= 1%` on DRAM-bound shapes.
- **Working-set fidelity:** Match serving's buffer reuse pattern!
  - If a kernel runs repeatedly on **distinct** buffers across layers, benchmark a multi-buffer graph (e.g., 8 distinct buffer pairs); if it reuses one scratch buffer per chunk, benchmark single-buffer reuse. As `query128` proved, 1–4 buffers vs. 8 distinct buffers can **invert** the fusion verdict ([Hardware Guide §3.4](gb10-hardware-performance-guide.md#34-geometry-parallelism-fusion--pdl)).
  - If a reference kernel mutates inputs in place (e.g., `_rope_heads`), capture a `cuMemcpyDtoDAsync` restore node **before** the start event node, with both GPU timing event nodes recorded inside the graph.

#### 1.2 Block-of-Instructions Ablation Method

To measure the **true unperturbed wall-clock cost** of a specific block of instructions (e.g., partner resolution, softmax pair math, radix histogram, warp reduction, scale conversion, or compaction):

1. Create a timing-only variant kernel in a scratch `.cu` file (`variants.cu`, compiled with the exact Stage 1 flags) that removes or replaces **only** the target block.
2. **Prevent dead-code elimination (DCE):** Keep every upstream load live by folding its result into the final output with a single cheap operation (e.g., `val = kv0 + kv1` instead of `expf`/divides), so `ptxas` does not delete the loads that fed the ablated block.
3. **Confirm isolation in SASS before trusting the delta:**
   - Diff `cuobjdump -res-usage` and `cuobjdump -sass` between full and ablated cubins.
   - Verify register count and occupancy did not jump across a cliff (if `REG` dropped from 44 → 30 and unlocked an extra block/SM, the measured delta mixes occupancy gain with block cost!).
   - Verify the upstream `LDG` instructions and schedule remain intact.
4. **Compute the block's ceiling:**
   $$\Delta_{\text{block}} = t_{\text{full}} - t_{\text{ablated}}$$
   - Because latency-hiding overlap is nonlinear, $\Delta_{\text{block}}$ measures the **marginal critical-path cost** of that block inside the real kernel — i.e., the exact maximum speedup any optimization of that block can deliver.
   - Example: Ablating the entire pair arithmetic (`v_final128_nopair`) in `compress2x_norm128` gave `1.522 → 1.400 µs` at `R=7`, proving `Δ_pair = 0.122 µs` and immediately capping any dedupe or `expf` rewrite at `<= 0.05–0.12 µs` (`runs/fused-next-20261121/wp2/RESULTS.md` Item 4).

---

### Tier 2 — In-Kernel `clock64()` / `%globaltimer` Block Cycle Probes

Use Tier 2 when Tier 1 shows an issue/latency-bound kernel with substantial headroom (`> 0.5 µs`) and multiple sequential instruction blocks inside each CTA (e.g., `Stage A: load & unpack` → `Stage B: histogram` → `Stage C: prefix/cutoff` → `Stage D: compact & store`), and you want a **single-run cycle breakdown across every block**.

#### 2.1 Choosing the Timer Register

| Timer | PTX / Intrinsic | Unit & Scope | When to Use |
|---|---|---|---|
| **SM core clock** | `clock64()` / `mov.u64 %0, %clock64;` | SM clock cycles (~0.42 ns/cycle at 2.38 GHz), **per-SM local** | Measuring instruction-block latencies **within one warp or one CTA** (same SM guaranteed). Highest precision. |
| **Global timer** | `mov.u64 %0, %globaltimer;` | Nanoseconds (fixed frequency), **chip-wide monotonic** | Comparing timestamps **across CTAs / SMs**, measuring CTA launch skew, or PDL producer-to-consumer overlap (`build.sh` `delay_copy`). |

#### 2.2 Safe Probe Macros (Preventing `ptxas` Reordering)

A naive `long long t0 = clock64(); ... long long t1 = clock64();` is **unsafe**: `ptxas` freely hoists independent loads above `t0` or sinks pure arithmetic past `t1`, producing negative or wildly understated block cycles.

Enforce ordering at every block boundary with **both** a compiler memory barrier and a hardware data/warp dependency:

```cuda
// Read SM cycle counter without compiler reordering.
__device__ __forceinline__ unsigned long long rd_clock64() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

// Force a value `v` (register result of the block) to settle BEFORE reading the clock.
// The empty asm consumes `v`, creating a true register scoreboard dependency so the
// clock read cannot issue until `v`'s producing instruction (including LDG/LDS/MUFU) completes.
template <typename T>
__device__ __forceinline__ unsigned long long rd_clock64_after(T v) {
    unsigned long long t;
    asm volatile("" :: "r"(*reinterpret_cast<const unsigned int*>(&v)) : "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}
```

**Block-boundary rules:**
1. **Warp-level block** (single warp executes the block): bracket with `__syncwarp(mask);` before `t0`, and `rd_clock64_after(live_reg)` (plus `__syncwarp(mask);` if lanes diverge) at `t1`.
2. **CTA-level block** (multiple warps cooperate via shared memory): bracket with `__syncthreads();` so `t1 - t0` measures the slowest warp's completion including shared writes; note that `__syncthreads()` itself adds barrier cost, so only place it where a barrier already exists or calibrate the extra barrier separately.
3. **Keep probes in registers; store ONLY in the epilogue:**
   ```cuda
   // Inside instrumented probe kernel (timing-only build, NEVER production):
   unsigned long long t0, t1, t2, t3;
   __syncthreads();
   t0 = rd_clock64();

   // --- Block 1: vectorized load + unpack ---
   float acc = block1_load_unpack(...);
   t1 = rd_clock64_after(acc);

   // --- Block 2: warp/CTA reduction ---
   float red = block2_reduce(acc);
   t2 = rd_clock64_after(red);

   // --- Block 3: scale + quantize + store ---
   block3_store(red, out);
   __threadfence_block(); // or __syncthreads() if multi-warp store completion matters
   t3 = rd_clock64();

   // Write probe deltas ONCE at the very end, from lane 0 (or threadIdx.x == 0) only!
   if (threadIdx.x == 0 && probe_out != nullptr) {
       unsigned long long* p = probe_out + (size_t)blockIdx.x * 4;
       p[0] = t1 - t0;  // Block 1 cycles
       p[1] = t2 - t1;  // Block 2 cycles
       p[2] = t3 - t2;  // Block 3 cycles
       p[3] = t3 - t0;  // Total body cycles
   }
   ```
4. **Calibrate probe overhead and check perturbation:**
   - Measure an **empty block** (`t_a = rd_clock64(); t_b = rd_clock64();`) on GB10 and record `probe_floor_cycles` (typically a few tens of cycles for `CS2R SR_CLOCKLO` + sync).
   - Compare `cuobjdump -res-usage` between the uninstrumented and instrumented cubins: each extra `unsigned long long` timestamp consumes **2 registers**. If probing introduces `STACK > 0` or drops `blocks_per_sm`, probe **one block at a time** (2 timestamps = 4 registers) instead of all blocks simultaneously.
   - Check that `sum(block_deltas) / clock_GHz` is consistent with Tier 1's `(kernel_us - empty_floor_us)` across CTAs (report **median, p90, and max** across all CTAs in the grid — max CTA sets wall time!).

---

### Tier 3 — Hardware Counter & SASS-Level PC-Sampling Attribution (`ncu`)

Use Tier 3 when Tier 1 + Tier 2 identify a hot instruction block, and you need **ground-truth hardware stall reasons and dynamic executed instruction counts** without perturbing kernel source.

#### 3.1 GB10 `ncu` Invocation Rules

- Requires **root** on the Sparks (`sudo /usr/local/cuda/bin/ncu ...`).
- **Disable CUDA graphs** (or replay a single eager launch via `--launch-count 1 --launch-skip 20` after warmup); never trust wall-clock timings from an `ncu` run because multi-pass replay perturbs timing.
- Compile the diagnostic cubin with **`-lineinfo`** (never `-G`), verifying identical register/spill counts against the production cubin.
- Hold `~/tf-bench/gpu.lock` and wait for `live.connections == 0`.

#### 3.2 Targeted Metric Commands for GB10 (`sm_121`)

**A. Dynamic opcode execution behind branches** (e.g., proving a warp guard actually skips `SHFL` or `LDG`, since static SASS counts stay unchanged):
```bash
sudo /usr/local/cuda/bin/ncu --target-processes all \
    --kernel-name regex:<kernel_name> --launch-count 1 --launch-skip 20 \
    --metrics smsp__inst_executed.sum,smsp__inst_executed_op_shfl_pred_on.sum,smsp__inst_executed_op_branch.sum,smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_shared_atom.sum \
    python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
    --kernel <family> --shape "<shape>" --parity-only
```

**B. Throughput, issue efficiency & warp-stall breakdown** (identifying *why* a latency/issue-bound block is slow):
```bash
sudo /usr/local/cuda/bin/ncu --target-processes all \
    --kernel-name regex:<kernel_name> --launch-count 1 --launch-skip 20 \
    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
l1tex__t_bytes.sum,\
lts__t_sectors_op_read_lookup_miss_realtime.sum \
    python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
    --kernel <family> --shape "<shape>" --parity-only
```

**C. Per-SASS-line PC sampling & stall attribution** (directly pinpointing which opcodes inside a basic block accumulate warp stalls):
```bash
# Capture a full report with source/SASS correlation
sudo /usr/local/cuda/bin/ncu --set full --import-source yes \
    --target-processes all --kernel-name regex:<kernel_name> \
    --launch-count 1 --launch-skip 20 -o /tmp/profile \
    python3 tools/bench.py bin-diag bin-diag /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins \
    --kernel <family> --shape "<shape>" --parity-only

# Dump per-SASS-instruction stall & executed counts to text/CSV for review
/usr/local/cuda/bin/ncu --import /tmp/profile.ncu-rep \
    --page source --source-level sass --csv > /tmp/profile_sass.csv
```

> **GB10 Counter Warning:** Ignore DRAM/MC byte counters, `lts__d_sectors_fill_device.sum` (reads `0`), and NVML memory utilization (reads `0%`). For off-L2 read bandwidth, use `lts__t_sectors_op_read_lookup_miss_realtime.sum × 32 B` ([Hardware Guide §4](gb10-hardware-performance-guide.md#4-hardware-counters-profiling-quirks--measurement-traps-on-gb10)).

---

## Stage 4 — Evaluating Resulting SASS Opcodes for Optimality

Inspecting SASS is not guesswork — run through the four steps below systematically for every hot instruction block identified in Stage 3.

### Step 4.1 — Hard Resource & Spill Gate

From `tools/fused_bench/sass.sh` / `cuobjdump -res-usage`:

1. **`STACK == 0` and `LOCAL == 0` (mandatory):** Any non-zero value means register spills (`LDL`/`STL` in SASS) or non-constant-indexed local arrays. Fix by reducing live ranges, unrolling fixed-size array indexing, or shrinking per-thread state before doing any micro-tuning.
2. **Occupancy cliff check (`GB10`: 1,536 threads/SM, 65,536 regs/SM, 102,400 B smem/SM):**
   - Check `blocks_per_sm` from `sass.py`. Adding 2 registers is free if it stays under the cliff (e.g., 30 → 32 regs at 512 threads keeps 3 blocks/SM = 100%), but crossing a cliff (e.g., 42 → 43 regs at 256/512 threads) drops occupancy immediately.
   - Remember: at `grid <= 48` CTAs (decode `R = 1..48`), only 1 CTA sits per SM anyway; do **not** sacrifice per-thread ILP or unrolling to chase theoretical occupancy that the grid cannot fill.
3. **Static shared memory check:** Every unused 2 KiB of static shared memory adds `0.020–0.038 µs` to the launch floor (`wp2` Item 1). Split width-specialized kernels when static shared differs substantially (e.g., `norm128` 1,040 B vs `norm512` 3,088 B).

### Step 4.2 — Block-by-Block SASS Opcode Audit Checklist

Examine the disassembly (`cuobjdump -sass` / `nvdisasm -c -g -plr`) for each hot instruction block against this GB10 `sm_121` checklist:

| Opcode Family | What To Look For in SASS | Optimal on GB10 (`sm_121`) | Suboptimal Red Flag |
|---|---|---|---|
| **Global Loads (`LDG`)** | Width (`LDG.E.128`, `.64`, `.32`, `.U16`), stride across lanes, offset position relative to `STG`/`BAR` | Aligned `LDG.E.128` (`float4`) with static unrolled trip count; cold pointer/index loads **hoisted early** before stores and ballots | Scalar `LDG.E` loop over contiguous floats; cold `LDG` scheduled *after* ring `STG` or `VOTE`; runtime `bsize` loop back-edge |
| **Global Stores (`STG`)** | Width & lane stride, coalescing across warp | Coalesced 32-lane contiguous `STG.E` / `STG.E.64` / `STG.E.128` | Strided per-lane stores (`64 B` apart = 5.5× slower); low-CTA kernel absorbing a wide streaming writer |
| **Local Traffic (`LDL`, `STL`)** | Any occurrence | **0 instructions** (`STACK=0 LOCAL=0`) | Any `LDL`/`STL` inside a hot block |
| **Shared Memory (`LDS`, `STS`, `ATOMS`)** | Count per element, bank stride, warp-aggregation overhead | Simple shared `ATOMS` (`sm_121` coalesces intra-warp uniform atomics); direct register reuse instead of `STS`→`BAR`→`LDS` when thread-local | `MATCH` + `POPC` + leader `ATOMS` (slower by 2.9–4.4 µs); unnecessary shared staging buffer |
| **Warp Ops (`SHFL`, `REDUX`, `VOTE`, `WARPSYNC`)** | Mask width, duplicated codegen, executed ratio | Whole-warp `REDUX` (`0xffffffffu`); 4-stage `SHFL.BFLY` for half-warp (16-lane) groups; warp-uniform guard skipping `SHFL` when majority of warps are unrotated | Half-warp masked `REDUX` (ptxas emits 6 static `REDUX`, +0.10 µs slower); unconditional `SHFL` when 14/16 warps do not rotate; guard where only 2/4 warps skip on a latency-bound chain |
| **Barriers & PDL (`BAR.SYNC`, `PREEXIT`, `ACQBULK`)** | Count per pass/tile; exact hex offset order | Minimum `BAR.SYNC` needed for cross-warp correctness; PDL order `@P0 PREEXIT` → independent weight `LDG` → `ACQBULK` → first dependent `LDG`, with non-PDL branching cleanly over it | `BAR.SYNC` added to avoid trivial L1-hit per-warp ballot/loads; `PREEXIT` before `ACQBULK` with zero independent work between them; dependent `LDG` hoisted above `ACQBULK` (data race!) |
| **Divisions & Transcendentals (`MUFU.*`, `FCHK`, `CALL.REL`)** | Hidden `__fdiv_rn` expansions, prologue integer divides, duplicate `EX2` | Bit-extracted scale + exact power-of-two `FMUL` instead of `__fdiv_rn`; 2D grid `(rows, heads)` eliminating prologue integer div; verify NVCC already CSE'd duplicate `MUFU.EX2` | Per-element `MUFU.RCP` + 4–6 Newton `FFMA`/`FMUL` steps when scaling by a power of 2; prologue `IMAD`/`MUFU.RCP` integer divide on `blockIdx.x`; `FCHK`+`CALL` on hot path that wasn't needed |
| **FP4 & Tensor Core vs Scalar (`FSETP`/`FSEL`, `CVT`, `HMMA`)** | Emulated conversion chains, scalar loops replacing GEMM/dot | Keep kit Tensor-Core dots (`HMMA`/`QGMMA`); on pinned `sm_121` keep compact manual E2M1 threshold logic (do not use emulated FP4 intrinsics) | Scalar CUDA-core FP32 dot replacing a Tensor-Core dot (`0267` −6 to −10%); FP4 intrinsics compiled under `-arch=sm_121` (emulated, +5–22% slower) |
| **Branches & Predicates (`BRA`, `BRA.U`, `FSETP`/`ISETP`)** | Uniform vs divergent branches, hot-loop checks | Static unrolled loops (no back-edge `BRA`); uniform `BRA.U` (`UP*`) for warp-invariant paths | Extra `isfinite` / safety predicates inside a hot per-element loop (`0207` +13–17% slower); runtime `tid < 128` guard blocking warp-sum folding (`+0.12 µs`) |

### Step 4.3 — Stall-Reason-to-Action Decision Matrix

Match Tier 3 (`ncu`) dominant stalls + Tier 1 roofline + Step 4.2 SASS findings to the right intervention:

| Dominant Signal | What It Means on GB10 | First Action to Try | What NOT to Do |
|---|---|---|---|
| **Roofline: Launch floor** (`<= 1.10× empty`) | Fixed launch/graph-node overhead dominates | Fuse adjacent tiny launches, drop unused static shared memory, or apply split PDL (`0250`) | Micro-tuning opcodes inside the body |
| **Roofline: Bandwidth** (`>= 85%` of 224/236/954 GB/s) | Memory bus or L2 copy pipe is saturated | Reduce live bytes (early-exit radix `0254`/`0290`, skip dead outputs `0214`, pack/reuse across layers `0259`) | Adding unrolling, tweaking ALU ops, or chasing occupancy |
| **Stall: `long_scoreboard`** | Warps waiting on global `LDG` latency (L2/DRAM) | Hoist cold `LDG` earlier in source/SASS (`0230`), widen to `LDG.E.128` (`0213`/`0255`), spread wide work across more CTAs (`0215`), or preload immutable weights pre-`ACQBULK` (`0250`) | Adding `isfinite` or shrinking CTA below optimal scan width |
| **Stall: `short_scoreboard`** | Warps waiting on `LDS`, `MUFU`, `SHFL`, or `REDUX` latency | Replace `__fdiv_rn` (`MUFU.RCP` chain) with exact bit-derived `FMUL` (`0240`); replace 4× `SHFL` with whole-warp `REDUX` (`0240`); drop unnecessary `STS`/`LDS` exchange; skip dead `SHFL` on wholly unrotated warps (`0242`) | Using half-warp masked `REDUX` (`+0.10 µs`); adding `BAR.SYNC` to avoid cheap `LDS`/`VOTE` |
| **Stall: `wait` (fixed-latency dependency)** | Single-thread serial ALU chain (e.g., 7-compare E2M1 or 4-channel coarsening) | Keep more threads/fewer channels per thread (`D=512` 512×1 beat 128×4 in `0230`); break serial dependency chains (tree reduce, independent accumulators); evaluate `sm_121a` if FP4 conversion dominates | Thread coarsening on latency-bound decode kernels |
| **Stall: `mio_throttle` / `lg_throttle`** | Shared/LSU/special-function or global load/store issue queue full | Vectorize `LDG`/`STG` to cut transaction count; fix 64B-strided `STG` (`0206`); reduce redundant `MUFU` or `LDS` traffic | Warp-aggregating `ATOMS` with `__match_any_sync` (`+2.9–4.4 µs`) |
| **Stall: `barrier`** | Warps idling at `BAR.SYNC` due to load imbalance or excess barriers | Remove unnecessary `BAR.SYNC` (`0230`, `0214` mask-only skip); replace serial thread-0 bin search with warp-0 parallel search (`0254`); equalize work per warp | Adding a block-level sync to deduplicate cheap L1-cached warp work |
| **Stall: `math_pipe_throttle`** | ALU/FP32/INT32 execution pipe saturated | Reduce instruction count per element (bit ops, static constants, strength reduction, algebraic early-out); move matrix/dot work back to Tensor Cores (`0267` lesson) | Replacing Tensor-Core cubins with scalar CUDA-core kernels |

### Step 4.4 — Optimality Stopping Criteria

Stop iterating and declare a kernel **practically optimal on GB10 (`sm_121`)** once **any** of the following holds:

1. **Launch-floor bound:** `measured_us <= 1.10 × empty_floor_us` (or within `~0.2 µs` of floor).
2. **Bandwidth-roofline bound:** `achieved_GBps >= 0.85 × applicable_roofline` (`224 GB/s` DRAM copy, `231–236 GB/s` scattered read, or `954 GB/s` cache copy) and live bytes are already minimal for the mathematical contract.
3. **Clean latency floor:** `STACK == 0`, `LOCAL == 0`, vectorized aligned loads where alignment allows, no hidden `__fdiv_rn`/prologue-div/half-warp-`REDUX`/strided-`STG`, cold loads hoisted, minimal barriers, **and** `(remaining_headroom_us × served_calls_per_cycle) < 0.25%` of the serving cycle.

---

## Worked Examples from This Repository

These four cases illustrate how Stages 0–4 operated on real GB10 `sm_121` kernels in this repo.

### Example A — Eliminating Hidden `MUFU.RCP` Divisions & Auditing `REDUX` Codegen (`0240`, WP3)

- **Stage 0 (Triage):** `tf_ds_fused_prompt_idx_key_kernel` (`R=1024`) took `6.174 µs` against a `1.111 µs` launch floor and a `0.35 µs` cache roofline (`0.33 MB`) — clearly **issue/latency-bound** with `~5 µs` of body headroom (`runs/fused-next-20261121/wp3/RESULTS.md` Item 2).
- **Stage 1 & 4 (SASS Inspection):** `cuobjdump -sass` showed `632` instructions, `MUFU=6`, `SHFL=13`, `BRA=39`, `REG=34`. Each element ran `__fdiv_rn(x, scale)` (expanding to `MUFU.RCP` + 4–6 Newton `FFMA`/`FMUL` steps), and the 32-lane max used 4 `SHFL`/`FMNMX` stages.
- **Transformation & Stage 2 (Parity):** Replaced float scale derivation with BF16 bit extraction (`tf_ds_scale_byte_bits`), replaced `__fdiv_rn` with exact power-of-two `__fmul_rn(x, tf_ds_recip_bits(byte))`, and replaced the 32-lane shuffle max with whole-warp `__reduce_max_sync(0xffffffffu, ...)`. Verified **0 differing bytes** across all served shapes plus all 65,536 BF16 encodings.
- **Stage 3 & 4 (Resulting SASS & Timing):**
  - `prompt_idx_key`: `632 → 480` insns, `MUFU 6 → 2`, `SHFL 13 → 3`, `REDUX 0 → 2`, `BRA 39 → 21`, `REG 34 → 30`, `STACK=0`. Runtime: **`4.270 → 1.909 µs` (−55%)** at `R=128`, **`6.174 → 3.087 µs` (−50%)** at `R=1024`.
  - `rope_qd`: `368 → 168` insns, `MUFU 4 → 0`, `SHFL 12 → 1`, `REG 26 → 22`.
  - **Negative control caught by SASS + Tier 1:** Applying the same `__reduce_max_sync` to `rope_store`'s **half-warp** (16-lane) groups made `ptxas` emit **6 static `REDUX`** across 3 duplicated paths and regressed runtime **`1.541 → 1.644 µs` (+0.10 µs)**. Reverted immediately; half-warp groups kept 4 `SHFL.BFLY` stages.

### Example B — Identical Opcode Histogram Scheduling Win & Tier-1 Block Ablation (`0230`, WP2)

- **Stage 1 & 4 (SASS Schedule vs. Histogram):** Splitting `compress2x_norm` into dedicated `norm128` / `norm512` kernels dropped static shared memory (`3,104 → 1,040 B` at `D=128`), `BAR.SYNC` (`3 → 1`), and `LDS`/`STS` (`6/3 → 1/1`). Hoisting partner slot loads (`gi[2g]`, `gi[2g+1]`) ahead of the ring scatters produced the **exact same 392-instruction histogram**, yet improved runtime from **`1.460 → 1.316 µs`** (`R=1`) and **`1.668 → 1.521 µs`** (`R=7`) because `ptxas` scheduled the cold `LDG.E.64` loads before `STG` and `VOTE` (`runs/fused-next-20261121/wp2/RESULTS.md` Item 1).
- **Stage 3 Tier 1 (Block Ablation Saved Dead-End Work):** Replacing the entire pair softmax block (`2× expf`, `2× div`, multiplies) with a live pass-through (`v_final128_nopair`) dropped `R=7` from `1.522 → 1.400 µs` (**`0.122 µs` total block cost**). That single ablation proved:
  1. Hand-optimizing `2× expf → 1× expf` (branchless) could gain `0.000 µs` (confirmed: `1.522 vs 1.520 µs`);
  2. Windowed 7-row deduplication could save at most `3/7 × 0.122 = 0.05 µs` (confirmed: `0.044–0.061 µs` saved, overwhelmed by `+0.57 µs` from extra `BAR.SYNC` and 896-thread CTA serialization → dropped).

### Example C — Vectorized Loads (`0213`) vs. Strided Stores (`0206`) & Hardware-Coalesced `ATOMS` (WP1)

- **Vectorized loads (`0213`):** Templating phase 1 (`tf_ds_block_max_t<8>`) emitted **2× `LDG.E.128`** + 16 unrolled `FSETP`/`FSEL` predicates with **no loop back-edge** (`REG=38` unchanged, `STACK=0`). Runtime dropped **−13.4% to −20.1%** on decode shapes (`10.12 → 8.42 µs` at `R=7, nb=4096`; `runs/fused-next-20261121/wp1/RESULTS.md` Item 2).
- **Isolated store microbenchmark (`storepat.cu`):** Before committing to folding `_score_keys` into `idx_fused`, an isolated 4-kernel store microbenchmark timed the exact store pattern per 32,768-key CTA-row: coalesced 8B `STG` = **`7.5 µs`**, 16B strided `STG.128` = **`21.2 µs`** (plus `STACK=64` spill in the full kernel), 64B-strided 8B `STG` = **`41.7 µs`** (5.5× slower). Combined with 7 CTAs vs. the kit's 224 CTAs, the fold lost at every decode shape and was dropped (`wp1/RESULTS.md` Item 1).
- **Hardware-coalesced shared atomics (`wp1` Item 3):** Replacing shared `ATOMS` with `__match_any_sync` (`MATCH` + `POPC` + leader `ATOMS`) compiled cleanly (`REG=39, STACK=0`) but ran **2.9–4.4 µs slower** (`11.33 vs 8.42 µs` random, `21.51 vs 17.15 µs` ties) because `sm_121` already coalesces intra-warp uniform shared atomics in hardware.

### Example D — Verifying Exact SASS Instruction Offsets for Split PDL (`0250`, WP4)

- **Stage 1 & 4 (SASS Offset Verification):** For split PDL, the histogram is not enough — you must verify the **exact hex instruction order** in `cuobjdump -sass` (`runs/fused-next-20261121/wp4/RESULTS.md` Item 2):
  ```text
  /*0x00a0*/ @!P0 BRA <non_pdl_body>          // default non-PDL path skips preload entirely
  /*0x00b0*/  @P0 PREEXIT                     // griddepcontrol.launch_dependents
  /*0x00c0*/      LDG.E.U16 R?, [w_ptr]       // proven-independent immutable norm weight load
  /*0x00f0*/      ACQBULK                     // griddepcontrol.wait
  /*0x0110*/      LDG.E.64  R?, [rslot_ptr]   // FIRST dependent load strictly AFTER ACQBULK
  ```
- **Stage 2 & 3 (Canary + Chain Timing):** `chain.py --mode canary` (`delay_copy` 200 µs producer) passed with `canary_diff=0` and `race_control_diff > 0`. Putting `PREEXIT` before `ACQBULK` *without* the weight load regressed the chain by `+0.6 µs`; adding the pre-wait weight load won `0.17–0.35 µs` with PDL on while the two-instantiation branch kept the default non-PDL path unchanged (`REG 30 → 32`, occupancy unchanged at 100%, `STACK=0`).

---

## Stage 5 — Evidence Bundle & Handoff to Serving Gates

No standalone kernel win enters production without both a complete microbenchmark record and a serving-level gate:

1. **Required microbenchmark artifact bundle** (saved under `runs/<campaign>/`):
   - Source commit / patch diff and `old`/`new`/kit cubin SHA-256 hashes.
   - Parity JSON (`0 differing bytes`, including fallback and adversarial cases) + `compute-sanitizer` logs (`0 errors`).
   - Interleaved Tier-1 timing JSON (`timing_clean: true`, `timing_retries: 0`, median + IQR, empty floor, working-set basis).
   - `sass.json` (before/after opcode histograms, `REG`, `STACK=0`, `LOCAL=0`, `SHARED`, occupancy) and any Tier-2/Tier-3 probe or `ncu` CSVs.
   - Explicit serving projection: `Δµs × served_calls_per_cycle` vs. the 42.8 ms solo cycle (or prompt chunk ms).
2. **Engine integration rules:**
   - Gate behind an **opt-in `TF_DS_*` switch** (off by default until adopted).
   - Log a **one-time `std.log.info` engagement banner** on first execution (so a null serving result can never be confused with an unengaged switch; `continue.md`, "k265 lesson").
3. **Two-rank serving gate:**
   - Run alternating same-image controls (`ctl1, test_a, ctl2, test_b`) on both Sparks.
   - Require **39/39 exact panel IDENTICAL** against the current exact reference, **sampled identity** (`--sampled` with fixed seeds), and `validate.py --compare` (`concurrent == solo` 8/8).
   - Score decode speed on the GPU clock stretch **`per_round.tail[0]`** via `runs/overnight-20261122/table_gpu.py` (never quantized host `forward_ms_per_round`) plus `pass_ms` / `absorb_ms` for drafter work, and record the adoption/rejection in [`decisions.md`](../decisions.md) and [`continue.md`](../continue.md).
