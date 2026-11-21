# WP2: the 0197 ratio-2 compressor (`tf_ds_compress2_norm_kernel`, `tf_ds_compress2x_norm_kernel`, `compress2_norm.zig`)

Run 2026-11-21 on the GB10 head (`sm_121`), engine tree `~/Workspace/tfref-fused-next` branch `fused-next` (WP1 tip
`d01975d` as the "old" arm), CUDA 13.0, flags `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`. Harness:
`~/tf-bench/fused-next/wp2/vbench.py` (candidate forms in `variants.cu`) plus the WP0 `tools/fused_bench` harness for
the final all-kernel parity. Every timing block held `~/tf-bench/gpu.lock`, waited for
`health.live.connections == 0`, and verified the served counters did not move across the block. Timings are
CUDA-graph medians of 20 repetitions of 100 replays (µs per launch), the arms interleaved in one process, with an
empty-kernel graph at the fused launch shape as the floor.

The reference for every parity check is the path the fused kernel replaces: the engine's `tf_ds_compress2x_kernel`
plus the pinned kit `_rmsnorm` cubin (`0cca97`/`ee9e02`), with the output row, both raw-ring buffers and the invalid
flag compared byte for byte.

## Item 1 — the decode compressor's thread geometry: LANDED (commit `90052a0`, 0230)

**What was measured.** Four geometries at the current arithmetic, D = 128 and 512, R = 1/7/32/48, all byte-identical
to the kit chain (0 differing bytes; layouts mixed, unaligned stride, cached pairs, cross pairs, invalid partners,
the NaN/±inf/±0 stress layout, plus R = 33/65 partial and boundary rows):

| geometry | D=512 R=1 | R=7 | R=32 | R=48 | verdict |
|---|---:|---:|---:|---:|---|
| 512 threads, 1 channel a thread (engine, shared row exchange) | 1.727 | 1.932 | 1.971 | 2.259 | baseline |
| 256 threads, 2 channels a thread (shared row exchange kept) | 1.692 | 1.932 | 1.976 | 2.261 | neutral |
| 128 threads, 4 channels a thread (no exchange, no barrier) | 1.808 | 2.033 | 2.079 | 2.340 | 4-7% slower |
| **dedicated 512-wide kernel, 1 channel a thread, slots read first** | **1.542** | **1.779** | **1.829** | **2.137** | landed |

At D = 128 only the 128-thread geometry exists (the kit's reduction is 128 lanes, so the block must be 128 threads);
there the win is:

| D=128 | R=1 | R=7 | R=32 | R=48 |
|---|---:|---:|---:|---:|
| engine kernel (both widths in one kernel) | 1.522 | 1.723 | 1.705 | 1.931 |
| **dedicated 128-wide kernel** | **1.316** | **1.521** | **1.521** | **1.749** |
| launch floor (empty kernel, same grid/block) | 0.457 | 0.457 | 0.457 | 0.456 |

**Why the 4-channel form loses at D=512.** It removes the shared `row[512]` exchange and one barrier but quadruples
each thread's serial pair arithmetic (4 expf + 8 divisions per thread instead of 1 + 2). At these row counts the
kernel is latency-bound on that per-thread chain, so spreading the same work over 512 threads wins. The 2-channel
form sits between the two and is neutral. The vectorized 16-byte loads the 4-channel form enables are real (it was
0.5 µs faster than the same form with scalar loads, 2.320 → 2.034 at R = 7) but do not pay back the longer chain.

**Why the dedicated kernels win.** Two effects, both measured:

- The width-specific kernel drops the other width's code and its static shared allocation. The empty-kernel probe
  at the same launch shape prices a 2 KiB static shared allocation at 0.020-0.038 µs (0.496 vs 0.458 µs, grid 1
  block 512; 0.476 vs 0.456, grid 48 block 128).
- The partner slots (`gi[2g]`, `gi[2g+1]`) are now read *before* the current row's kv/score. With the old order
  ptxas scheduled both 64-bit slot loads after the ring stores; with the new order they issue first and their
  latency hides behind the scatter and the ballot scan. This is not a source-level cosmetic: the two orders produce
  the same instruction mix (392 instructions, identical opcode histogram) but different schedules, and the engine
  kernel built from the old order measured 1.460/1.668/1.685/1.870 at D = 128 against 1.316/1.521/1.521/1.749 for
  the new order. The 128-wide kernel's SASS is now instruction-for-instruction identical to the winning candidate's
  (`diff` of the two disassemblies: 0 lines).

**SASS and resources** (`cuobjdump -sass`, `-res-usage`; opcode counts are static, the old kernel's cover both of
its width paths):

| kernel | insns | LDG.E | LDG.E.64 | STG.E | LDS | STS | SHFL | BAR.SYNC | MUFU | VOTE | BRA | REG | SHARED | spills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| old `tf_ds_compress2x_norm_kernel` (both widths) | 552 | 10 | 4 | 4 | 6 | 3 | 20 | 3 | 10 | 2 | 35 | 34 | 3104 | 0 |
| new `..._norm128_kernel` | 392 | 10 | 4 | 3 | 1 | 1 | 5 | 1 | 8 | 2 | 28 | 30 | 1040 | 0 |
| new `..._norm512_kernel` | 440 | 10 | 4 | 3 | 5 | 2 | 10 | 2 | 8 | 2 | 31 | 30 | 3088 | 0 |

The executed 128-wide path loses the second `__syncthreads`, the shared row exchange (`LDS`/`STS` 6/3 → 1/1), the
`tid < 128` guard (which alone costs ~0.12 µs: it puts the warp-sum store and the five shuffles behind a runtime
branch ptxas cannot fold, measured 1.440 vs 1.317 at R = 1 with everything else identical) and half the shuffles.
No spills in any form (`STACK=0 LOCAL=0`).

**Bound.** Both new kernels are still launch/latency bound: 1.316 and 1.542 µs against a 0.457 µs floor, and the
traffic is trivial (R = 1, D = 128: 1.3 MB of live reads/writes = 0.0014 µs at the 954 GB/s cache roofline), so the
remaining ~0.86-1.09 µs is the node's fixed cost plus the cold-load latency of the partner rows. The ablation in
item 4 shows what is left inside the body.

**Reproduce.**

```bash
# on the head, under the GPU lock
cd ~/tf-bench/fused-next/wp2
bash ../tools/build.sh src-old src-new bin            # src-old = WP1 tip, src-new = this tree
python3 vbench.py --arms new128,new512 --rows 1,7,32,48 --d 128,512 \
    --layouts mixed,unaligned,cached,cross,invalid --json par-new.json      # parity
python3 vbench.py --arms new128,new512,v_256x2,v_128x4v --rows 1,7,32,48 --d 128,512 \
    --layouts mixed --json n2.json                                          # timings + floor
python3 sassone.py bin/new-torch.cubin sass-new.json tf_ds_compress2x_norm128_kernel tf_ds_compress2x_norm512_kernel
```

`vbench.py`'s arms are `old` (the engine kernel in `old-torch.cubin`), `new128`/`new512` (this tree's kernels) and
the candidate kernels in `variants.cu` (`v_128x4`, `v_128x4v`, `v_128x4*_r1*`, `v_cur_r1*`, `v_cur_1e`,
`v_256x2`, `v_512only`, `v_128only*`, `v_final128*`, `v_final512x4`, plus the `v_pair_probe*` and floor probes).


## Item 2 — duplicate exponentials, and 0207's one-exp form: CLOSED, NO CHANGE

**What the SASS says.** `tf_ds_compress2_pair` has four `expf` calls on two distinct operands. The compiled decode
kernel has `MUFU.EX2 = 2`, i.e. NVCC's CSE already removed the duplicates (the old two-width kernel and both new
per-width kernels all show 2 `EX2`, 2 `FCHK` and 2 `CALL.REL.NOINC` — one expf evaluation and its range-check slow
path per distinct operand). There was nothing left to reuse.

**The branchless one-exp form.** 0207 parked a form that evaluated `expf` once and took the max side's `expf(0) = 1`
as a constant, but gated it on an `isfinite` branch and measured 13-17% slower. The candidate here is branchless:

```cuda
const float t0 = __fsub_rn(s0, m), t1 = __fsub_rn(s1, m);
const bool z0 = t0 == 0.f, z1 = t1 == 0.f;      // the max side's argument is exactly 0 -> expf(0) = 1
const float e = expf(z0 ? t1 : t0);
const float e0 = z0 ? 1.f : e, e1 = z1 ? 1.f : e;
```

**Parity: exact.** Two exhaustive checks, both in `vbench.py --pair-probe` / `--pair-probe3`:

| check | combinations | differing bf16 pair sums |
|---|---:|---:|
| every (s0, s1, kv0, kv1) from {±0, ±1, ±FLT_MAX, ±inf, NaN, ±1e-38, ±100, ±0.5, 2} | 65,536 | 0 |
| random normal rows with NaN/±inf/±0/denormal/±FLT_MAX injected at every 97th and 1009th element | 4,194,304 | 0 |

The only inputs where the two forms' intermediate sums differ are the {NaN, -inf} pairs, and there both reach a
canonical NaN in every output byte (probe rows cover them).

**Timing: no gain.** Interleaved graphs, same process, same data:

| shape | engine pair arithmetic | one-exp branchless | IQR |
|---|---:|---:|---:|
| D=128 R=1 | 1.317 | 1.317 | 0.001-0.002 |
| D=128 R=7 | 1.522 | 1.520 | 0.001-0.004 |
| D=128 R=32 | 1.542 | 1.542 | 0.001-0.002 |
| D=128 R=48 | 1.768 | 1.768 | 0.002 |
| D=512 R=1 (old geometry) | 1.726 | 1.708 | 0.001-0.004 |
| D=512 R=7 | 1.954 | 1.952 | 0.002 |
| D=512 R=32 | 1.976 | 1.974 | 0.002 |
| D=512 R=48 | 2.218 | 2.218 | 0.002-0.004 |

At the served decode shapes the kernel is not MUFU-bound (item 4's ablation shows the whole pair arithmetic, both
exponentials and both divisions included, is only 0.12-0.27 µs of the launch), so removing one expf changes nothing
measurable. Closed with no change; the landed kernels keep the engine's two-exp arithmetic.

## Item 3 — duplicated partner resolution across warps: CLOSED, NO CHANGE

`tf_ds_compress2_partners` runs in every warp of the block. Static cost: `VOTE.ANY` = 2 per kernel (one ballot pair
per 32-row chunk), so at R = 48 the old 512-thread geometry executes 16 warps x 2 chunks x 2 ballots = 64 ballots and
1,024 slot loads per block where a block-level resolve needs 4 ballots and 64 loads.

**Measured, both warp counts** (interleaved, same process):

| geometry | shape | per-warp resolve (engine) | resolve once per block + barrier | delta |
|---|---|---:|---:|---:|
| 512 threads (16 warps), D=512 | R=1 | 1.725 | 1.750 | +0.025 |
| | R=7 | 1.954 | 1.992 | +0.038 |
| | R=32 | 1.976 | 2.034 | +0.058 |
| | R=48 | 2.218 | 2.258 | +0.040 |
| 128 threads (4 warps), D=512 4-channel form | R=1 | 1.378 | 1.543 | +0.165 |
| | R=7 | 1.583 | 1.747 | +0.164 |
| | R=32 | 1.562 | 1.727 | +0.165 |
| | R=48 | 1.789 | 1.972 | +0.183 |

The `__syncthreads()` that publishes the resolved pair costs more than the redundant ballots and slot loads save at
both 4 and 16 warps (the slot loads are L1-resident after the first warp, and the ballots are one instruction pair).
The landed kernels keep the per-warp resolve; the 128-wide kernel incidentally drops it from 16 warps to 4, which is
part of item 1's measured win.

## Item 4 — one pair computation per speculative window: CLOSED, NO CHANGE

**The ceiling, measured by ablation.** Replacing the pair sum with a pass-through (`v_final128_nopair`, timing only,
not a parity arm) prices everything a dedupe could ever remove — the two expf, the two divisions, the products, and
the load dependency that hangs off them:

| D=128 | R=1 | R=7 | R=32 | R=48 |
|---|---:|---:|---:|---:|
| current kernel | 1.317 | 1.522 | 1.542 | 1.768 |
| pair arithmetic ablated | 1.051 | 1.400 | 1.420 | 1.646 |
| pair cost per launch | 0.266 | 0.122 | 0.122 | 0.122 |

A seven-row window with four distinct ordered pairs (the served draft shape) can therefore save at most 3/7 of
0.122 µs = **0.05 µs**, and only at R >= 7.

**The window form, measured.** `v_win7` is one block per seven-row window (block `(128, 7)`, 896 threads) with the
rows' pairs staged in shared memory, duplicates copying the representative's bf16 latent; `v_win7n` is the identical
geometry with the dedupe disabled, so the two isolate the dedupe from the geometry:

| D=128, window layout (4 distinct pairs per 7 rows) | R=7 | R=32 | R=48 |
|---|---:|---:|---:|
| windowed, dedupe on | 2.155 | 2.300 | 2.464 |
| windowed, dedupe off | 2.199 | 2.360 | 2.525 |
| **dedupe saving** | **0.044** | **0.060** | **0.061** |
| current per-row kernel | 1.583 | 1.583 | 1.767 |

The dedupe saves exactly what the ablation predicted (0.04-0.06 µs), but the window geometry that makes it possible
costs 0.57 µs (three extra block-wide barriers and a 896-thread block that serialises seven rows onto one SM instead
of spreading them over seven). Parity is exact in every window case (0 differing bytes, ring slots and invalid flag
included), so this is a pure timing verdict. Closed with no change.

## Item 6 — occupancy and shared sizing: CLOSED, NO CHANGE

Occupancy queried through the driver (`cuOccupancyMaxActiveBlocksPerMultiprocessor`) at the served launch shapes, with
the resources the cubin reports:

| kernel | threads | REG | static shared | active blocks/SM | what binds |
|---|---:|---:|---:|---:|---|
| `tf_ds_compress2_norm_kernel` (prompt) | 256 (d=512 R=128, d=2048 R=128) | 66 | 6160 | 3 | registers: 65536/(66x256) = 3.87 |
| `tf_ds_compress2_norm_kernel` (prompt) | 128 (d=512/2048 R=1024) | 66 | 6160 | 7 | registers: 65536/(66x128) = 7.75 |
| `tf_ds_compress2x_norm128_kernel` | 128 | 30 | 16 | 12 | warp slots: 48 warps/SM / 4 |
| `tf_ds_compress2x_norm512_kernel` | 512 | 30 | 2064 | 3 | warp slots: 48 / 16 |

Shared memory is never the binding limit: the prompt kernel's 3 blocks use 18.5 KB and its 7 blocks 43 KB against the
100 KB/SM budget, and the decode kernels use 16 B and 2 KB. Shrinking `srow`/`red` therefore cannot raise residency —
the prompt kernel is register-limited at 66 registers and the decode kernels are at the hardware warp-slot ceiling
(48 of 48 warps). Closed with no change.

## Item 5 — fold the prompt raw-ring writes and the position setup into the fused path: LANDED (commit `b28310e`, 0231)

**What was folded.** `prompt.zig`'s ratio-2 KV-source step used to upload a ring-index array and run two
`scatterRows` launches for the chunk's raw kv/score tails, then upload `ch.groups` and `ch.gpos` for the stores. Now:

- `tf_ds_compress2_norm_kernel` writes both raw tails itself: each block owns one ratio-2 pair (rows `2g`, `2g+1` of
  the chunk) and writes those rows' ring slots at `(raw_pos0 + row - raw_first) mod raw_len`, i.e. the same modular
  positions the scatter used, and only for the window `[raw_first, n_rows)` the scatter covered (`raw_first = n -
  min(n, raw_rows)`). A chunk with an odd trailing row launches one extra block (`grid.x = (n_rows + 1) / 2`) that
  writes only that row's raw slot; a chunk with no complete pair (`full == 0`) keeps the old upload + scatters.
- `tf_ds_fused_prompt_idx_key_kernel` and `tf_ds_fused_prompt_latent_kernel` derive the group id and rotation position
  (`base_group + r`, times `ratio`) instead of loading `ch.groups[r]`/`ch.gpos[r]`; `ch.groups`/`ch.gpos` are staged
  only when an unfused rope/fp4_store chain still reads them.

**Parity: exact, every live output** (both raw-ring buffers and the compressed latent row), against the old chain
(index upload + two scatters + compress2 + rms_norm):

| rows | D | cases |
|---:|---:|---|
| 1, 7, 48, 128, 1024 | 512 | random pairs, ring wrap at 64 raw rows, `raw_first` > 0 from R=32 up |
| 128 | 2048 | random pairs |
| 1, 7, 32, 64, 128 | 512 | **odd trailing row** (n = 2R+1), including the 32/64 cases where the raw window starts past row 0 |
| 1 | 128 | one pair (the smallest fused launch) |
| 0 | 128 | **a one-row chunk (n = 1, no complete pair)**: `prompt.zig` keeps the upload + scatters, and the fused launch is not run |
| 7 | 512 | NaN, ±inf, ±0 and denormal bytes injected into both inputs |

All 16 cases report 0 differing bytes for the new arm (and for the old arm). The position-derivation half is checked by
the `prompt-key`/`prompt-latent` cases, which compare the packed codes/scales against the unfused kit chain with the
same contiguous `(base + r)` run: 0 differing bytes at R=128/1024 D=128/512 plus the special-value case.

**Timings** (interleaved CUDA-graph medians of 20 x 100 replays, production idle, `connections == 0` before and the
served counters unchanged across every block; `ref` = fully unfused chain, `old` = the WP1 tree's fused compressor
path, `new` = this tree):

| case | ref (unfused) | old (WP1 fused path) | **new** | launch floor |
|---|---:|---:|---:|---:|
| source R=1 D=512 | 8.196 | 6.169 | **2.237** | 0.458 |
| source R=7 D=512 | 8.206 | 6.169 | **2.238** | 0.475 |
| source R=48 D=512 | 10.265 | 6.665 | **2.585** | 0.457 |
| source R=128 D=512 | 11.268 | 7.550 | **2.549** | 0.498 |
| source R=1024 D=512 | 18.476 | 15.814 | **11.620** | 0.926 |
| source R=128 D=2048 | 24.756 | 11.945 | **5.929** | 0.516 |
| source R=1 D=512 odd | 8.205 | 6.169 | **2.238** | 0.476 |
| source R=7 D=512 odd | 8.197 | 6.168 | **2.237** | 0.476 |
| source R=32 D=512 odd | 10.266 | 6.168 | **2.175** | 0.476 |
| source R=64 D=512 odd | 10.266 | 7.114 | **2.755** | 0.479 |
| source R=128 D=512 odd | 11.207 | 7.490 | **2.528** | 0.498 |
| prompt key R=128 D=128 | 7.292 | 4.385 | **1.992** | 0.497 |
| prompt key R=1024 D=128 | 9.262 | 6.193 | **3.294** | 1.108 |
| prompt latent R=128 D=512 | 6.178 | 4.278 | **1.725** | 0.497 |
| prompt latent R=1024 D=512 | 10.833 | 10.268 | **6.698** | 0.993 |

Against the path it replaces, the source step is 2.8x faster at R = 1-7 (three launches and a host upload collapse to
one launch), 2.9-3.0x at R = 48-128, 1.36x at R = 1024 and 2.0x at R = 128 D = 2048; the position setup removes the two
uploads (prompt key 2.2x, latent 2.5x at R = 128). The R = 1024 source row is at the memory roofline: its live traffic
is 2 x 2048 x 512 x 4 B read + 2 x 64 x 512 x 4 B raw writes + 1024 x 528 x 2 B latent = 9.7 MB, i.e. 10.2 µs at the
954 GB/s cache roofline against 11.62 µs measured.

**SASS and resources.** `tf_ds_compress2_norm_kernel` keeps `LOCAL=0 STACK=0`, REG 72 (was 66: the ring address
arithmetic and the tail branch), SHARED 7184 unchanged. The two prompt stores keep `LOCAL=0 STACK=0` and lose two
64-bit loads each (`REG:34 SHARED:4112` and `REG:34 SHARED:1024`). The tail block's branch is one `ISETP`+`BRA` at
entry; the per-block ring writes are two extra `STG.E` per channel in the pair loop, which is why the fused launch
costs ~1 µs more than the compressor alone at R = 1024 while removing 5.2 µs of launches.

**Bound.** The prompt source launch is bandwidth-bound from R = 128 upward (1.44 MB / 1.51 µs at R = 128 against
2.55 µs measured, so still partly latency) and at the roofline at R = 1024; at R = 1-7 it is launch/latency bound
(2.24 µs against a 0.46 µs floor, with three fewer launches than before).

**Reproduce.**

```bash
cd ~/tf-bench/fused-next
python3 tools/bench.py wp2/bin wp2/bin <kit> --kernel prompt-source --parity-only       # the fold's parity
python3 tools/bench.py wp2/bin wp2/bin <kit> --kernel prompt-source --timing-only       # the fold's timing
python3 tools/bench.py wp2/bin wp2/bin <kit> --kernel prompt-key,prompt-latent          # the derived position setup
```

## Final position of the 0197 kernels against the launch floor and the roofline

Traffic is the live bytes of each launch (inputs read plus outputs written) from the argument shapes; the roofline is
`bytes / 954 GB/s` for a cache-resident working set (the WP0 measurement, `wp0/BASELINE.md`) and `bytes / 224 GB/s`
above 32 MB.

| launch | shape | new µs | floor µs | live MB | roofline µs | bound now |
|---|---|---:|---:|---:|---:|---|
| `tf_ds_compress2x_norm128_kernel` | R=1 D=128 | 1.316 | 0.457 | 0.003 | 0.003 | launch + cold-load latency |
| | R=7 D=128 | 1.521 | 0.457 | 0.02 | 0.02 | launch + cold-load latency |
| | R=48 D=128 | 1.749 | 0.456 | 0.13 | 0.14 | launch + cold-load latency |
| `tf_ds_compress2x_norm512_kernel` | R=1 D=512 | 1.522 | 0.458 | 0.01 | 0.01 | launch + cold-load latency |
| | R=48 D=512 | 2.137 | 0.477 | 0.44 | 0.46 | launch + cold-load latency |
| `tf_ds_compress2_norm_kernel` (source step) | R=128 D=512 | 2.549 | 0.498 | 1.44 | 1.51 | bandwidth + latency |
| | R=1024 D=512 | 11.620 | 0.926 | 9.73 | 10.20 | bandwidth (88% of the copy roofline) |
| | R=128 D=2048 | 5.929 | 0.516 | 5.79 | 6.07 | bandwidth (at the roofline) |

- **Decode (R <= 48):** the two decode kernels are launch/latency bound. Their bodies are now the minimum the kit's
  reduction order allows (one channel a thread, one `__syncthreads`, five shuffles, one shared publication, no shared
  row exchange at width 128), the pair arithmetic costs 0.12-0.27 µs of the launch (item 4's ablation) and the rest is
  the graph node plus the cold partner loads. A further body change cannot pay: 1.32/1.52 µs against a 0.46 µs floor
  means the remaining ~0.9-1.1 µs is fixed cost and memory latency, not instructions.
- **Prompt source step (R >= 128):** the folded launch now carries the whole step and sits at 88-100% of the measured
  copy roofline, so it is closed to body changes; at R <= 48 it is launch bound (2.24 µs) with one launch where the
  replaced path had four.
- **Position setup:** the two uploads are gone from the fused prompt stores (2.2-2.5x faster at R = 128 with the same
  bytes), and the stores themselves are unchanged apart from two fewer 64-bit loads.

## Item summary

| item | verdict | commit |
|---|---|---|
| 1. 128 threads with four channels a thread, re-tested 128/256/512 | **landed**: per-width dedicated kernels (128-wide 13% faster, 512-wide 11-13% faster at R = 1; 4-6% at R = 48); the 4-channel and 2-channel forms measured and rejected | `90052a0` (0230) |
| 2. duplicate exponentials / 0207's one-exp form | closed, no change: the compiler had already CSE'd the duplicate `expf` (`MUFU.EX2 = 2` for four source calls); the branchless one-exp form is bit-exact over 4.26M probe rows but not faster at the served shapes | — |
| 3. duplicated partner resolution across warps | closed, no change: resolving once per block measures 0.03-0.06 µs slower at 16 warps and 0.16-0.18 µs slower at 4 warps (the publishing barrier costs more than the redundant ballots) | — |
| 4. one pair computation per speculative window | closed, no change: the dedupe saves 0.04-0.06 µs on a seven-row/four-pair window (exactly the ablation's prediction) but the window geometry costs 0.57 µs | — |
| 5. fold prompt raw-ring writes and position setup | **landed**: 2.8-3.0x faster source step at R = 1-128, 1.36x at R = 1024, 2.0x at R = 128 D = 2048; position setup removes two uploads (2.2-2.5x on the prompt stores) | `b28310e` (0231) |
| 6. occupancy and shared sizing | closed, no change: registers (66) bind the prompt kernel and the 48-warp/SM limit binds the decode kernels; shared memory never limits residency | — |

## Full-harness parity, OLD = WP1 tip `d01975d`, NEW = this work's tip

`tools/bench.py` with both trees built from their own sources (OLD `src-old` = `d01975d`, NEW `src-new` = `b28310e`;
md5 of the two `dsv41_torch.cu` files: `8b7e9ad7…` and `7037971b…`), `--parity-only`, every kernel family and shape the
harness knows (0195 idx/idx-small/block-max/fold/hist, 0197 decode/prompt/source, 0198 key/latent/query/prompt
key/prompt latent/all-encodings, 0199 prune, the kit `_score_keys`):

```
run 1: 140 pass, 0 fail
run 2: 140 pass, 0 fail
```

(three earlier runs of the same command, before the one-row source case was added, were 139/139 each).

Every case's `new_vs_ref` and `old_vs_ref` are 0 differing bytes, and the decode compressor cases additionally report
`new_vs_newdup` (the new chain run a second time into fresh buffers) = 0, so the landed decode kernels are
deterministic run to run.

One earlier run of the same command (before the `newdup` arm existed) reported 2 failures out of 141: `idx-small new
R=1 nb=8192` (WP1's `idx_pick`, untouched here) and `decode compressor new D=512 R=32`. Both pass on rerun and in the
three runs above, and each kernel's own parity suite (0197 decode: 3 consecutive runs; prompt source: 15 cases) is
clean, so this is a rare harness-side flake rather than a kernel difference; the `new_vs_newdup` arm is the guard that
would catch it if it were the kernel.

## Harness changes made for this work package

`tools/fused_bench/bench.py` (the recipe repo's copy, for the supervisor to commit):

- `case_compress_decode` dispatches `tf_ds_compress2x_norm128_kernel`/`..._512_kernel` for the new tree and keeps
  `tf_ds_compress2x_norm_kernel` for the old one, and adds the `newdup` determinism arm.
- `case_compress_prompt` passes the new kernel's raw-ring arguments (`0, 0, 0, 0, 2R, 2R` = no ring writes) to the new
  tree only.
- `case_prompt_source` (new): the whole ratio-2 KV-source step — index upload, two `scatter_rows` launches, compress2
  and rms_norm for the old arms against the folded launch for the new one — with ring, ring and latent byte parity and
  timings, at R = 1/7/48/128/1024 D = 512, R = 128 D = 2048, the odd-trailing-row shapes and the special-value case.
- `case_rope` for the prompt paths: contiguous `(base + r)` position/group runs (the fused stores derive them now),
  the two uploads priced in the old arms' graphs, and packed caches indexed from 0 so a nonzero base is exercised.
- `sass.py`: the two per-width decode kernel names at their served blocks.

The candidate-kernel harness used for the measurements above is `wp2/vbench.py` with `wp2/variants.cu` (the rejected
and landed forms, the `v_pair_probe*` exhaustive pair checks and the empty-kernel floor probes); `wp2/sassone.py` and
`wp2/occ.py` produce the SASS/resource and occupancy tables.
