# WP0 baseline: every fused kernel on `fused-next`, ranked by what is left to gain

Run 2026-11-21 on the GB10 head (`sm_121`), engine tree `~/Workspace/tfref-fused-next` branch `fused-next` at
`3c54a9d`, CUDA 13.0, flags `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`. The timing run was taken with production
idle: the harness waited for `connections == 0` before every timing block and verified the served-token counters did not
move across it (`health before == health after`, `timing_retries: 0`, `timing_clean: true` in `wp0.json`). The harness
held `~/tf-bench/gpu.lock` for the run. All numbers are CUDA-graph medians of 20 repetitions of 100 replays (µs per
launch); `iqr` is the 75th-25th percentile spread.

Raw artifacts in this folder: `wp0.json` (all parity + timing rows), `bench.log`, `sass.json`, `sass.log`, `bw.json`,
`bw.log`, `seeded.log`. Reproduce with `tools/fused_bench` (see `RESULTS.md`).

## Rooflines used

`bw.py` on this GPU, 16-byte grid-stride copy, CUDA graphs, median of 20 x 50 replays:

| copy | us | GB/s |
|---|---:|---:|
| 1 MiB | 2.43 | 863 |
| 8 MiB | 17.59 | 954 |
| 64 MiB | 599.03 | 224 |
| 256 MiB | 2420.41 | 222 |

- **DRAM copy roofline 224 GB/s** (64/256 MiB; L2 is 24 MiB, so these cannot be cache-resident).
- **Cache-resident copy roofline 954 GB/s** (8 MiB; an earlier run measured 1510 GB/s at 8 MiB, so the cache figure
  varies with clocks — 954 is the conservative one used below).
- The roofline for a launch is `bytes / 224 GB/s` when its working set is over 32 MB (DRAM basis), else
  `bytes / 954 GB/s` (cache basis). `bytes` is the live traffic of the fused launch: inputs read plus the outputs it
  writes, from the kernel's argument shapes. It is an upper bound on traffic, so the roofline is a lower bound on time.
- **Launch floor** is an empty kernel (`floor.cubin:empty_kernel`) at the fused launch's grid/block, timed the same way.
  A kernel within ~0.2 µs of it cannot get faster by changing its body.

## Ranked baseline (fused kernel, `fused-next`, served shapes)

Headroom = measured − max(floor, roofline). Positive headroom is what a body/geometry change could still recover.
Bound: "launch floor" if within 10% of the floor, "bandwidth" if the roofline is within 15% of the measured time,
otherwise "issue/latency".

| kernel | shape | us | floor | MB | roof us | GB/s | basis | bound | headroom us |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| 0195-idx | idx R=128 nb=4096 width=32765 k=2048 mask ties | 36.525 | 0.822 | 17.30 | 18.14 | 473.7 | cache | issue/latency | 18.382 |
| 0195-idx | idx R=7 nb=8192 width=65533 k=2048 mask flat | 19.236 | 0.497 | 1.89 | 1.98 | 98.4 | cache | issue/latency | 17.252 |
| 0195-idx | idx R=32 nb=8192 width=65533 k=2048 list ties | 20.506 | 0.496 | 8.65 | 9.07 | 421.9 | cache | issue/latency | 11.434 |
| 0195-idx | idx R=128 nb=4096 width=32765 k=2048 mask random | 28.398 | 0.845 | 17.30 | 18.14 | 609.3 | cache | issue/latency | 10.255 |
| 0195-idx | idx R=7 nb=4096 width=32765 k=2048 list random | 10.123 | 0.498 | 0.97 | 1.02 | 96.3 | cache | issue/latency | 9.101 |
| 0195-idx | idx R=7 nb=2049 width=16389 k=2048 list random | 6.436 | 0.498 | 0.52 | 0.54 | 80.2 | cache | issue/latency | 5.895 |
| 0198-prompt-latent | rope prompt latent R=1024 D=512 | 6.501 | 0.990 | 1.34 | 1.41 | 206.7 | cache | issue/latency | 5.092 |
| 0198-query | rope query R=48 D=128 | 4.021 | 1.214 | 0.79 | 0.82 | 195.6 | cache | issue/latency | 2.807 |
| 0198-prompt-key | rope prompt key R=1024 D=128 | 3.528 | 1.110 | 0.33 | 0.35 | 94.0 | cache | issue/latency | 2.418 |
| 0198-query | rope query R=32 D=128 | 2.978 | 0.926 | 0.52 | 0.55 | 176.1 | cache | issue/latency | 2.052 |
| 0199-prune | prune R=48 dead | 2.301 | 0.458 | 0.00 | 0.00 | 1.2 | cache | issue/latency | 1.843 |
| 0199-prune | prune R=48 | 2.240 | 0.457 | 0.00 | 0.00 | 1.2 | cache | issue/latency | 1.783 |
| 0199-prune | prune R=7 dead | 2.238 | 0.457 | 0.00 | 0.00 | 0.2 | cache | issue/latency | 1.781 |
| 0199-prune | prune R=32 | 2.217 | 0.456 | 0.00 | 0.00 | 0.8 | cache | issue/latency | 1.761 |
| 0199-prune | prune R=7 | 2.199 | 0.459 | 0.00 | 0.00 | 0.2 | cache | issue/latency | 1.740 |
| 0198-prompt-key | rope prompt key R=128 D=128 | 2.134 | 0.498 | 0.04 | 0.04 | 19.4 | cache | issue/latency | 1.636 |
| 0195-idx | idx R=48 nb=8 width=61 k=8 list ties | 2.050 | 0.498 | 0.01 | 0.01 | 6.8 | cache | issue/latency | 1.552 |
| 0195-idx | idx R=32 nb=8 width=61 k=8 list ties | 2.036 | 0.497 | 0.01 | 0.01 | 4.6 | cache | issue/latency | 1.539 |
| 0195-idx | idx R=7 nb=8 width=61 k=8 list random | 2.033 | 0.498 | 0.00 | 0.00 | 1.0 | cache | issue/latency | 1.535 |
| 0195-idx | idx R=7 nb=8 width=61 k=8 list ties | 2.017 | 0.495 | 0.00 | 0.00 | 1.0 | cache | issue/latency | 1.522 |
| 0195-idx | idx R=7 nb=8 width=61 k=8 list flat | 2.013 | 0.498 | 0.00 | 0.00 | 1.0 | cache | issue/latency | 1.515 |
| 0197-compress-decode | compress-decode R=48 D=128 mixed | 1.952 | 0.457 | 0.26 | 0.27 | 133.0 | cache | issue/latency | 1.495 |
| 0197-compress-decode | compress-decode R=7 D=512 mixed | 1.915 | 0.476 | 0.15 | 0.16 | 78.7 | cache | issue/latency | 1.439 |
| 0195-idx | idx R=1 nb=8 width=61 k=8 list ties | 1.851 | 0.477 | 0.00 | 0.00 | 0.2 | cache | issue/latency | 1.374 |
| 0199-prune | prune R=1 | 1.829 | 0.460 | 0.00 | 0.00 | 0.0 | cache | issue/latency | 1.369 |
| 0198-key | rope key R=32 D=128 | 1.788 | 0.457 | 0.01 | 0.01 | 5.8 | cache | issue/latency | 1.331 |
| 0198-key | rope key R=48 D=128 | 1.788 | 0.457 | 0.02 | 0.02 | 8.7 | cache | issue/latency | 1.331 |
| 0198-key | rope key R=7 D=128 | 1.772 | 0.458 | 0.00 | 0.00 | 1.3 | cache | issue/latency | 1.314 |
| 0198-key | rope key R=1 D=128 | 1.768 | 0.457 | 0.00 | 0.00 | 0.2 | cache | issue/latency | 1.311 |
| 0197-compress-decode | compress-decode R=32 D=128 mixed | 1.747 | 0.457 | 0.17 | 0.18 | 99.1 | cache | issue/latency | 1.290 |
| 0197-compress-decode | compress-decode R=7 D=128 mixed | 1.729 | 0.458 | 0.04 | 0.04 | 21.9 | cache | issue/latency | 1.271 |
| 0197-compress-decode | compress-decode R=1 D=512 mixed | 1.727 | 0.458 | 0.02 | 0.02 | 12.5 | cache | issue/latency | 1.269 |
| 0198-prompt-latent | rope prompt latent R=128 D=512 | 1.767 | 0.498 | 0.17 | 0.18 | 95.0 | cache | issue/latency | 1.269 |
| 0197-compress-decode | compress-decode R=32 D=512 mixed | 1.951 | 0.477 | 0.69 | 0.72 | 353.2 | cache | issue/latency | 1.228 |
| 0197-compress-prompt | compress-prompt R=128 D=512 | 2.382 | 0.498 | 1.18 | 1.24 | 497.0 | cache | issue/latency | 1.141 |
| 0197-compress-decode | compress-decode R=48 D=512 mixed | 2.197 | 0.478 | 1.03 | 1.08 | 470.5 | cache | issue/latency | 1.113 |
| 0198-latent | rope latent R=48 D=512 | 1.542 | 0.457 | 0.06 | 0.07 | 40.8 | cache | issue/latency | 1.085 |
| 0198-latent | rope latent R=32 D=512 | 1.541 | 0.457 | 0.04 | 0.04 | 27.2 | cache | issue/latency | 1.084 |
| 0198-latent | rope latent R=1 D=512 | 1.527 | 0.456 | 0.00 | 0.00 | 0.9 | cache | issue/latency | 1.071 |
| 0197-compress-decode | compress-decode R=1 D=128 mixed | 1.522 | 0.457 | 0.01 | 0.01 | 3.6 | cache | issue/latency | 1.065 |
| 0198-latent | rope latent R=7 D=512 | 1.524 | 0.459 | 0.01 | 0.01 | 6.0 | cache | issue/latency | 1.065 |
| 0198-query | rope query R=7 D=128 | 1.564 | 0.518 | 0.11 | 0.12 | 73.3 | cache | issue/latency | 1.046 |
| 0198-query | rope query R=1 D=128 | 1.440 | 0.457 | 0.02 | 0.02 | 11.4 | cache | issue/latency | 0.983 |
| 0197-compress-prompt | compress-prompt R=1024 D=512 | 10.594 | 0.928 | 9.47 | 9.93 | 893.9 | cache | bandwidth | 0.664 |
| 0197-compress-prompt | compress-prompt R=128 D=2048 | 5.513 | 0.498 | 4.72 | 4.95 | 856.6 | cache | bandwidth | 0.561 |
| 0197-compress-prompt | compress-prompt R=1024 D=2048 | 143.302 | 1.028 | 37.78 | 168.62 | 263.6 | DRAM | bandwidth | -25.320 |
| 0195-idx | idx R=1024 nb=4096 width=32765 k=2048 mask ties | 572.901 | 3.844 | 138.42 | 617.76 | 241.6 | DRAM | bandwidth | -44.862 |
| 0195-idx | idx R=2048 nb=4096 width=32765 k=2048 mask ties | 1119.294 | 7.213 | 276.83 | 1235.53 | 247.3 | DRAM | bandwidth | -116.232 |

The 0195 DRAM-bound rows exceed the 224 GB/s *copy* roofline (247 GB/s): the kernel is a read-dominated stream, and a
read stream beats a copy. They are at the memory roofline and cannot get faster by changing the body.

## What the ranking says can still pay

1. **0195 indexer at the mid sizes and on tie-heavy input is the only large recoverable block** (18.4 µs at R=128
   nb=4096 mask/ties, 17.3 µs at R=7 nb=8192 flat, 11.4 µs at R=32 nb=8192 list/ties, 10.3 µs at R=128 random,
   9.1 µs at R=7 nb=4096). It sits at 96–610 GB/s against a 954 GB/s cache roofline and its SASS is
   `LDG.64=2, LDS=12, STS=14, SHFL=20, BAR=8` with `LDG.128=0` — scalar/64-bit loads, eight barriers and a per-row
   single-CTA layout (one block of 1,024 threads per row; at R=7 only 7 CTAs run on 48 SMs). The flat/tie rows are the
   radix selector's tie passes; the R=7 rows are the small-row parallelism limit. These match the open todo items
   "specialize and vectorize aligned served-shape loads", "specialize 0195's block size" and "small-row parallelism for
   0195".
2. **0198 prompt latent/key and query at the larger prompt/decode rows**: prompt-latent R=1024 has 5.1 µs over the
   roofline, prompt-key R=1024 2.4 µs, query R=48 2.8 µs and R=32 2.1 µs. The query fused launch is *slower* than the
   unfused two-launch kit chain at R=32/48 (2.98 vs 2.73 µs; 4.02 vs 3.02 µs) because it launches `rows*heads` blocks
   (1,536 at R=48) where the kit's `_fp4qd_p2` uses 192 — a geometry regression worth pricing before spending elsewhere.
3. **0199 prune is a single-CTA latency bound**: 2.2 µs against a 0.46 µs floor with no bytes moved (one block of
   32/64 threads, `ATOM=6`, `STG=64`). Headroom 1.7–1.8 µs; the audit's stats-on minus stats-off was 0.73 µs a layer,
   so this is the counter half. At 40 layers it is ~90 µs a round (~0.2% of 42 ms), low priority.
4. **The 0197 decode compressor and the 0198 key/latent decode kernels are 1.0–1.5 µs above the floor and ~1.5–2 µs
   total**: they are launch/latency bound at the decode row counts, which is what the previous campaign concluded
   ("measure the decode launch floor before spending on micro-optimization"). There is little left here.
5. **0197 prompt compressors are at the bandwidth roofline** (857–894 GB/s cache at R=128/1024 D=512/2048; 264 GB/s
   DRAM at R=1024 D=2048, above the 224 GB/s copy rate). Closed to body changes.

## Reference (unfused/kit) versus fused, for context

| kernel | shape | ref (unfused/kit) us | fused us | floor us |
|---|---|---:|---:|---:|
| 0195-idx | idx R=1 nb=8 | 20.241 | 1.851 | 0.477 |
| 0195-idx | idx R=7 nb=8 | 26.650 | 2.017 | 0.495 |
| 0195-idx | idx R=32 nb=8 | 26.651 | 2.036 | 0.497 |
| 0195-idx | idx R=48 nb=8 | 26.615 | 2.050 | 0.498 |
| 0195-idx | idx R=7 nb=4096 list/random | 44.687 | 10.123 | 0.498 |
| 0195-idx | idx R=32 nb=8192 list/ties | 61.791 | 20.506 | 0.496 |
| 0195-idx | idx R=128 nb=4096 mask/ties | 158.962 | 36.525 | 0.822 |
| 0195-idx | idx R=1024 nb=4096 mask/ties | 1764.924 | 572.901 | 3.844 |
| 0195-idx | idx R=2048 nb=4096 mask/ties | 3534.455 | 1119.294 | 7.213 |
| 0195-idx | idx R=7 nb=2049 list/random | 44.398 | 6.436 | 0.498 |
| 0195-idx | idx R=7 nb=8192 mask/flat | 46.791 | 19.236 | 0.497 |
| 0197-compress-decode | R=1 D=128 | 2.997 | 1.522 | 0.457 |
| 0197-compress-decode | R=1 D=512 | 3.406 | 1.727 | 0.458 |
| 0197-compress-decode | R=7 D=128 | 3.528 | 1.729 | 0.458 |
| 0197-compress-decode | R=7 D=512 | 3.878 | 1.915 | 0.476 |
| 0197-compress-decode | R=32 D=128 | 6.314 | 1.747 | 0.457 |
| 0197-compress-decode | R=32 D=512 | 6.646 | 1.951 | 0.477 |
| 0197-compress-decode | R=48 D=128 | 7.892 | 1.952 | 0.457 |
| 0197-compress-decode | R=48 D=512 | 8.323 | 2.197 | 0.478 |
| 0197-compress-prompt | R=128 D=512 | 6.376 | 2.382 | 0.498 |
| 0197-compress-prompt | R=128 D=2048 | 19.159 | 5.513 | 0.498 |
| 0197-compress-prompt | R=1024 D=512 | 13.440 | 10.594 | 0.928 |
| 0197-compress-prompt | R=1024 D=2048 | 171.555 | 143.302 | 1.028 |
| 0198-key | R=1/7/32/48 D=128 | 2.898–2.956 | 1.768–1.788 | 0.457 |
| 0198-latent | R=1/7/32/48 D=512 | 2.054–2.075 | 1.524–1.542 | 0.456 |
| 0198-query | R=1/7/32/48 D=128 | 2.144/2.158/2.731/3.017 | 1.440/1.564/2.978/4.021 | 0.457–1.214 |
| 0198-prompt-key | R=128/1024 D=128 | 4.246/5.887 | 2.134/3.528 | 0.498/1.110 |
| 0198-prompt-latent | R=128/1024 D=512 | 3.560/7.707 | 1.767/6.501 | 0.498/0.990 |
| 0199-prune | R=1/7/32/48 | 5.044/8.670/9.284/9.284 | 1.829/2.199/2.217/2.240 | 0.460/0.457 |
| 0199-prune | R=7/48 dead | 8.690/9.737 | 2.238/2.301 | 0.457/0.458 |

## The kit's `_score_keys` launch, at 0195's fold widths (nothing had priced it)

Launched the way `tri_index.scoreKeys` records it: grid `(rows, cdiv(width, 1024))`, 128 threads, 4 KiB shared, the
`HAS_TMAX=1` cubin (`df85d0…`), tmax `[rows, cdiv(width,64)]`. Widths are `% 4 == 0` (the kit faults otherwise). Median
of 20 x 100 graph replays, production idle.

| rows | width | us | iqr | MB | GB/s | basis |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 16,384 | 1.088 | 0.006 | 0.20 | 183 | cache |
| 1 | 32,768 | 1.152 | 0.002 | 0.40 | 345 | cache |
| 1 | 65,536 | 1.524 | 0.011 | 0.79 | 521 | cache |
| 7 | 16,384 | 1.952 | 0.002 | 1.39 | 712 | cache |
| 7 | 32,768 | 2.939 | 0.015 | 2.78 | 946 | cache |
| 7 | 65,536 | 4.943 | 0.012 | 5.56 | 1,125 | cache |
| 32 | 16,384 | 5.579 | 0.007 | 6.36 | 1,139 | cache |
| 32 | 32,768 | 20.149 | 0.199 | 12.71 | 631 | cache |
| 32 | 65,536 | 91.016 | 3.935 | 25.43 | 279 | DRAM |
| 128 | 16,384 | 99.965 | 0.317 | 25.43 | 254 | DRAM |
| 128 | 32,768 | 231.624 | 0.439 | 50.86 | 220 | DRAM |
| 128 | 65,536 | 471.906 | 0.491 | 101.71 | 216 | DRAM |

The kit's `_score_keys` is a pure streaming writer once it leaves L2: 216–220 GB/s at R=128, i.e. at the 224 GB/s copy
roofline. At R=128 width 32,768 it costs 231.6 µs, which is consistent with the previous campaign's fold measurement
(0195 with 0206's key fold 316.6 µs against 38.5 µs without at the same width: the fold adds the keys/tmax writes the
kit also does). This tree does not carry 0206/0207/0208, so `_score_keys` is priced here only to size the later fold.

## SASS and resources (baseline; old and new cubins are md5-identical on this tree)

`cuobjdump -sass` / `-res-usage`, per kernel, at the served launch block. `LDG.128=0` for every kernel: no 16-byte
global loads anywhere. No spills (`STACK=0 LOCAL=0`).

| kernel | insn | LDG | LDG.128 | LDG.64 | LDG.U16 | STG | STG.128 | LDS | STS | SHFL | BAR | ATOM | MUFU | BRA | REG | SHARED | block | occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| idx_fused | 632 | 3 | 0 | 2 | 0 | 2 | 0 | 12 | 14 | 20 | 8 | 1 | 4 | 33 | 38 | 2208 | (1024,1) | 1 block/SM, 66.7% |
| compress2_norm | 5136 | 172 | 0 | 0 | 36 | 36 | 0 | 121 | 52 | 6 | 24 | 0 | 152 | 190 | 66 | 7184 | (32,8) | 3 blocks/SM, 50.0% |
| compress2x_norm | 552 | 16 | 0 | 4 | 2 | 4 | 0 | 6 | 3 | 20 | 3 | 0 | 10 | 35 | 34 | 3104 | (512,1) | 3 blocks/SM, 100% |
| fused_idx_key | 496 | 8 | 0 | 2 | 2 | 3 | 0 | 4 | 3 | 21 | 3 | 0 | 6 | 30 | 30 | 2080 | (128,1) | 12 blocks/SM, 100% |
| fused_rope_store | 488 | 8 | 0 | 2 | 2 | 3 | 0 | 0 | 0 | 20 | 0 | 0 | 2 | 31 | 34 | 1024 | (256,1) | 6 blocks/SM, 100% |
| fused_rope_qd | 368 | 4 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 12 | 0 | 0 | 4 | 33 | 26 | 1024 | (128,1) | 12 blocks/SM, 100% |
| fused_prompt_idx_key | 632 | 20 | 0 | 2 | 14 | 3 | 0 | 4 | 8 | 13 | 4 | 0 | 6 | 39 | 34 | 4112 | (32,2) | 24 blocks/SM, 100% |
| fused_prompt_latent | 488 | 8 | 0 | 2 | 2 | 3 | 0 | 0 | 0 | 20 | 0 | 0 | 2 | 31 | 34 | 1024 | (256,1) | 6 blocks/SM, 100% |
| prune_six | 1344 | 26 | 0 | 0 | 0 | 64 | 0 | 5 | 4 | 0 | 2 | 6 | 16 | 137 | 48 | 1168 | (64,1) | 21 blocks/SM, 87.5% |

`compress2_norm`'s static counts cover five `ITEMS` specialisations plus the generic path; the executed path is one of
them. `prune_six` has no SHFL (warp reductions use `RED`/`ATOMS`), six atomics and 64 stores.

## Bounds summary

- **0195 at R≥1024** (nb=4096/8192): DRAM bandwidth, 240–247 GB/s (at/above the 224 GB/s copy rate). No body headroom.
- **0195 at R≤128, and nb=8**: issue/latency, 18.4 µs (R=128 ties) down to 1.4 µs (R=1). One CTA per row, eight
  barriers, scalar/64-bit loads, radix tie passes. The largest recoverable block in the fused set.
- **0197 prompt compressors**: bandwidth, 857–894 GB/s cache / 264 GB/s DRAM; closed.
- **0197 decode compressor, 0198 key/latent/query decode, 0199 prune**: launch/latency, 1.0–2.8 µs above the floor;
  the floor is 0.46–1.2 µs. Little to gain except the query geometry regression at R=32/48 and prune's counters.
- **0198 prompt key/latent at R=1024**: 2.4/5.1 µs over the roofline; the prompt latent is the second-largest
  recoverable item after 0195.
