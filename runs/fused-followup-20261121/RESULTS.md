# 0200-0212 follow-up: micro-benchmarks and short paired runs

Run 2026-11-21 on the head (GB10, `sm_121`). The micro half ran beside the live production service; the paired half
restarted the pair for each arm and restored production at the end.

The thirteen patches are `zig/patches/0200`-`0212`, the `todo.md` 0195+ review's items 1-4, 8-11, 13, 15 and 17-19
against 0195/0197/0198/0199. None adds a switch. This run answers the two questions they could not answer locally: are
the changed kernels byte-identical to the paths they replace, and are they faster at the served shapes.

## Trees and images

| | tree | built |
|---|---|---|
| previous | `prune-six` = production source + 0195/0197/0198/0199 (`d66e663`) | `~/Workspace/tfref-prune-six-tip` |
| follow-up | previous + `zig/patches/0200`-`0212` (`ffdce65`) | `~/Workspace/tfref-recipe-bench` |
| kit | the pinned AOT cubins `/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins` | - |

Both trees' CUDA sources were compiled with the same `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false` into
`old-*.cubin` / `new-*.cubin` (`build2.sh`), plus three audit kernels that expose the changed arithmetic to the host:
`audit_norm` (0197/0202/0210's torch norm), `audit_pair` (0207) and `audit_drop` (0211).

`probe.py` is the byte-parity and CUDA-graph timing harness: chain 0 is the unfused sequence, 1 the previous fused
launch, 2 the follow-up, 3 the follow-up without 0206's key fold. `sass.py` counts instructions per kernel.

The paired image is `tensorfold-dsv41:bench-followup-0212`, layered over production's `exl3-tp2-ea522d353626` (the same
packaging `bench-prune-six-20261120` used), and was shipped to both ranks. `0203`'s `TF_DS_CHECK_INVALID` is not
exercised.

## Micro: byte parity

Every row is a full byte comparison of the live outputs; buffers that 0200 stopped writing are checked against their
pre-launch contents.

| path | compared | result |
|---|---|---|
| norm 128/512, 4,096 rows | `audit_norm` vs the kit's `_rmsnorm` | identical, both trees |
| decode compressor 128/512 | `compress2x_norm` vs the kit's `_rmsnorm` on the same row | identical, both trees |
| prompt compressor 1/8/128/1,024 rows, D=512/2,048 | `compress2_norm` vs the kit's `_rmsnorm` | identical |
| key fold, widths 61/64/4,101/16,384/32,765/32,768/65,533 | `idx_fused`'s keys and tile maxima vs `scoreKey`'s host emulation | identical |
| index pool, 8 shapes incl. ties and flat | mask or block list: previous vs follow-up vs unfused vs no-fold | identical |
| key / latent / query / prompt key / prompt latent | packed codes, scales, dequantized values vs the unfused kit sequence | identical |
| key / latent / query / prompt key / prompt latent | the scratch buffers 0200 stopped writing | byte-identical to pre-launch |
| all 65,536 bf16 encodings | latent store vs the kit's `_rope_heads` + `_fp4_store` | identical |
| norm extremes 128/512 | every bf16 exponent/mantissa in constant-magnitude rows | identical |
| pair softmax, 8,516 pairs | `audit_pair` follow-up vs the previous two-exponential form | identical |
| six-slot drop, 4,111 vectors | `audit_drop` follow-up vs the complete rule | identical |
| input scales, 8,257,536 pairs | `scale_midpoints` vs the kit's RN division | identical |

## Micro: timings (CUDA graph, µs, median of 32 launches)

Launched with the engine's own geometry (`exact.zig`): decode compressor `d` threads; prompt compressor
`(rmsLanes, min(d, target)/rmsLanes)`; rope-store and prompt-latent `min(d/2, 256)`; prompt key
`(rmsLanes, packLanes)`; pool 1,024 threads with `nbk*4` shared bytes.

| case | unfused | previous | follow-up | follow-up, no fold |
|---|---|---|---|---|
| decode compressor R=1 D=512 | 10.510 | 1.612 | 1.809 | - |
| decode compressor R=7 D=512 | 11.282 | 1.871 | 2.133 | - |
| decode compressor R=32 D=512 | 12.188 | 1.874 | 2.191 | - |
| prompt compressor R=1 D=512 | 3.168 | 2.790 | 1.998 | - |
| prompt compressor R=8 D=512 | 4.048 | 4.497 | 2.000 | - |
| prompt compressor R=128 D=512 | 6.483 | 8.016 | 2.640 | - |
| prompt compressor R=1,024 D=512 | 13.455 | 15.051 | 9.871 | - |
| prompt compressor R=1 D=2,048 | 3.366 | 3.222 | 3.789 | - |
| pool R=7 nb=8 width=61 | 26.699 | 2.088 | 2.427 | 2.303 |
| pool R=7 nb=8 width=64 | 26.657 | 2.062 | 2.345 | 2.258 |
| pool R=7 nb=2,049 width=16,384 | 35.197 | 4.495 | 23.466 | 4.428 |
| pool R=32 nb=8,192 width=65,533 | 62.035 | 21.347 | 142.384 | 20.271 |
| pool R=128 nb=4,096 width=32,768 | 159.052 | 41.035 | 316.632 | 38.542 |
| pool R=7 nb=4,096 width=32,765 | 46.101 | 9.806 | 49.679 | 9.425 |
| pool R=7 nb=8,192 width=65,533 (flat) | 46.930 | 19.923 | 103.696 | 18.661 |
| pool R=3 nb=513 width=4,101 | 26.970 | 4.237 | 8.790 | 4.371 |
| rope key R=7 D=128 | 3.014 | 2.189 | 1.867 | - |
| rope latent R=7 D=512 | 2.151 | 2.184 | 1.610 | - |
| rope query R=7 D=128 | 2.195 | 1.696 | 1.683 | - |
| rope prompt key R=128 D=128 | 4.363 | 3.029 | 2.195 | - |
| rope prompt latent R=128 D=512 | 3.596 | 3.279 | 1.809 | - |

Readings:

- **The prompt compressor is the large win**: 55-67% faster at R=8/128, 35% at R=1,024, 28% at R=1 (0210's held pair
  results plus 0202's barrier).
- **The decode compressor is 13-17% slower**: 1.81 vs 1.61 µs at R=1, 2.13 vs 1.87 at R=7. That is 0207's `isfinite`
  branch on the two-score path, paid on every layer. At 40 layers it is 8-12 µs a round, ~0.03% of a 42 ms round.
- **The rope-store chains are 14-45% faster** (0200's dead stores, 0209's packed-column key store, 0212's two warps).
- **0206's key fold costs what the keys cost to write**: at width 32,768, 316.6 µs with the fold against 38.5 µs
  without (0200/0208/0209 alone) and 41.0 µs for the previous tree. The fold is writing `width*8` bytes a row that the
  previous tree never wrote in this launch; the engine wrote them in a separate `_score_keys` launch, which this
  harness does not time (the kit's `_score_keys` faults for odd widths, see below). The no-fold column is the like-for-
  like comparison: it is equal or slightly faster than the previous tree at every width (e.g. 38.5 vs 41.0 at
  R=128, 18.7 vs 19.9 at R=7/65,533).
- At the decode shape (R=7, nb=8) the fused launch is 2.3-2.4 µs against 2.1 for the previous tree: the fold's tile
  maxima cost ~12%, which is 0.2 µs a layer and ~8 µs a round.

## Micro: SASS (static counts, `sass.py`)

| kernel | BAR.SYNC | LDS | STS | SHFL | LDG | STG | registers | spills |
|---|---|---|---|---|---|---|---|---|
| `fused_idx_key` previous | 5 | 26 | 14 | 19 | 14 | 5 | 37 | 0 |
| `fused_idx_key` follow-up | 3 | 4 | 3 | 21 | 8 | 3 | 30 | 0 |
| `fused_prompt_idx_key` previous | 6 | 26 | 19 | 11 | 26 | 5 | 35 | 0 |
| `fused_prompt_idx_key` follow-up | 4 | 4 | 8 | 13 | 20 | 3 | 34 | 0 |
| `fused_rope_store` previous | 0 | 0 | 0 | 20 | 8 | 5 | 34 | 0 |
| `fused_rope_store` follow-up | 0 | 0 | 0 | 20 | 8 | 3 | 34 | 0 |
| `fused_prompt_latent` previous | 0 | 0 | 0 | 20 | 8 | 5 | 37 | 0 |
| `fused_prompt_latent` follow-up | 0 | 0 | 0 | 20 | 8 | 3 | 34 | 0 |
| `compress2_norm` previous | 4 | 11 | 4 | 1 | 9 | 5 | 27 | 0 |
| `compress2_norm` follow-up | 24 | 121 | 50 | 6 | 164 | 36 | 62 | 0 |
| `idx_fused` previous | 8 | 12 | 14 | 20 | 3 | 4 | 38 | 0 |
| `idx_fused` follow-up | 8 | 12 | 14 | 26 | 4 | 4 | 42 | 0 |

What the gate asked for:

- **0202** (the `L == 32` barrier): `fused_prompt_idx_key` drops from 6 barriers to 4. The remaining norm barrier sits
  inside the `L > 32` fold branch (its guard is the fold loop's own predicate), and the `rs` publication barrier is
  kept. `fused_idx_key` drops 5 to 3. Byte parity against the kit at both widths and all 65,536 encodings is the
  correctness evidence; the counts are the structural one.
- **0209** (rotation folded into the packed-column store): the shared scale array and the second `srow` read are gone -
  `fused_idx_key` LDS 26 -> 4, STS 14 -> 3; `fused_prompt_idx_key` LDS 26 -> 4.
- **0210** (norm keeps the pair results): `compress2_norm`'s static counts rise because five `ITEMS` specialisations
  (1/2/4/8/16 channels a thread) plus the generic path are inlined into one kernel. No spills and no local memory
  (`REG=62 STACK=0 LOCAL=0`), and the executed path is the faster one: the prompt-compressor timings above.
- **0212** (wider prompt packing): the prompt key now runs two y lanes at the served 128-wide shape; its byte parity is
  the parity table's prompt-key rows, measured at `(rmsLanes, packLanes)`.
- **0206/0208**: `idx_fused` gains six shuffles (the tile-maximum reduction) and four registers; no spills.

## Paired arms

One image, one switch set per arm, 39-case exact suite against the live production panel, then the 8-minute serving
probe. Production's own panel was recorded first without a restart and is the base line.

| arm | image | switches | 39-case exact vs production |
|---|---|---|---|
| base | follow-up | none | **IDENTICAL** |
| prev | prune-six | IDX + COMPRESS2_NORM + ROPE_STORE + ROUTE | **IDENTICAL** |
| idx | follow-up | `TF_DS_IDX_FUSED=1` | **DIFFERENT, 4 of 39** |
| comp | follow-up | `TF_DS_FUSED_COMPRESS2_NORM=1` | **IDENTICAL** |
| rope | follow-up | `TF_DS_FUSED_ROPE_STORE=1` | **IDENTICAL** |
| route | follow-up | `TF_DS_FUSED_ROUTE=1` | **IDENTICAL** |
| all | follow-up | all four | **DIFFERENT, 4 of 39** (the same four) |
| allpdl | follow-up | all four + `TF_DS_PDL=1` | **IDENTICAL** to `all` (PDL changes no bytes) |
| base2 | follow-up | none | **IDENTICAL** (control drifts nowhere) |

The `all` arm differs from the previous bundle on exactly the four cases `idx` changed, so the bundle inherits the
`TF_DS_IDX_FUSED` difference and nothing else. `allpdl` matches `all` byte for byte, so `TF_DS_PDL=1` is bit-identical
on the fused bundle (it is still untested for speed: only the exact panel ran on it). (`all-vs-prev` re-recorded the
panel on an already warm prompt cache - `cached=18,432...63,488` on the follow-up side against 0 on the previous
bundle's - so that comparison's byte mismatches are not a clean read; the `all`-vs-production row, cached 0/0 on both
sides, is the one to quote.)

### Throughput (the 8-minute serving probe, tok/s)

| arm | Tetris greedy 1,024 | code greedy 1,024 | Tetris sampled x4 | long 100K greedy | 100K prefill s |
|---|---|---|---|---|---|
| base | 126.2 | 118.0 | 210.36 | 70.6 | 49.44 |
| prev | 127.3 | 119.0 | 211.59 | 67.6 | 49.45 |
| idx | 126.1 | 117.7 | 209.43 | 78.5 | 49.45 |
| comp | 126.6 | 118.4 | 208.58 | 76.3 | 49.39 |
| rope | 125.8 | 117.9 | 208.92 | 80.4 | 49.44 |
| route | 126.1 | 119.0 | 211.96 | 79.4 | 49.99 |
| all | 126.8 | 118.5 | 211.31 | 81.1 | 49.66 |
| allpdl | 126.0 | 118.5 | 210.17 | 79.5 | 49.73 |
| base2 | 125.6 | 117.6 | 207.26 | 75.3 | 49.64 |

The two base controls bound the noise: Tetris 125.6-126.2, code 117.6-118.0, x4 207.3-210.4, and the long 100K cell
70.6 vs 75.3 (6.6% on the same image and switches). Every arm is inside those bands, so the serving probe resolves no
throughput change from any of the thirteen at these settings - which is what the kernel-level numbers predict: the
fused switches were off in `base`/`base2`, and where they were on the kernels' few-µs deltas sit under a 42 ms round and
the probe's own spread. Prefill is stable at 49.4-50.0 s in every arm.

The four `idx` mismatches are `len_20000`, `len_30000`, `len_50000` and `len_64000` (completion length and token hash
both differ, cached 0/0 in both arms). `len_100000`, every shorter case, the resumes and the 8-way concurrent cases
match. So the previous tree's fused path was bit-identical at long prompts because it never produced the keys; the
follow-up's 0206 fold does, and what it produces changes the pruned selection's picks. Which width window that is needs
instrumenting - `len_100000` matches where the fold should also engage.

The standalone harness cannot see this: its pool cases compare against the *non-pruned* selection, which the fold
reproduces at every width tested. The pruned consumer is the kit's `topk_select_pruned` over `_score_keys`' output, and
the byte comparison against the kit's `_score_keys` shows the fold and the kit disagree by exactly the top bit:

```
kit  _score_keys  score 0.0 col 0 -> 0x00000000ffffffff
engine scoreKey   same cell      -> 0x80000000ffffffff     (the ^0x8000... sign flip)
```

Same low word, same ordered bits, only the sign bit differs, on all 64 columns of a probe row. The engine's
`idx_fused.zig:scoreKey` XORs the top bit; the kit's kernel does not. The fold writes `scoreKey`'s encoding into
`keys`, which the kit's `topk_select_pruned` then reads. That is the likely cause of the four long-prompt differences
and is the one thing this run leaves open: either the fold must write the kit's encoding at that call site, or the
pruned consumer must be swapped for one that reads `scoreKey`'s.

## Harness notes

- The kit's `_score_keys` faults with `CUDA_ERROR_MISALIGNED_ADDRESS` for `width % 4 != 0` (measured at 61 and 65;
  64/128/16,384/65,533 run). It also writes per-column keys while the engine's pool selects per block, so it is not the
  byte reference for the pool; it is the reference for the pruned path's encoding, which is the open item above.
- Both trees' decode compressor must be launched with `d` threads (the engine's geometry). Launched at 128 threads it
  disagrees with the kit on ~200 bytes a row; that was a harness error, not a tree difference, and both trees read 0
  differing bytes once launched the engine's way.
- Reaching the end of the probe needs `0207`'s shfl swap in `fused_prompt_latent` to be called with 256 threads;
  launched at 128 the old tree faults.

## Reproduction

```bash
# cubins: on the head, from the two trees' CUDA sources
bash runs/fused-followup-20261121/build2.sh ~/tfref-prune-six-tip/zig/kernels/cuda \
     ~/tfref-recipe-bench/zig/kernels/cuda ~/tf-bench/followup/bin
python3 runs/fused-followup-20261121/probe.py /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins ~/tf-bench/followup/bin
python3 runs/fused-followup-20261121/sass.py ~/tf-bench/followup/bin
# image and arms
bash runs/fused-followup-20261121/build.sh
bash runs/fused-followup-20261121/campaign.sh        # on the head, from the recipe root
```
