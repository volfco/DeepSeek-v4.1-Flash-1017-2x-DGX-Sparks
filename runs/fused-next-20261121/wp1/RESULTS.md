# WP1: the 0195 indexer (fused pool, idx_fused.zig, candidate-source arms)

Run 2026-11-21 on the GB10 head (`sm_121`), engine tree `~/Workspace/tfref-fused-next` branch `fused-next` at
`3c54a9d`, CUDA 13.0, flags `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`. Harness: `tools/fused_bench/` (WP0),
extended per item. Every timing block held `~/tf-bench/gpu.lock`, waited for `connections == 0` and verified the served
counters did not move (`timing_clean`, retries 0). Timings are CUDA-graph medians of 20 repetitions of 100 replays (µs
per launch), old and new interleaved in one process, with the empty-kernel floor at the same launch shape.

Item numbering below is WP1's; commits are numbered from 0213.

## Item 1 — the key fold (old 0206) and the high-word cache (old 0208): CLOSED, NO CHANGE

**What it was.** 0206 made the fused pool launch write `_score_keys`' outputs (every column's key and each 64-key
tile's maximum) so the decode arm could skip that kit launch. The parked patch wrote `idx_fused.zig:scoreKey`'s key,
which flips the top bit; the gate failed 4 of 39 replies. 0208 cached 0206's key high words in shared memory.

**The kit's bytes (the reference).** Read from the kit cubin `df85d0b3…` (`_score_keys`, `HAS_TMAX=1`), both by
`cuobjdump -sass` and by running it on probe rows (`/tmp/probe_keys.py`, kept in `probe_keys.py` here):

- Its transform is the engine's `scoreKey` transform without the final flip: `bits == 0x80000000 -> 0`,
  `ordered = sb < 0 ? sb ^ 0x7fffffff : sb`, `key = (sign-extended ordered << 32) | (0xFFFFFFFF - col)`. The SASS shows
  the `LOP3` -0 test, the `^0x7fffffff` under `ISETP.GE`, and `~col` as `LOP3` with `~k` masks.
- So `_score_keys`' key is `scoreKey(x, j) ^ 0x8000000000000000` (sign bit clear). `topk_select_pruned` compares them
  as torch int64 does (signed); `scoreKey`'s flip is what makes the engine's own radix select unsigned.
- Its tile maximum is the **signed** maximum of those keys (`IMNMX.S64` in the SASS), out-of-range columns padded to
  `INT64_MIN`; i.e. the unsigned maximum of `scoreKey` over the tile, flipped back.
- Probe bytes pinned in the engine host test: `(0.0,0)=00000000ffffffff`, `(-0.0,1)=00000000fffffffe`,
  `(1.0,2)=3f800000fffffffd`, `(-1.0,3)=c07ffffffffffffc`, canonical NaN at col 8 in tile 0 = `7fc00000fffffff7`.

**The fix.** `keys[i] = scoreKey(x,i) ^ 0x8000000000000000`; `tmax[t] = (unsigned max of scoreKey over the tile) ^
0x8000000000000000`. Host `idx_fused.kitKey`/`foldKeys` and three host tests (the bytes above, NaN ranking, the window)
were added. The fold's window is the fused pool's regime: the call site needs `nb > candidate_blocks * 8 = 16,384`
columns and `fuseOk` needs `nbk = ceil(width/8) <= 8192`, i.e. width in (16,384, 65,536]. The served widths there are
powers of two (32,768 and 65,536 at ratio 2). `len_100000` matched under the broken encoding because at 131,072 bucket /
2 = 65,536 columns the fold *did* engage, but the old encoding's signed/unsigned order only differs when the candidate
pool mixes score signs, and that round's pool did not — not evidence that the fold was right.

**Parity (kit bytes as the reference), all PASS, 0 differing bytes.** `fused_bench --kernel fold` (new case
`case_idx_fold`, which runs the kit `_score_keys` as the reference and the unfused `block_max`+`topk_indices`+
`pool_pick` chain for the mask/list):

| rows | widths | distributions | result |
|---|---|---|---|
| 1, 7, 32, 48 | 16,388; 32,768; 65,536 | mixed-sign random | keys, tmax, cblk all identical to the kit's bytes and the unfused chain |
| 7 | 16,388; 32,768 | random, ties, flat (-inf) | identical |
| 32 | 65,536 | random, ties, flat | identical |
| 1, 7, 32, 48 | 16,388; 65,532; 65,536 | NaN, ±inf, ±0, partial last tile (60 keys) | identical |

**Timings — the fold does not pay.** Arms: `old` = fused pool + kit `_score_keys` (today's path), `new` = fused pool
with the fold, `newnf` = the same new kernel with the fold off, `kit` = the kit launch alone. The first fold form wrote
its keys with the natural per-lane 8-byte stores; a second form wrote them coalesced (4 × 16-byte stores) and was
2× faster than the first but still lost:

| shape | old | new (8B stores) | new (16B stores) | newnf | kit alone | floor |
|---|---:|---:|---:|---:|---:|---:|
| R=1 w=32,768 | 10.45 | 27.44 | 20.79 | 10.72 | 1.13 | 0.48 |
| R=7 w=32,768 | 12.23 | 49.52 | 30.99 | 12.54 | 2.88 | 0.50 |
| R=32 w=32,768 | 28.69 | 63.34 | 42.53 | 31.58 | 10.25 | 0.56 |
| R=48 w=32,768 | 71.24 | 103.65 | 91.28 | 68.93 | 86.83 | 0.56 |
| R=1 w=65,536 | 17.17 | 53.59 | 41.74 | 17.42 | 1.48 | 0.48 |
| R=7 w=65,536 | 20.92 | 101.00 | 61.03 | 21.30 | 4.92 | 0.50 |
| R=32 w=65,536 | 108.85 | 146.14 | 111.09 | 110.70 | 91.60 | 0.56 |
| R=48 w=65,536 | 212.48 | 259.18 | 208.53 | 215.03 | 193.72 | 0.57 |

Only R=48 width 65,536 is faster with the fold (208.5 vs 212.5, 1.9%, inside the IQR). Every other served decode shape
is slower, so under the keep/drop rule it is a drop.

**Why.** The fused pool is one CTA per row: 7 CTAs at R=7, where the kit launches 224. A store microbenchmark on this
GPU (`storepat.cu`, same launch shape, writing one row's 32,768 keys per CTA) measures the fold's per-lane 8-byte
64-byte-strided store pattern at **41.7 µs** per CTA-row against **7.5 µs** for a coalesced 8-byte store and 21.2 µs
for the 16-byte form; the 16-byte fold also spilled (`STACK=64`). The fold cannot out-write a 224-CTA streaming kit
launch from 7-48 CTAs. Its saving is one launch per round (the candidate-source layer only), ~2.9 µs at R=7 and
~194 µs at R=48 width 65,536 — and it gives back more than that at every shape but one.

**SASS/resource diff (the fold's kernel, 16-byte form, vs the current kernel).** `cuobjdump -res-usage`/`sass.py`:

| kernel | insn | LDG | LDG.64 | STG | STG.128 | STG.64 | SHFL | BAR | ATOM | REG | STACK | LOCAL | shared | block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| idx_fused (current) | 632 | 3 | 2 | 2 | 0 | 0 | 20 | 8 | 1 | 38 | 0 | 0 | 2208 | 1024 |
| idx_fused (fold, 16B) | 760 | 4 | 2 | 8 | 4 | 2 | 26 | 8 | 1 | 43 | 64 | 0 | 2208 | 1024 |

Bound: the current pool launch at R≤48 is issue/latency-bound (WP0: 9.1-18.4 µs of headroom over the cache roofline);
the fold turns it into a store-throughput problem at 1 CTA/row. **0208** (the high-word cache) was only a way to avoid
re-deriving `scoreKey` in the fold, so it is not re-landed either.

**Reproduce.** `bash tools/build.sh src-old src-new bin` with `src-old` = 3c54a9d and `src-new` = 3c54a9d + the 0206
patch (encoding fixed, 16-byte stores); `python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins
--kernel fold` (parity and timings), `--parity-only` for the byte comparison. Artifacts: `wp1-fold.json`,
`sass-fold.json`, `probe_keys.py`, `storepat.cu`.

## Item 2 — compile-time served block size + vectorized aligned loads: COMMIT 0213

**Change.** `tf_ds_idx_fused_kernel`'s phase 1 is now a template, `tf_ds_block_max_t<BS>`: `BS == 8` (the served
candidate block size) with a 16-byte-aligned row stride (`ss % 4 == 0`) does two `float4` loads a block, unpacks them
into eight registers and runs the same comparison order as before (a NaN wins; otherwise a greater score replaces the
maximum), fully unrolled, with the row's last partial block on a guarded scalar tail. Every other `bsize` and every
unaligned stride keeps the runtime loop, which is the same arithmetic. The kernel body itself is single-instance (an
earlier whole-body template grew the code and cost the small shapes 0.2 µs).

**Parity: exact everywhere.** `--kernel idx` byte parity (new vs old vs the unfused `block_max`+`topk_indices`+
`pool_pick` chain), 0 differing bytes on every case: R=1/7/32/48 nb=8; R=7 nb=2049; R=7/32 nb=4096/8192, mask and
list, ties/random/flat; R=128/1024/2048 nb=4096 mask ties; R=128 nb=4096 mask random; NaN/±inf/±0 special; and the
two unaligned-stride cases (pad=1, pad=3 → the runtime fallback), all PASS.

**Timings (µs, median of 20 × 100 graph replays; floor = empty kernel at the same launch shape).**

| shape | old | new | Δ | floor |
|---|---:|---:|---:|---:|
| R=7 nb=2049 list random | 6.42 | 5.56 | −13.4% | 0.50 |
| R=7 nb=4096 list random | 10.12 | 8.42 | −16.8% | 0.50 |
| R=32 nb=8192 list ties | 20.51 | 17.15 | −16.4% | 0.50 |
| R=7 nb=8192 mask flat | 19.24 | 15.37 | −20.1% | 0.50 |
| R=128 nb=4096 mask random | 28.41 | 25.87 | −8.9% | 0.90 |
| R=128 nb=4096 mask ties | 36.46 | 34.23 | −6.1% | 0.83 |
| R=1024 nb=4096 mask ties | 580.57 | 583.43 | +0.5% | 3.85 |
| R=2048 nb=4096 mask ties | 1127.81 | 1132.62 | +0.4% | 7.22 |

The two large-row shapes are at the DRAM roofline (WP0: 240-247 GB/s, above the 224 GB/s copy rate); their +0.4-0.5% is
smaller than the run-to-run spread of the old arm itself (580.6 vs 582.9 µs across two runs of this same case), so it
is not a resolvable regression. The nb=8 shapes (width 61) are 0.1-0.2 µs slower (2.01→2.22 at R=7); both call sites
only reach this kernel at `nbk > candidate_blocks = 2048` (`round.zig`'s `nb > candidate_blocks * cb`, `prompt.zig`'s
`pooled = nbk > candidate_blocks`), so nb=8 is a harness edge case, not a served shape.

**SASS/resources** (`cuobjdump -sass`/`-res-usage`; old = 3c54a9d, new = 0213):

| kernel | insn | LDG.E.128 | LDG.E.64 | LDG.E | FSETP | FSEL | PRMT | REG | STACK | LOCAL | SHARED | block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| old | 632 | 0 | 2 | 1 | 3 | 3 | 1 | 38 | 0 | 0 | 2208 | 1024 |
| new | 760 | 2 | 2 | 2 | 21 | 6 | 18 | 38 | 0 | 0 | 2208 | 1024 |

The served path is the two `LDG.E.128` (`desc[UR8][R20.64]` and `+0x10`) followed by 16 predicate instructions (eight
NaN tests, eight greater-than selects) with no loop back-edge: the body is fully unrolled. No spills, registers and
occupancy unchanged (1 block/SM, 66.7%). Bound: still issue/latency at 8.4 µs against a 0.50 µs floor and a
1.02 µs cache roofline for 0.97 MB (WP0's byte basis) — the loads are no longer the limiter at the decode shapes;
what is left is the radix selector and the per-row single-CTA layout (items 3-5).

**Reproduce.** `bash tools/build.sh src-old src-new bin` (src-old = 3c54a9d, src-new = 0213);
`python3 tools/bench.py bin bin <kit> --kernel idx --json wp1-idx2b.json`; the unaligned cases are in the same run
(`pad=1`, `pad=3`). Artifacts: `wp1-idx2.json`, `wp1-idx2b.json`, `sass2.json`.

## Item 3 — warp-aggregated histogram: CLOSED, NO CHANGE

**Change tried.** `tf_ds_pool_kth`'s per-element shared `atomicAdd` replaced by `__match_any_sync` grouping plus one
leader `atomicAdd` of each group's count (the same integer histogram: the group counts sum to the same per-digit
totals), with every lane kept in the loop (sentinel digit 256 for lanes past the row or outside the prefix). The
kernel also gained per-tile zeroing of `warp_end[warps..31]` so a CTA below 1024 threads is correct (the old kernel
leaves those offsets stale), which is what makes the 256/512-thread sweep runnable.

**Parity.** All 12 sweep cases PASS: R=7 nb=4096 × {random, ties, flat} × {256, 512, 1024} and R=32 nb=8192 ties ×
{256, 512, 1024}, mask/list cblk identical to the unfused chain.

**Timings (µs, median of 20 × 100 graph replays; the same new kernel at each block size).**

| shape | 256 thr | 512 thr | 1024 thr | floor (1024) | 0213, no aggregation |
|---|---:|---:|---:|---:|---:|
| R=7 nb=4096 random | 21.31 | 13.96 | 11.33 | 0.50 | **8.42** |
| R=7 nb=4096 ties | 27.55 | 17.05 | 13.07 | 0.50 | — (0213 ties not run; R=7 nb=8192 ties 17.15) |
| R=7 nb=4096 flat | 26.63 | 16.49 | 12.54 | 0.50 | — |
| R=32 nb=8192 ties | 50.91 | 29.52 | 21.51 | 0.50 | **17.15** |

**Verdict: drop.** At the served 1024-thread geometry the aggregated histogram is slower than 0213 on the served
(random-score) distribution by 2.9 µs (11.33 vs 8.42) and on ties by 4.4 µs (21.51 vs 17.15): the shared atomics are
not the bottleneck, so `MATCH` + `POPC` + leader election costs more than the atomics it saves. The sweep also shows
the current 1024-thread CTA is the fastest of the three; smaller CTAs lose 25-45% because the radix scan then needs
more grid-stride passes. The SASS of the aggregated form had one `MATCH`, one `ATOMS`, REG 39, no spills.

**Reproduce.** `python3 tools/bench.py bin bin <kit> --kernel idx-hist --json wp1-hist.json` (old = 3c54a9d,
new = 0213 + the aggregation). Artifacts: `wp1-hist.json` (the aggregated kernel's SASS was measured inline: one `MATCH`, one `ATOMS`, REG 39, no spills).

## Item 4 — mask-only prompt arm skips compaction, cutoff ties resolve during compaction: COMMIT 0214

**Change.** `tf_ds_idx_select` (one copy of the selection, shared by the fused kernel and item 5's second launch) now
writes the mask arm's bits straight from the cutoff when `cblk == nullptr` (the prompt call site): no ballot, no
cross-warp prefix sum, no compaction barriers — only when the cutoff bin straddles the slots left (the tied columns)
does it need one prefix to keep their ascending order. The radix scan stops after the score's four bytes
(`tf_ds_pool_cut`); the column bytes are no longer scanned, and when the cutoff bin holds more columns than the slots
left (`want > 0`) the compaction takes the bin's lowest `want` columns, which is the same total order the column-digit
passes produced. When `want == 0` (the cutoff bin completes exactly) both arms take the old one-comparison path, so
untied rows cost exactly what they did before.

**Parity: exact everywhere.** `--kernel idx` byte parity (new vs old vs the unfused chain), 0 differing bytes on every
case: R=1/7/32/48 nb=8; R=7 nb=2049; R=7/32/4096/8192, mask and list, ties/random/flat/all-ones (new "same" case),
R=128/1024/2048 nb=4096 ties and random, NaN/±inf/±0 special, and the unaligned-stride fallbacks. That includes the
cases where the cutoff straddles ties (the "ties", "same" and "flat" rows) for both the mask and the list arm.

**Timings (µs, median of 20 × 100 graph replays; floor 0.50-0.90).** `old` = 3c54a9d, `item 2` = 0213, `new` = 0214:

| shape | old | item 2 | new | Δ vs old | Δ vs item 2 |
|---|---:|---:|---:|---:|---:|
| R=7 nb=8192 mask flat | 19.24 | 15.37 | 13.42 | −30.2% | −12.7% |
| R=128 nb=4096 mask ties | 36.56 | 34.23 | 29.79 | −18.5% | −13.0% |
| R=128 nb=4096 mask random | 28.33 | 25.87 | 23.25 | −17.9% | −10.1% |
| R=32 nb=8192 list ties | 20.51 | 17.15 | 17.11 | −16.6% | −0.2% |
| R=7 nb=4096 list random | 10.12 | 8.42 | 8.46 | −16.4% | +0.5% |
| R=7 nb=2049 list random | 6.44 | 5.56 | 5.58 | −13.4% | +0.4% |
| R=1024 nb=4096 mask ties | 593.7 | 583.4 | 590.2 | −0.6% | (noise: the old arm moved 12 µs between runs) |
| R=2048 nb=4096 mask ties | 1147.8 | 1132.6 | 1148.3 | 0.0% | (same) |

The decode (list) arm is at parity with 0213 within 0.5% (the `want == 0` fast path is the old comparison); the mask
arm gains 10-13% from dropping the compaction. The nb=8 shapes are 0.1-0.3 µs slower (2.03 → 2.30); they are the
harness's edge cases (both call sites only reach this kernel at nbk > 2048).

**SASS/resources.** 1016 instructions (0213: 760; 3c54a9d: 632), SHFL 40 (20), BAR 15 (8), STG 3, REG 38, STACK 0,
LOCAL 0, SHARED 2208, 1 block/SM (66.7%). The extra SHFL/BAR are the `want > 0` tie branch; the executed `want == 0`
path keeps 0213's instruction sequence.

**Bound.** The prompt mask arm is now at 78% of the cache roofline (R=128 nb=4096: 23.25 µs against 18.1 µs for
17.3 MB at 954 GB/s) and the decode shapes remain issue/latency-bound (8.46 µs against a 0.50 µs floor and a 1.0 µs
roofline for 0.92 MB).

**Reproduce.** `bash tools/build.sh src-old src-new bin` (src-old = 3c54a9d, src-new = 0214); `python3
tools/bench.py bin bin <kit> --kernel idx --json wp1-idx4b.json`. Artifacts: `wp1-idx4.json` (the earlier form),
`wp1-idx4b.json`, `sass4.json`.

## Item 5 — small-row parallelism: COMMIT 0215 (only where it wins; a shape dispatch)

**Change.** `tf_ds_idx_select` is also called by a new `tf_ds_idx_pick_kernel` (one CTA a row) that reads block_max's
global maxima. For `rows <= 7` and `nbk >= 6144` the decode arm now launches `block_max` (the pool's own 2D grid: 16
CTAs a row at nbk 4096, 32 at 8192) and then `idx_pick`, instead of the one-CTA-a-row fused launch. The selection,
the tied-cutoff rule and the ascending compaction are the same function, so the block list is byte-identical; the
merge order is the same total order on (score bits, inverted column) by construction.

**Parity.** PASS at every tested shape: R=1/7 × nb=2049/4096/8192 random and R=7 nb=4096 ties, new vs the unfused
chain and vs the one-launch form, 0 differing bytes.

**Timings (µs).** `fused` = 0214's one-launch form; `two-phase` = block_max + idx_pick; floor 0.48-0.50.

| shape | fused | two-phase | Δ | dispatched? |
|---|---:|---:|---:|---|
| R=1 nb=2049 | 5.33 | 5.49 | +3.1% | no |
| R=1 nb=4096 | 7.70 | 7.32 | −5.0% | no |
| R=1 nb=8192 | 12.14 | 10.53 | −13.2% | yes |
| R=7 nb=2049 | 5.58 | 5.76 | +3.2% | no |
| R=7 nb=4096 | 8.46 | 8.67 | +2.4% | no |
| R=7 nb=8192 | 12.44 | 11.80 | −5.1% | yes |
| R=7 nb=4096 ties | 10.22 | 10.35 | +1.2% | no |

Against 3c54a9d (6.05/9.26/15.88 at R=1, 6.38/10.08/16.10 at R=7) the dispatched shape is 9.6-33% faster; the shapes
below the threshold keep the single launch so none regresses. The threshold is the 8192-block pool: below it the
second launch plus the staging costs more than the parallel read saves. `block_max` alone measures 1.42/2.05/2.44 µs
at R=1 nb=4096, R=7 nb=4096/8192, so the parallel read is 2.4-2.9 µs of the two-phase's total.

**SASS/resources.** `tf_ds_idx_pick_kernel`: REG 38, STACK 0, LOCAL 0, SHARED 2208, 1024 threads, 1 block/SM. The
fused kernel keeps REG 40 (the shared helper's inlining).

**Bound.** At R=7 nb=8192 the two-phase is 11.8 µs against a 0.50 µs floor and a 2.0 µs roofline (1.9 MB): still
issue/latency-bound, but the phase-1 read is no longer the limiter.

**Reproduce.** `python3 tools/bench.py bin bin <kit> --kernel idx-small --json wp1-small2.json` (arms: unfused,
3c54a9d fused, 0214 fused, two-phase). Artifacts: `wp1-small2.json`, `sass5.json`.

## Item 6 — prompt bypass when every compressed key fits: COMMIT 0216; retained-block scoring: CLOSED, NO CHANGE

**Landed (bypass).** `prompt.indexer` now returns early when `0 < n_comp_end <= index_topk` (512): it writes the
visible ids with `iotaVis` and sets `sh.kk = n_comp_end`, skipping the query projection (cuBLAS), its RoPE and FP4
quantization (one 0198 launch), the heads' weights (cuBLAS + bf16 scale), the scoring (`_index_score`) and the
selection (`topk_indices`/`topk_select`) — six launches per indexer layer per short chunk. The picks are the same set
in the same order: the engine's own `tf_ds_topk_indices_kernel` sorts its k columns ascending (bitonic sort at
`dsv41_ops.cu:190-211`) and masks columns at or past `vis` to -1, which is exactly `iotaVis`'s output for
`kk = n_comp_end`; the keyed arm's `topk_select` over all `n_comp_end` keys returns the same set, and the decode arm
already takes this identity for `nb <= index_topk`. `pooled` is false at these widths (`nbk > candidate_blocks` needs
`n_comp_end > 16,384`), so `sh.pool` is left exactly as the full path left it. No kernel changed, so the byte-parity
harness cannot see it; the engine's host tests pass (794/796, the two pre-existing failures) and the supervisor's
39-case exact suite is the arbiter. Cost removed, measured floor basis: six launches at WP0's 0.50-0.90 µs empty-kernel
floor is ≥3 µs per indexer layer per short chunk, plus the scoring work (n_comp_end ≤ 512 columns a row). This is the
short-prompt path (a 127-token cold prompt compresses to ≤ 64 columns), so it applies to every short chunk.

**Closed (retained-block scoring).** The later pooled layers (after the candidate source) still score the full
`n_comp_end` and then `applyPool`. The todo's "score directly from retained block IDs and map picks back to original
columns" needs a kernel that scores a compacted block list: the kit's `_index_score` has 16 AOT variants and every
`HAS_CAND=1` one writes int64 keys (`KEYS=1`), so the scores arm has no candidate-masked variant; the compacted-keys
path (`_index_keys_cand`, `HAS_CAND`) returns picks in compacted order, which then need a mapping kernel back to the
original columns. The engine's `topkSelect` also needs a power-of-two key count, which a prompt chunk's
`n_comp_end` need not be. So this half is a new kernel plus a mapping step, not a launch removal; it is left as a
follow-up with the numbers to size it: at 65,536 columns and 128 rows the scores are 33.5 MB and each row needs at
most 16,384 scored columns (a 4x reduction in the dots), against 33.5 MB of int64 keys if the keys path were used.

## Item 7 — 100K+ contexts: COMMIT 0217

**Change.** `idx_fused.max_nbk` 8192 → 16384 (131,072 columns, a 262,144-token bucket at ratio 2), and `ops.load`
opts both pool kernels into dynamic shared memory at `max_nbk * 4 = 65,536` bytes. The bound is the GPU's opt-in
per-block shared limit, queried here: `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101,376` bytes (25,344 fp32 maxima), so
16,384 blocks (64 KiB) is the largest served power of two. The radix shortcut (the columns' two constant high bytes)
would still hold to 65,536 blocks; at 65,537 a column's inverted id leaves 0xFFFE there, which is why the cap is not
pushed further; past `max_nbk` the call sites keep the unfused sequence.

**Parity: exact.** PASS at nb=8193 (the old cap's first unfused block, 65,541 columns), nb=16384 (131,069 columns) and
nb=16385 (131,077 columns, past the new cap), R=7 list/random and R=32 mask/ties, new vs old vs the unfused chain, 0
differing bytes. The opt-in launch is accepted by the driver at 64 KiB (the harness sets the same attribute).

**Timings (µs, median of 20 × 100 graph replays; floor 0.50).** `ref` = the unfused three-launch chain (what the
engine runs past the cap), `old` = 3c54a9d's kernel body at the same width, `new` = 0217:

| shape | ref | old | new | new vs ref |
|---|---:|---:|---:|---:|
| R=7 nb=8193 (65,541 cols) | 46.54 | 17.31 | 13.83 | −70% |
| R=32 nb=8193 | 63.15 | 21.26 | 15.69 | −75% |
| R=7 nb=16384 (131,069 cols) | 59.14 | 69.60 | 23.33 | −61% |
| R=32 nb=16384 | 76.57 | 85.20 | 25.50 | −67% |
| R=7 nb=16385 (past the cap) | 60.86 | 69.00 | 22.58 | (engine: unfused, 60.86) |

So 100K-260K-token buckets now take the fused launch instead of the three-launch chain, 2.5-3.4x faster; the raise is
what lets 0213's vectorized phase 1 apply at those widths (the 3c54a9d body at 131,069 columns is 69.6 µs, slower than
the unfused chain). Bound: at nb=16384 the fused launch moves 3.7 MB (R=7) in 23.3 µs (160 GB/s) against a 3.9 µs
cache roofline — issue/latency-bound at seven rows; the 64 KiB shared allocation still allows 1 block/SM (1536
threads/SM, 1024-thread block), unchanged occupancy.

**Reproduce.** `python3 tools/bench.py bin bin <kit> --kernel idx --shape "nb=1638"` / `--shape "nb=8193"` (parity and
timings), `bash tools/build.sh src-old src-new bin` (src-old = 3c54a9d, src-new = 0217).

## The kernel's final position

At the served decode shapes the fused pool is still issue/latency-bound, not bandwidth-bound: R=7 nb=4096 is 8.46 µs
against a 0.50 µs launch floor and a 0.96 µs cache roofline for 0.92 MB (115 GB/s), and R=1/7 with the 8192-block pool
take the two-phase form (10.5/11.8 µs). At the prompt shapes it is now close to the cache roofline: R=128 nb=4096 mask
is 23.25 µs against 18.1 µs for 17.3 MB at the measured 954 GB/s (78% of the roofline; it was 36.6 µs, 49%), and the
R=1024/2048 shapes sit at the DRAM roofline (590/1148 µs for 138/277 MB, 234/241 GB/s, above the 224 GB/s copy rate).
What is left is the single-CTA-a-row selection at the decode row counts and the radix scan's serialised digit passes;
items 3 (aggregation), 5 (below the 8192-block pool) and 6b (retained-block scoring) measured or reasoned as not
paying.

## Summary

| item | verdict | commit |
|---|---|---|
| 1 key fold (0206) + high-word cache (0208) | closed, no change: encoding fixed and proven against the kit's bytes, but the fold is slower at every served shape (only R=48 w=65,536 is 1.9% faster, inside the IQR) | — |
| 2 bsize==8 template + vectorized loads | landed: −6% to −20% at the decode/prompt work-bound shapes, byte-identical | 0213 |
| 3 warp-aggregated histogram | closed, no change: 2.9-4.4 µs slower at 1024 threads on random and ties; 256/512-thread CTAs are 25-45% worse | — |
| 4 mask-only compaction skip + cutoff ties in compaction | landed: mask arm −10% to −13% (vs 0213), decode arm at parity, byte-identical | 0214 |
| 5 small-row form | landed only at the 8192-block pool (R=1 −13%, R=7 −5% vs the one-launch form; 9.6-33% vs 3c54a9d), dispatched by `small_rows`/`small_nbk` | 0215 |
| 6 prompt bypass / retained-block scoring | bypass landed (six launches a layer a short chunk removed); retained-block scoring closed, no change (no scores+CAND kit variant; compacted picks need a mapping kernel) | 0216 |
| 7 100K+ contexts | landed: 16,384 blocks / 131,072 columns through the opt-in shared limit, 2.5-3.4x faster than the unfused chain at 100K-260K tokens, byte-identical | 0217 |

Open: item 6b (retained-block scoring) as a follow-up; the 39-case exact suite (the supervisor's gate) decides items
6's bypass, whose pick order is identical in the engine's own ascending sort but cannot be byte-checked by this
harness; `ncu` counters were not needed (no decision depended on them).
