# fused_bench: parity + timing + SASS harness for the dsv41 fused kernels

Runs on the GB10 head (no nvcc on the workstation). Builds both CUDA trees with the engine's flags, then checks every
live output byte and times each fused kernel through CUDA graphs, with the launch floor and the copy roofline.

## One-time layout on the head

```
~/tf-bench/fused-next/tools/     this directory (rsynced from the recipe repo)
~/tf-bench/fused-next/src-old/   dsv41_ops.cu + dsv41_torch.cu of the old tree
~/tf-bench/fused-next/src-new/   dsv41_ops.cu + dsv41_torch.cu of the new tree
```

## Commands

```bash
# 1. build cubins for both trees + the launch-floor/bandwidth nodes (on the head)
bash tools/build.sh src-old src-new bin

# 2. parity + timings, all kernels (holds ~/tf-bench/gpu.lock; refuses below 3 GiB or while production serves)
python3 tools/bench.py bin bin /nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins --json wp0.json

# one kernel family / one shape (substring of the printed shape label)
python3 tools/bench.py bin bin <kit> --kernel idx --shape "idx R=7 nb=8 width"
python3 tools/bench.py bin bin <kit> --kernel prune --shape "prune R=48 dead"
python3 tools/bench.py bin bin <kit> --parity-only          # byte parity, no timing
python3 tools/bench.py bin bin <kit> --kernel score-keys     # the kit _score_keys launch cost

# 3. SASS evidence: opcode histogram by width, registers/spills/shared, occupancy
bash tools/sass.sh bin --json sass.json

# 4. copy bandwidth (memory roofline)
python3 tools/bw.py bin --json bw.json

# 5. ranked baseline table: time, floor, bytes, roofline, achieved bandwidth, bound, headroom
python3 tools/report.py wp0.json bw.json sass.json --md BASELINE-table.md

# 6. CPU unit tests for timing-label, idle-health and stream-sync guards (no GPU / libcuda required)
python3 -m unittest discover -s tools/fused_bench -p 'test_*.py' -v
```

The same pipeline is a recipe in the recipe repo: `just fused-bench` syncs the two CUDA source trees and this
directory to the head, builds, benches, sasses and pulls the artifacts into `runs/fused-next-20261121/wp0/`.
Overrides are environment variables: `OLD=... NEW=... KERNEL=idx SHAPE="idx R=7 nb=8 width" JSON=one.json just
fused-bench`. `PARITY=1 just fused-bench` is the durable check: byte parity of every kernel below plus the unfused
fallbacks, no timing and no idle-production wait, printing the per-case PASS/FAIL tally and exiting non-zero on any
difference. `OUT=<dir>` moves the pulled artifacts; `JSON=<name>` names the result file.

## Kernel and fallback coverage

| kernel (switch) | reference the harness compares against | rounding points the fused arm must keep | guarded shapes |
|---|---|---|---|
| 0254 `topk_indices_cut` (TF_DS_IDX_FUSED) | NumPy selects the highest integer keys, sorts the columns and applies visibility | signed zero, NaN payloads, infinities and ties to the lower column | k from 0 to 2048, partial rows, widths on both sides of 256 and 65536 |
| 0255 `cand_fast8` (TF_DS_IDX_FUSED) | NumPy computes the per-block mask and pins the newest block | any NaN excludes an unpinned block, all −∞ excludes it, and a pinned block always wins | eight scores, 16-byte-aligned start and stride, scalar fallbacks with a misaligned view, odd stride or another block size |
| 0195 `idx_fused`, `idx_pick` (TF_DS_IDX_FUSED) | the unfused `block_max` + `topk_indices` + `pool_pick` chain (the old tree's bytes) | the int64 key (`scoreKey`: -0 as +0, inverted column), the block maxima's NaN rule, ties to the lower block | nbk <= 16384, k <= 2048, k <= nbk, width/bsize agreement; small-row form at rows <= 7 with nbk >= 6144 |
| 0195 fallback (`idx_fused` off or past its bounds) | the old tree's same three launches | — | nbk > 16384 (131,080+ columns): `fallback idx` cases |
| 0197 `compress2x_norm{128,512}` (TF_DS_FUSED_COMPRESS2_NORM) | `compress2x` + the kit `_rmsnorm` | the pair softmax in fp32, the bf16 store, torch's mean order, the kit's four-lane 512 fold | d == 128 or 512 (decodeOk), any rows |
| 0197 `compress2_norm` prompt half (same switch) | `compress2` + `rms_norm` | torch's reduce order and mean factor, the raw-ring tail writes | pow2 d in [128, 2048] |
| 0197 fallback (no kit norm for the width) | the old tree's `compress2x` (the norm is the kit's in both) | — | d not 128/512: `fallback compress-decode` cases |
| 0198 `fused_idx_key`, `fused_rope_store`, `fused_rope_qd{,_pair}`, `fused_prompt_idx_key`, `fused_prompt_latent` (TF_DS_FUSED_ROPE_STORE) | the kit `_rmsnorm` + `_rope_heads` + `_fp4_store` / `_fp4qd_p2` | the kit's RoPE contraction (fma forms), E2M1 ties to even, the E8M0/E4M3 scale bytes, the 65,536-value bf16 sweep for the latent | d == 128 for the key, pow2 head dims in [128, 512], pow2 rope dims <= d, warp-aligned shapes; pair geometry at rows >= 32 |
| 0199 `prune_six` (TF_DS_FUSED_ROUTE) | `expert_prune` | the pick/weight/counter bytes, ties to the lower id | six slots, topk <= slots; else `expert_prune` runs (the reference arm) |

`just fused-bench` with the default `KERNEL=all` runs every row above.

`--kernel` accepts `all` or a comma list of `idx, idx-hist, idx-small, block-max, topk-indices, cand-mask, fold, compress-decode,
compress-prompt, prompt-source, key, latent, query, prompt-key, prompt-latent, prune, score-keys`. `--reps` (default 20) graphs and `--batch` (default 100) replays per graph.

## What each case checks

- **0195 idx**: fused mask / block list vs the unfused `block_max`+`topk_indices`+`pool_pick` chain, at decode
  R=1/7/32/48 and prompt R=128/1024/2048, nb=8/2049/4096/8192 (the 16K-65K column window), random/tied/flat and
  NaN/+-inf/signed-zero rows.
- **0197 decode compressor**: fused `compress2x_norm` vs `compress2x`+kit `_rmsnorm`, D=128/512, R=1/7/32/48, mixed,
  unaligned, cached, cross, invalid and stress layouts (rings, stride padding and the invalid flag included).
- **0197 prompt compressor**: fused `compress2_norm` vs `compress2`+`rms_norm`, D=512/2048, R=128/1024.
- **0197 prompt source step** (`--kernel prompt-source`): the whole ratio-2 KV-source step — index upload, two
  `scatter_rows` launches, `compress2` and `rms_norm` for the old trees against this tree's single folded launch —
  with both raw-ring buffers and the latent compared byte for byte, at R=1/7/48/128/1024 D=512, R=128 D=2048, the
  odd-trailing-row shapes, a one-row chunk with no pair, and NaN/±inf/±0 inputs.
- **0198 five kernels**: fused key / latent / query / prompt key / prompt latent vs the kit's `_rmsnorm`,
  `_rope_heads`, `_fp4_store`, `_fp4qd_p2`; codes/scales (or dequantized query) only, because the rotated intermediates
  have no reader; plus every one of the 65,536 bf16 encodings through the latent store.
- **0199 prune**: fused `prune_six` vs `expert_prune` (pick, weights, all six counters), R=1/7/32/48, dead mask,
  ties/NaN/inf/negative shares.
- **kit `_score_keys`**: the launch 0195's key fold would replace, priced at 16K-65K columns (width % 4 == 0 only).

Parity failures are listed and the process exits non-zero. Device `Buffer` uploads and host readbacks synchronize on the
non-blocking benchmark stream (`cuMemcpyHtoDAsync_v2` + `cuStreamSynchronize` before launch, and `cuStreamSynchronize`
before `cuMemcpyDtoH_v2`), avoiding legacy stream-0 DMA races against `CU_STREAM_NON_BLOCKING`. Timings are median and IQR of `--reps` graphs of `--batch`
replays, old and new interleaved, with an empty-kernel graph at the fused launch shape as the floor. On guarded runs,
each timing block requires `/health` `live.connections == 0` and valid unchanged `(prompt_tokens_total,
generation_tokens_total)` before and after the block; each timed row records `timing_valid`, `timing_retries` and
`timing_guarded` (emitting `timing: {}` if all `--tries` fail), and `health.timing_clean` is `true` only when guarded,
overall before/after counters are valid, and every timed row has `timing_valid == true`.
