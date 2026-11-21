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
```

The same pipeline is a recipe in the recipe repo: `just fused-bench` syncs the two CUDA source trees and this
directory to the head, builds, benches, sasses and pulls the artifacts into `runs/fused-next-20261121/wp0/`.
Overrides are environment variables: `OLD=... NEW=... KERNEL=idx SHAPE="idx R=7 nb=8 width" JSON=one.json just
fused-bench`.

`--kernel` accepts `all` or a comma list of `idx, idx-hist, idx-small, block-max, fold, compress-decode,
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

Parity failures are listed and the process exits non-zero. Timings are median and IQR of `--reps` graphs of `--batch`
replays, old and new interleaved, with an empty-kernel graph at the fused launch shape as the floor.
