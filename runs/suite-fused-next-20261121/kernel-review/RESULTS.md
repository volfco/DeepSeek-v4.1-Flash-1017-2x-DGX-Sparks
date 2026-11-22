# GB10 indexer kernel review

The November 21 serving suite compared production with engine revision `d5ecba0`. Every comparable reply matched. In the two eight-stream comparisons, aggregate tokens/s fell 0.59% and 0.69%, while ms/round moved +0.06% and −0.43%. Those measurements do not identify a slower kernel. The salted long-context cells used different inputs across arms. [Serving measurements](../RESULTS.md)

## Repairs

- **0254, score-to-column top-k** (`b4f6d18`). The non-keyed prompt indexer calls `Ops.topkIndices` on its score rows. Deep decode also calls it on the block maxima when its pool exceeds the fused limit. The original kernel read the row for all eight radix digits and had one thread search up to 256 histogram bins per digit. The new kernel skips constant high bytes of the inverted column, uses one warp to locate the cutoff bin, and stops when the entire bin fits. At that point the partial prefix is an inclusive lower bound admitting exactly k keys. If the bin straddles k, selection continues through the varying column bytes to preserve ties. Selecting every column writes its visible range directly. [Recipe patch](../../../zig/patches/0254-dsv41-topk-cutoff.patch)
- **0255, decode candidate mask** (`d0c9fb1`). Below the pool threshold, the candidate-source layer calls `Ops.candFast`. Each thread now reads its eight FP32 scores with two aligned 16-byte loads. Any NaN excludes the block, all −∞ excludes it, and the newest visible block is pinned. The launcher checks the block size, score address and stride before choosing this kernel. A misaligned view, odd stride or different block size selects the original scalar kernel. [Recipe patch](../../../zig/patches/0255-dsv41-candidate-mask-loads.patch)

Both changes use the existing `TF_DS_IDX_FUSED=1` switch. The standard recipe still gives its separate `TF_DS_TOPK_FAST` selector precedence when enabled. The two engine commits are on the local `fused-next` branch. The recipe exports apply after its existing 134 patches on `fbbc971`.

## Measurements

CUDA-graph medians on one GB10. Each arm ran 20 interleaved measurements of 100 launches or chains. Production’s token counters held steady throughout the timing run. The JSON includes each IQR, the empty-node floor, and the kernel symbol selected on each direct test. [Timing data](index-timing.json)

| Operation | Rows × width | Old µs | New µs | Time saved |
|---|---|---:|---:|---:|
| Top-k, random scores | 128 × 4,096 scores, k=512 | 77.955 | 30.861 | 60.4% |
| Top-k, random scores | 128 × 32,768 scores, k=512 | 237.022 | 100.717 | 57.5% |
| Top-k, all −∞ | 128 × 32,768 scores, k=512 | 227.199 | 138.788 | 38.9% |
| Top-k, mixed special values | 7 × 65,537 scores, k=513 | 142.804 | 119.970 | 16.0% |
| Deep fallback chain | 7 × 16,385 blocks, k=2,048 | 61.264 | 37.620 | 38.6% |
| Deep fallback chain | 32 × 16,385 blocks, k=2,048 | 75.160 | 48.488 | 35.5% |
| Candidate mask | 7 × 2,048 blocks | 1.689 | 1.318 | 22.0% |
| Candidate mask | 48 × 2,048 blocks | 3.369 | 2.849 | 15.4% |

At all 12 aligned decode mask shapes the median time fell by 15–23%. All three fallback layouts selected the scalar symbol. In the optimized mask the disassembly shows two 128-bit global load instructions and 22 registers per thread, down from 31. Top-k uses 36 registers, up from 32, with the same 18,464 bytes of static shared memory. Both optimized kernels report zero stack and local bytes. [GB10 disassembly counts and resource use](sass.json)

## Correctness and scope

The standalone suite passed **194/194 cases** in two runs. The 32 new cases compare output bytes with a NumPy reference, including switch-off and PDL-on launches. They cover tied scores, signed zero, NaN payloads, infinities, visibility masks, stride padding, and selection boundaries. The deep fallback additionally compares the three-launch chain against the previous engine. [Parity results](parity.json)

Compute Sanitizer reported zero race hazards on the tied 256-score top-k case and zero memory errors across all 32 new top-k and mask cases. [Racecheck](racecheck.log) · [Memcheck](memcheck.log)

The engine’s fresh-cache host suite passed **798/800**. The two failures were the previously recorded prompt-endstate eligibility and expert-hot-cache tie tests. The recipe’s eight standalone indexer tests passed and its patched CUDA compilation unit built for `sm_121`. Both service executables, `tensorfold-native` and `tf-dsv41-lanes`, built for Linux x86-64 and AArch64 with the patched `sm_121` code embedded (local CUDA 13.5). The GPU measurements above used CUDA 13.0 on the Spark. [Previously recorded host failures](../../fused-next-20261121/review/SUPERVISOR.md)

These measurements establish per-kernel and per-chain savings. End-to-end prefill and decode throughput on the patched revision remain unmeasured. The original eight-stream throughput dip and the earlier intermittent deep-indexer mismatch remain unresolved.

## Reproduction

From `tools/fused_bench/`, supply CUDA source directories from the two engine revisions. The harness records the resolved kernel symbols, waits for an idle service and holds the shared GPU lock during benchmarking.

```bash
bash build.sh "$OLD_CUDA" "$NEW_CUDA" bin
python3 bench.py bin bin "$KIT_CUBINS" --kernel topk-indices,cand-mask,fallback --json index-timing.json
python3 bench.py bin bin "$KIT_CUBINS" --parity-only --json parity.json
python3 sass.py bin --json sass.json
```

The comparison uses `d5ecba0` as the old source and `d0c9fb1` as the new source. The engine flags were `-O3 -arch=sm_121 --fmad=false --ftz=false` with CUDA 13.0.88.
