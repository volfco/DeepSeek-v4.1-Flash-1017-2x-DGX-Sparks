# 128-row indexer query rotation and FP4 conversion

The deployed kit’s two-launch chain was faster for an isolated 128-row prompt indexer query on the GB10 head. It took 8.35–8.64 µs versus 10.82–12.05 µs for `query_pair`, a 23–28% reduction in median elapsed time across two runs. The result reversed in a graph with eight consecutive invocations using distinct query and output buffers. The same-buffer control favored the kit, so the reversal depends on the workload used for the benchmark.

Each row reports 80 paired samples, in µs per query. Parentheses show IQR for each arm. The interval is a 95% bootstrap interval for the paired median difference, resampling complete pairs. Positive differences favor the kit.

| Graph workload | Run | Fused median (IQR) | Kit median (IQR) | Paired fused − kit, 95% interval | Kit faster |
|---|---|---:|---:|---:|---:|
| 1 query | Forward | 10.816 (1.248) | 8.352 (0.360) | +2.560 [2.400, 2.656] | 79/80 |
| 1 query | Reverse | 12.048 (1.320) | 8.640 (0.584) | +3.296 [2.848, 3.472] | 80/80 |
| 2 distinct buffers | Forward | 9.568 (0.344) | 7.288 (0.652) | +2.304 [2.208, 2.528] | 80/80 |
| 2 distinct buffers | Reverse | 9.736 (0.368) | 7.528 (0.336) | +2.200 [2.112, 2.368] | 79/80 |
| 4 distinct buffers | Forward | 8.388 (0.132) | 6.312 (0.380) | +2.100 [2.040, 2.172] | 80/80 |
| 4 distinct buffers | Reverse | 8.908 (0.112) | 6.476 (0.388) | +2.408 [2.352, 2.480] | 80/80 |
| 8 distinct buffers | Forward | 8.444 (0.026) | 9.572 (0.272) | −1.140 [−1.172, −1.044] | 0/80 |
| 8 distinct buffers | Reverse | 8.916 (0.218) | 9.814 (0.256) | −0.976 [−1.022, −0.892] | 0/80 |
| 8 uses of one buffer, identity RoPE | Forward | 6.508 (0.185) | 5.444 (0.205) | +1.068 [1.032, 1.112] | 80/80 |
| 8 uses of one buffer, identity RoPE | Reverse | 6.632 (0.173) | 5.464 (0.210) | +1.080 [1.036, 1.128] | 80/80 |

## Scope

- The test ran on `gx10-eb0e` at 05:26 UTC on 2026-11-22. It loaded the original CUDA fatbin extracted from the shipped `bench-fused-next-84806bb` image. The native and lanes executables embedded byte-identical fatbins, and their hashes match the suite artifacts. Both arms ran in the same image and CUDA context.
- Production model configuration specifies 32 indexer heads, head dimension 128 and rotary dimension 64. Each query uses 128 contiguous rows with 16-byte-aligned positions, matching the prompt indexer. The fused arm dispatched `tf_ds_fused_rope_qd_pair_kernel` with grid (128, 4), block 256 and PDL disabled, as in the serving trace. The fallback used the deployed kit’s aligned-position `_rope_heads` (`f07b206…`, grid (128, 2), block 128), then `_fp4qd_p2` (`4cbef7…`, grid (512, 1), block 128), both with the kit’s PDL launch attribute enabled. [Candidate serving trace](../s05.rank0.log#L139)
- Both arms used the same query and output addresses, contiguous positions and seeded RoPE tables. A device copy restored pristine BF16 inputs before the first timing event in every graph replay because the kit rotates its input in place. Within a graph, each invocation used its own buffer pair unless the row explicitly specifies the one-buffer control. That control used identity rotation so repeated reads saw the same values. The two runs reversed graph-size order, and each run alternated which arm went first in each pair. GPU timestamp nodes were inside the graph.
- Every output byte matched for seeded and special-value inputs (1,048,576 bytes per input), and every captured graph matched. The benchmark held the existing GPU lock and checked for idle production before initializing CUDA and before each replay. The prompt and generation counters stayed at 1,174,064 and 16,331 throughout both runs. Production continued running.
- These tests measure isolated kernel chains with synthetic activations. A served layer adds the query projection, scoring and adjacent CUDA work, which can change cache residency and PDL overlap. The eight-buffer graph also changes the working set, so its reversal cannot be attributed to graph length alone. End-to-end serving impact remains unmeasured.

Raw samples, checksums and kernel metadata are saved for both runs. [Forward run](head.json) · [Reverse run](head-reverse.json)

Reproduce from the recipe root with `just fused-query128` and `just fused-query128 1`.
