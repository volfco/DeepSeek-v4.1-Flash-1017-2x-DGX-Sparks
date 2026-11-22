# 0215 Small-Row Crossover & 0254 Distribution Probe (`runs/pre-bench-20261121/crossover-0215`)

## Summary

1. **Byte-Exact Parity on GB10 (`sm_121`, `d0c9fb1`): 36 / 36 PASS (`0` byte diffs)**
   - Compiled `dsv41_ops.cu` + `empty_kernel` floor into isolated `100.95.49.21:~/tf-bench/pre-bench-20261121/crossover/bin/` (`nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`, CUDA 13.0.88).
   - Verified **32 `0215` small-row crossover cases** (`R in {1, 7}` × `nbk in {2049, 4096, 5120, 6144, 7168, 8192, 12288, 16384}` × `{random, ties}`, `k = 2048`):
     - Single-launch `tf_ds_idx_fused_kernel` vs two-launch `tf_ds_block_max_kernel + tf_ds_idx_pick_kernel` vs unfused three-launch `block_max + topk_indices_cut + pool_pick` vs NumPy exact reference: **`0` differing bytes across all 6 pairwise comparisons + tail sentinel checks (`parity.json`, `parity.log`)**.
   - Verified **4 `0254` `topk_indices_cut` distribution cases** at `128 × 32,768 (k=512)` (`random`, `applyPool` 50% `-inf`, `vis = 128 < k`, `all -inf`, both 16B-aligned and odd `pad=3` stride): **`0` differing bytes** across `topk_indices_cut` (`pdl=0/1`), legacy `topk_indices`, `apply_pool`, and NumPy reference.
   - Peak CUDA memory footprint: **`18.004 MiB`** (`< 192 MiB` budget), elapsed **`2.242 s`** under `flock ~/tf-bench/gpu.lock`, using stream-ordered `CU_STREAM_NON_BLOCKING` transfers (`test_measure.py` CPU unit tests passing).

2. **Guarded CUDA-Graph Timing Status (`timing_valid: false` — Live Production Serving Active)**
   - Per `todo.md` ("Make timing validity explicit for each benchmark case" and "Measure the 0215 small-row crossover... Save medians, IQRs and output parity for both paths before changing the threshold"), `run_timing_loop.py` polled `http://127.0.0.1:8000/health` for an idle window (`live.connections == 0` and valid unchanged `prompt_tokens_total`, `generation_tokens_total` across each case).
   - Production served continuous external client traffic throughout the window (`connections = 1`, `~69–73 decode tok/s`, prompt tokens advancing `74,020,420 -> 74,486,116`). During the single brief `connections == 0` instant at `+93.1 s`, a new client request arrived mid-burst (`before conns=1`); the strict guard rejected and discarded the contaminated sample (`0.023 s`), yielded `gpu.lock`, and recorded `timing_valid: false` (`timing: {}`) on all 36 rows in `results.json`.
   - **Threshold decision:** Keep `idx_fused.small_nbk = 6144` unchanged until an idle `/health` window captures `timing_valid: true` across `nbk = 4096..16384`.

## One-Command Resume When Production Is Idle (No Engine Restart Needed)

```bash
ssh colum@100.95.49.21 'cd ~/tf-bench/pre-bench-20261121/crossover && python3 run_timing_loop.py --wait-seconds 300'
rsync -a colum@100.95.49.21:tf-bench/pre-bench-20261121/crossover/{results.json,timing.log} runs/pre-bench-20261121/crossover-0215/
```
