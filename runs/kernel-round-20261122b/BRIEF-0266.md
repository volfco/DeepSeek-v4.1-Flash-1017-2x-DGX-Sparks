# BRIEF-0266 — 0197 decode compressor computes a repeated ratio-2 pair once per window

**Switch:** `TF_DS_COMPRESS2_PAIR_ONCE` (off by default; takes effect only with `TF_DS_FUSED_COMPRESS2_NORM=1`,
which production sets). **Module:** `zig/src/families/dsv41/compress2_pair.zig` (pure).
**Commit subject:** `0266 dsv41: TF_DS_COMPRESS2_PAIR_ONCE (off by default): 0197's decode compressor computes each distinct ratio-2 pair once a window`

## The item (todo.md, "Compression and FP4 packing")

Rows with the same ordered pair of resolved source slots get the same softmax and normalized latent. One contiguous
seven-row window has four distinct pairs. Compute each once and deliver its BF16 result to every corresponding output
row. Every input row still writes its own raw-ring slot. Check separate requests, ring wrap and partners read from an
earlier round. Start with the seven-row fast path. Byte-compare raw-ring and latent outputs at odd and even starting
positions, including mixed-request batches.

## Where

`tf_ds_compress2x_norm_body` / its `extern "C"` wrappers in `zig/kernels/cuda/dsv41_torch.cu` (~lines 592–680), the
host side in `ops.zig` / the decode call site (grep `compress2x_norm`). Patches `zig/patches/0197-*`, `0210-*` in the
recipe repo (`/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/zig/patches/`) explain the kernel.

## Contract

- Off: launch- and byte-identical.
- On: every output byte (raw ring KV and score slots, normalized latent rows, padding, invalid flag) identical to the
  off path. The computing row's arithmetic must be exactly the existing per-row arithmetic (same instruction order);
  only *which* thread/block computes it and how the result is broadcast changes. Ordering: a duplicate row must not read
  the shared result before it is written (in-block `__syncthreads` if the pair's rows share a block; if they are in
  different blocks, redesign so they share one, or stop).
- Decide the pair key from the resolved source slots exactly as the kernel resolves them (current round's rows vs ring
  partners from an earlier round, wrap). Rows from different requests never share.
- Report the current kernel's grid (blocks per row, threads), the new grid, and how many pair computations a 7-row
  window and a 4×(5..7)-row round save. Current timing is about 2.0 µs at 7 × 512 (continue.md, 0197 table), so the
  ceiling is small; if the new design adds a barrier that plausibly costs more than it saves, say so.

## Harness (`h0266/`)

Extract the kernel (p0194 `splice.py` + `fused_run.cpp` trick); this kernel has block-level cooperation, so emulate
barriers (run threads in phases between `__syncthreads`). Compare on: 1..7-row windows at odd/even start positions,
ring wrap, a partner from the previous round, mixed requests (2–4 streams per launch), widths 128 and 512. Against an
independent per-row reference and against the off-path kernel's emulation.
