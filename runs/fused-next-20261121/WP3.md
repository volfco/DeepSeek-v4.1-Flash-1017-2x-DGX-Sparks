# WP3: the 0198 rope-store kernels (decode key/latent/query, prompt key/latent; `rope_store.zig`, `exact.zig`)

Evidence folder: `runs/fused-next-20261121/wp3/`. Engine commits numbered from **0240**. Use the WP0 harness. Every
FP4/E8M0 change must match the kit's `_fp4_store`/`_fp4qd_p2` cubins on all 65,536 BF16 inputs at every reachable
scale (the earlier audit did 8,257,536 pairs; do at least that).

Items (todo.md text is the spec):

1. **Skip RoPE partner shuffles on wholly unrotated warps** (item 1): read SASS first; guard only whole unrotated warps,
   full-mask shuffles stay for mixed/rotated warps; the warp-alignment invariant (0204's `warpShapesOk`) must hold.
2. **E8M0 scale from BF16 bits** (no FP32 division by six), **exact reciprocal multiply** instead of per-element
   division, and **warp integer max** (`__reduce_max_sync` on sign-cleared bits, NaN mapped to zero) for the FP4 scale
   groups (Compression and FP4 packing items).
3. **CUDA FP4 conversion intrinsics** (research item): opt-in experiment, pre-clamp with `tf_ds_clamp6` (the intrinsic
   maps NaN to +MAXNORM, the kit needs -6); accept only on exhaustive parity plus SASS/timing evidence.
4. **Skip latent packing on non-owner decode rows** (Compression item): sharded decode sends them to a trash row; skip
   rotation, packing and stores after the index-key projection has consumed `rd.lat`. Mixed-owner rounds on both ranks
   must keep identical bytes where read.
5. **Larger fusion: compressor + latent RoPE/FP4** (research item "Larger fusions"): only if WP0/WP2 timings show the
   boundary between 0197 and 0198 costs more than a launch floor; `rd.lat` still feeds the index-key projection first.

Finish with `wp3/RESULTS.md` and each kernel's final position against the launch floor and roofline.
