# report-0263 — TF_DS_KV_RING_BATCH: NOT DONE (escalated, no commit)

**Status:** stopped before any code. The worktree stays at `33091fb` (0261). No commit, no harness run.

## Why: the QUANT step of `_kv_norm_rope` cannot be pinned on this machine

The brief asks for a kernel that reproduces `_kv_norm_rope`'s ring bytes bit for bit: RMSNorm, RoPE on the last
`rd` dims, then an "FP8 (E4M3) quant-dequant round trip" before the bf16 ring store. It prescribes the norm
(`tf_ds_triton_rmsnorm_row`) and the RoPE contraction. It does **not** say how the quant-dequant scales: which dims
are quantized, the tile size, the scale rule and floor, rounding, saturation, or NaN handling. None of the local
sources can supply that:

1. **The served kit's `_kv_norm_rope` source and cubin are not on this machine.** `docs/kernel-kit-provenance.md:17`
   names the source as `src/tensorfold/families/deepseek_v41/cuda/kernels.py` on the public TensorFold
   `deepseek-v41-tp2` branch. No local clone has that path: searched every `TensorFold*` checkout's branches, every
   `tfref-*`, `p0195`/`p0197`, the DeepSeek repo, and a filesystem-wide search for `deepseek_v41` dirs and
   `kernels.py`. No `cuobjdump`/`nvdisasm` either.
2. **The local references stop at the launch contract:**
   - `fixtures/launches.json` and `runs/overnight-opt/engineering02/bf16-partials/diagnostics/aot.json:8900` give
     only the signature and constants: `Y` bf16 [R, 512], `W` **bf16** [512], `COS`/`SIN` fp32 [1052672, 32],
     `POS`/`SLOT_OF` i64, `OUT`/`RING` bf16, `ring_size` i32, `eps` fp32 (1e-20 in the absorb records),
     consts `QUANT=1, D=512, RD=64`, num_warps 4, shared 16 B.
   - `tri_norm.zig:258` (`kv_norm_rope launches as recorded`) is a launch-conformance test (shapes and arguments). It
     replays no tensor bytes.
3. **The brief's "existing pattern" is not a value round trip.** The only `__nv_cvt_float_to_fp8` in
   `zig/kernels/cuda/*.cu` is `dsv41_torch.cu:873` in `tf_ds_latent_code`: the fp4 store's **E4M3 scale-byte
   encoder** (`f = max(amax16, 6*2^-9) / 6` -> byte, then the codes are compared in input units). That is a block
   scale for E2M1 codes, not an E4M3 quant-dequant of the values. The brief's host-test item ("pin all 256 byte
   values") suggests an unscaled per-element round trip. If the kit scales by tiles, an unscaled round trip would
   saturate above 448 and flush small values, which is wrong on most real rows.
4. **The one related CUDA port is a different kit.** `/home/colum/Workspace/TensorFold/zig/kernels/cuda/deepseek_v41/
   kv_glue.cu` (`kv_norm_store`, read only, not touched) shows DeepSeek-V4-style KV quantisation:
   - 64-element tiles over the first 448 (NoPE) dims;
   - a pow2 (UE8M0) scale from the tile's `fmaxf` peak with floor `0x38d1b717`,
     `e = exp(peak) - 135 + (mantissa > 0x600000)`;
   - an explicit fp32 pre-rounding before `__nv_cvt_float_to_fp8(SATFINITE, E4M3)`;
   - the RoPE dims kept bf16 with `ne = fma(even, c, -(odd*s))`, `no = fma(even, s, odd*c)`;
   - NoPE dims passed through `fma(even,1,-(0*odd))` / `fma(0,even,odd)`.

   It matches a different AOT pair (`_rms` + `_kv_store`, fp64 RMS tail, **fp32** weights, FP8 storage), while the
   served `_kv_norm_rope` takes a **bf16** `W` and dequantizes back into a bf16 ring. So it is a strong lead, but it
   does not specify the served kernel's RMSNorm instructions, its dequant order, or its bf16 rounding point.

Implementing any one of these schemes would mean improvising kit arithmetic. The h0263 harness could not catch a
wrong choice: it compares the kernel against a reference written from the same assumption, so it would pass for any
scheme. Only the stage/row/slot indexing would actually be verified. The brief's own "Triton bit-parity is
hardware-gate only" covers instruction-level rounding, not an unknown algorithm. A wrong scheme would also corrupt
every drafter ring the moment the switch is on.

## What the brief owner must supply (any one of these unblocks it)

- The served `_kv_norm_rope` source (`kernels.py`, provenance doc line 17), or its SASS / a local cubin dump.
  Deliberately **not** fetched from outside this machine: out of this round's scope.
- Or a written spec covering:
  - which dims are quantized (the first `D - RD` = 448?);
  - the tile size (64?);
  - the scale rule and format (UE8M0 pow2 with which floor and exponent formula?);
  - E4M3 rounding (RNE with the small-value path?), saturation and NaN handling;
  - the dequant multiply and its bf16 rounding point;
  - whether the RMSNorm is the `_rmsnorm` 512-cubin sequence `tf_ds_triton_rmsnorm_row<512>` replicates
    (squares4 order, `sqrt.approx` + `div.approx`) or something else (rsqrt, fp64 tail as in `kv_glue.cu`).
- Or real kit bytes for a few rows (Y, W, POS, COS/SIN rows and the resulting RING rows) to validate against on CPU.

## Groundwork that stays valid

- `dr.ao` has no reader: `draft.zig:321` (field), `:475` (allocation), and the two `kvNormRope` out arguments at
  `:505`/`:531` are its only uses. The batched kernel can drop the write.
- The call sites and slot arithmetic match the brief:
  - `absorb` (`draft.zig:497-506`): ring base `pool.view(j, slot)`, `ring_rows = pool.ring`,
    slots `(first + r) % pool.ring`.
  - `absorbMany` (`:518-532`): `pool.rings[j]`, `ring_rows = pool.slots * pool.ring`,
    slots `slot * pool.ring + p % pool.ring`.
  - `max_stages = 4` (`draft.zig:48`).
  - A third `kvNormRope` user, `draft.zig:752` (the pass, out `dr.kvb`), is outside this brief.
- `D = 512` is a supported `tf_ds_triton_rmsnorm_row` width (`static_assert(D == 128 || D == 512)`), so the 128-wide
  restriction in `rope_store.zig`'s `key_width` note does not apply here.
- The launch-count win stays as briefed: 3 launches -> 1 per absorb (served checkpoint: 3 stages), and 21 KiB of
  `dr.ao` writes per rank at seven rows go away.
