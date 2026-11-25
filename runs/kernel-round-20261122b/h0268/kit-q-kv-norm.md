# _q_kv_norm (kit source) — what 0268's writer pinned

Fetched 2026-11-22 from the public kit source `bertholomus/TensorFold` @ `deepseek-v41-tp2`,
`src/tensorfold/families/deepseek_v41/cuda/kernels.py` (the `_rmsnorm_rot`, `_q_kv_norm`, `q_kv_norm`,
`_kv_norm_rope` and `kv_norm_rope` sections; `_kv_norm_rope` is the kernel `runs/kernel-round-20261122/kit-kv-norm-rope.md`
documents in full). Key facts for 0268:

- **`_q_kv_norm` is `_rmsnorm_rot` and `_kv_norm_rope` of the same rows in one launch.** Its docstring: *"the two
  independent norms after attn_in, each program the code of the kernel it replaces"* — programs `(r, 0)` run
  `_rmsnorm_rot`'s body (the norm's OUT row, and with `SPLIT` off the rotations), `(r, 1)` runs `_kv_norm_rope`'s
  body verbatim (same even/odd-half reduction, same RoPE `ne * cs - no * sn` contraction, same FP8 round trip,
  same stores), and `(r, 2 + i)` with `SPLIT` recompute the norm (comment: "the same code, the same bits") and run
  rotation `i` alone. The served build runs `qkv_split` on (`SPLIT = len(rot) > 0`), one rotation for the drafter:
  grid `(rows, 3, 1)`.
- **The −1 sentinel is guarded exactly as in `_kv_norm_rope`.** Both kernels end with:
  ```python
  slot = tl.load(SLOT_OF + r)
  if slot >= 0:
      tl.store(RING + slot * D + 2 * i, re.to(tl.bfloat16))
      tl.store(RING + slot * D + 2 * i + 1, im.to(tl.bfloat16))
  ```
  The drafter's block rows pass −1 every row (`draft.zig`'s `row(5)`), so the joint launch leaves the ring untouched.
- The wrapper `q_kv_norm(x, w, eps, rot, y, wk, cos, sin, pos, ring, slots, quant, rd)`:
  `_q_kv_norm[(rows, 2 + (len(rot) if split else 0))](x, x.stride(0), w, out, out.stride(0), eps, *st, y, wk, cos,
  sin, pos, kout, ring, slots, ring.shape[0], QUANT=quant, D=d, BLOCK=triton.next_power_of_2(d), NROT=len(rot),
  DK=dk, RD=rd, num_warps=4, SPLIT=split, **_pdl())` — what `tri_norm.qKvNorm` reproduces (already recorded
  `test "q_kv_norm launches as recorded"`; the drafter's joint call is that launch at its own row counts).
- **Rounding:** each program is the code of the kernel it replaces, but `_q_kv_norm` is a different compiled cubin
  from `_rmsnorm_rot` and `_kv_norm_rope`. Source-level arithmetic is identical; compiler-level bit-identity between
  the cubins is a GPU-gate question (byte-compare OUT/dr.qr, H0/dr.xhq, KOUT/dr.kvb and the ring rows).
