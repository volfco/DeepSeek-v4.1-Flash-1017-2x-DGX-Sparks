# _kv_norm_rope (kit source) — the spec 0263's writer could not pin

Fetched 2026-11-22 from the public kit source `bertholomus/TensorFold` @ `deepseek-v41-tp2`, `src/tensorfold/families/deepseek_v41/cuda/kernels.py` (lines 1199–1265 at fetch time). Key facts for a CUDA replication:
- RMSNorm over D with the **even and odd halves summed separately** (`tl.sum(xe*xe) + tl.sum(xo*xo)`), `rinv = 1/sqrt(sum/D + eps)`, each half rounded to bf16 after the weight multiply. This is not `_rmsnorm`'s single reduction, so `tf_ds_triton_rmsnorm_row` does not apply as-is.
- RoPE on the last RD dims (adjacent pairs): `re = ne*cs - no*sn`, `im = ne*sn + no*cs`, each rounded to bf16 (Triton may contract to fma; see continue.md's register).
- FP8 when QUANT: per 32-element block (16 pairs), `amax = max(|re|,|im|)`, `s = _pow2_ceil(max(amax, 1e-4)/448)` (exponent from bits, +1 if the mantissa is non-zero, `exp2`), `clamp(v/s, ±448)` → `float8e4nv` → fp32 → `*s`, then stored as bf16 to OUT and to RING[slot] when slot ≥ 0.

```python
# -- quantization helpers -------------------------------------------------------------------------------------------
@triton.jit
def _pow2_ceil(v):
    bits = v.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    return tl.exp2(e.to(tl.float32))


@triton.jit
def _e2m1(code):
    """E2M1 nibble (sign bit 3) -> fp32 by building the float's bits: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6."""

    m = code & 7
    bits = tl.where(m >= 2, (((m >> 1) + 126) << 23) | ((m & 1) << 22), tl.where(m == 1, 0x3F000000, 0))
    bits = bits | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


# -- the window KV: RMSNorm, RoPE on the last 64 (adjacent pairs), FP8 quant-dequant per 32, into the ring ------
@triton.jit
def _kv_norm_rope(Y, W, COS, SIN, POS, OUT, RING, SLOT_OF, ring_size, eps, QUANT: tl.constexpr,
                  D: tl.constexpr, RD: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)
    xe = tl.load(Y + r * D + 2 * i).to(tl.float32)
    xo = tl.load(Y + r * D + 2 * i + 1).to(tl.float32)
    rinv = 1.0 / tl.sqrt((tl.sum(xe * xe, axis=0) + tl.sum(xo * xo, axis=0)) / D + eps)
    we = tl.load(W + 2 * i).to(tl.float32)
    wo = tl.load(W + 2 * i + 1).to(tl.float32)
    ne = (we * (xe * rinv)).to(tl.bfloat16).to(tl.float32)
    no = (wo * (xo * rinv)).to(tl.bfloat16).to(tl.float32)
    p = tl.load(POS + r)
    PAIRS0: tl.constexpr = HALF - RD // 2
    j = tl.maximum(i - PAIRS0, 0)
    cs = tl.load(COS + p * (RD // 2) + j)
    sn = tl.load(SIN + p * (RD // 2) + j)
    rot = i >= PAIRS0
    re = tl.where(rot, (ne * cs - no * sn), ne).to(tl.bfloat16).to(tl.float32)
    im = tl.where(rot, (ne * sn + no * cs), no).to(tl.bfloat16).to(tl.float32)
    if QUANT:
        # 32-element blocks = 16 pairs
        a = tl.maximum(tl.abs(re), tl.abs(im))
        amax = tl.max(tl.reshape(a, (HALF // 16, 16)), axis=1)
        s = _pow2_ceil(tl.maximum(amax, 1e-4) / 448.0)
        sb = tl.reshape(tl.broadcast_to(s[:, None], (HALF // 16, 16)), (HALF,))
        re = (tl.minimum(tl.maximum(re / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb
        im = (tl.minimum(tl.maximum(im / sb, -448.0), 448.0)).to(tl.float8e4nv).to(tl.float32) * sb
    tl.store(OUT + r * D + 2 * i, re.to(tl.bfloat16))
    tl.store(OUT + r * D + 2 * i + 1, im.to(tl.bfloat16))
    slot = tl.load(SLOT_OF + r)
    if slot >= 0:
        tl.store(RING + slot * D + 2 * i, re.to(tl.bfloat16))
        tl.store(RING + slot * D + 2 * i + 1, im.to(tl.bfloat16))


def kv_norm_rope(y: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor,
                 ring: torch.Tensor, slots: torch.Tensor, eps: float, quant: bool, rd: int,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    rows, d = y.shape
    if out is None:
        out = torch.empty_like(y)
    _kv_norm_rope[(rows,)](y, w, cos, sin, pos, out, ring, slots, ring.shape[0], eps, QUANT=quant, D=d, RD=rd,
                           num_warps=4, **_pdl())
    return out

```

**Priority (2026-11-22):** parked. 0263 removes two launches per drafter pass, about 10 µs of a 4.5–7 ms pass (≈0.02% of a round), and adds a bit-exactness risk in the split reduction.
