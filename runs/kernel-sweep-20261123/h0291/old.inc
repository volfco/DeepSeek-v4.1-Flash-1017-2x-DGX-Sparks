// extracted from /home/colum/Workspace/tfref-kstack4/zig/kernels/cuda/dsv41_torch.cu by emu/extract.py - do not edit

// E2M1 magnitude of an already-clamped t (|t| <= 6), ties to the even code (kernels.py `_e2m1_round`).
__device__ __forceinline__ unsigned tf_ds_e2m1_code(float t) {
    const float a = fabsf(t);
    const unsigned idx = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f)
                       + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return idx | ((t < 0.0f && idx > 0u) ? 8u : 0u);
}

// The E2M1 magnitude as a float (kernels.py's g table).
__device__ __forceinline__ float tf_ds_e2m1_mag(unsigned idx) {
    switch (idx) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        default: return 6.0f;
    }
}

// E4M3 byte -> float (the scale the codes are dequantized with).
__device__ __forceinline__ float tf_ds_e4m3_to_float(unsigned b) {
    const unsigned exp = (b >> 3) & 15u, mantissa = b & 7u;
    // The encoder returns positive scale bytes through 0x7e.
    return exp ? __uint_as_float(((exp + 120u) << 23) | (mantissa << 20)) : mantissa * 0x1p-9f;
}

// The kit's FMNMX instructions select the numeric operand for a NaN.
__device__ __forceinline__ float tf_ds_clamp6(float v) { return fminf(fmaxf(v, -6.f), 6.f); }

// A BF16 magnitude as sign-cleared bits, NaNs mapped to zero: the float reduction's fmaxf(0, |x|) drops a NaN the
// same way, and for non-negative finite values the bit pattern orders exactly as the number does.
__device__ __forceinline__ unsigned tf_ds_abs_bits(float v) {
    const unsigned b = __float_as_uint(v) >> 16;
    return ((b & 0x7fffu) > 0x7f80u) ? 0u : (b & 0x7fffu);
}

// The 32-element scale group's largest magnitude as those bits, through the warp's integer maximum: the key and query
// groups are whole warps (see the alignment note), so this is one instruction. The latent's 16-element groups keep the
// shuffle maximum below - a half-warp masked `__reduce_max_sync` measured 0.10 us slower at every served latent row
// (1.541 -> 1.644 us at R=1/7/32/48), so the integer maximum stays where a scale group is a whole warp.
__device__ __forceinline__ unsigned tf_ds_fp4_amax_bits(float x) {
    return __reduce_max_sync(0xffffffffu, tf_ds_abs_bits(x));
}

template<int BLOCK>
__device__ __forceinline__ float tf_ds_fp4_amax(float x) {
    float m = fmaxf(0.f, fabsf(x));
    #pragma unroll
    for (int delta = BLOCK / 2; delta; delta >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, delta, BLOCK));
    return m;
}

// The E8M0 scale byte of such a magnitude, from its BF16 exponent and mantissa instead of an FP32 division by six:
// the kit's f = max(amax, 6*2^-126) / 6, byte = biased_exponent(f) + (f has a mantissa), i.e. ceil(log2(amax/6)) +
// 127 floored at 1. amax = m * 2^E with m in [1, 2) gives byte = E + 125 for m <= 1.5 and E + 126 above it, so the
// byte is exp - 2 + (mantissa > 1.5's) with a floor of 1; an infinite magnitude keeps the kit's infinite scale.
__device__ __forceinline__ unsigned tf_ds_scale_byte_bits(unsigned bits) {
    const unsigned exp = (bits >> 7) & 0xffu, mant = bits & 0x7fu;
    if (exp == 0xffu) return 255u;
    const int b = (int)exp - 2 + (mant > 0x40u ? 1 : 0);
    return (unsigned)(b < 1 ? 1 : b);
}

// 1 / 2^(byte - 127) exactly, zero for an infinite scale (no other byte reaches 255). Multiplying by it is the same
// IEEE scaling as dividing by the scale: both are exact powers of two, so the quotient rounds identically.
__device__ __forceinline__ float tf_ds_recip_bits(unsigned byte) {
    return byte == 255u ? 0.f : __uint_as_float((254u - byte) << 23);
}


__global__ void __launch_bounds__(128) tf_ds_fused_rope_qd_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                                             const float* cos, const float* sin,
                                                                             const long long* pos,
                                                                             int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x, h = blockIdx.y;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    for (int i = threadIdx.x; i < HD; i += blockDim.x) {
        float v = __bfloat162float(x[base + i]);
        const float other = __shfl_xor_sync(0xffffffffu, v, 1);
        if (i >= HD - RD) {
            const int j = (i - (HD - RD)) / 2;
            const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
            const float rot = (i & 1) ? __fmaf_rn(other, sn, __fmul_rn(v, cs))
                                      : __fmaf_rn(v, cs, -__fmul_rn(other, sn));
            v = __bfloat162float(__float2bfloat16_rn(rot));
        }
        const unsigned byte = tf_ds_scale_byte_bits(tf_ds_fp4_amax_bits(v));
        const float scale = __uint_as_float(byte << 23);
        const float t = tf_ds_clamp6(__fmul_rn(v, tf_ds_recip_bits(byte)));
        const float q = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t) & 7u), t);
        out[base + i] = __float2bfloat16_rn(__fmul_rn(q, scale));
    }
}

// The pair geometry: a warp owns one head and walks it in 64-element chunks, and each lane holds the rotary pair
// (2l, 2l + 1) of the chunk in one 4-byte load, so the partner is a register instead of a shuffle and the 32-element
// scale groups are the two half-warps. Same arithmetic and same bytes as the one-element form above.
constexpr int TF_DS_QD_PAIR_WARPS = 8;

__device__ __forceinline__ void tf_ds_qd_pair_chunk(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                                    const float* sin, long long base, long long p, int chunk, int lane,
                                                    int HD, int RD) {
    const int i0 = chunk + 2 * lane;
    const uint32_t raw = *(const uint32_t*)(x + base + i0);
    const __nv_bfloat162 pr = *(__nv_bfloat162*)&raw;
    float v0 = __bfloat162float(pr.x), v1 = __bfloat162float(pr.y);
    if (i0 >= HD - RD) {
        const int j = (i0 - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float r0 = __fmaf_rn(v0, cs, -__fmul_rn(v1, sn));
        const float r1 = __fmaf_rn(v0, sn, __fmul_rn(v1, cs));  // the kit form: im = fma(xe, sn, xo*cs)
        v0 = __bfloat162float(__float2bfloat16_rn(r0));
        v1 = __bfloat162float(__float2bfloat16_rn(r1));
    }
    // the two half-warps are the chunk's two 32-element scale groups
    const unsigned amax = __reduce_max_sync((lane < 16) ? 0x0000ffffu : 0xffff0000u,
                                            max(tf_ds_abs_bits(v0), tf_ds_abs_bits(v1)));
    const unsigned byte = tf_ds_scale_byte_bits(amax);
    const float scale = __uint_as_float(byte << 23);
    const float recip = tf_ds_recip_bits(byte);
    const float t0 = tf_ds_clamp6(__fmul_rn(v0, recip)), t1 = tf_ds_clamp6(__fmul_rn(v1, recip));
    const float q0 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t0) & 7u), t0);
    const float q1 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t1) & 7u), t1);
    const __nv_bfloat162 o = {__float2bfloat16_rn(__fmul_rn(q0, scale)),
                              __float2bfloat16_rn(__fmul_rn(q1, scale))};
    *(uint32_t*)(out + base + i0) = *(const uint32_t*)&o;
}

__global__ void __launch_bounds__(32 * TF_DS_QD_PAIR_WARPS)
    tf_ds_fused_rope_qd_pair_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos, const float* sin,
                                    const long long* pos, int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.x, h = blockIdx.y * TF_DS_QD_PAIR_WARPS + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    for (int chunk = 0; chunk < HD; chunk += 64)
        tf_ds_qd_pair_chunk(x, out, cos, sin, base, p, chunk, lane, HD, RD);
}

