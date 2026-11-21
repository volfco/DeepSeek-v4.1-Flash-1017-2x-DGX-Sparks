// WP3 candidate kernels for the 0198 rope-store family. Built on the head with the engine's flags:
//   nvcc -O3 -arch=sm_121 --fmad=false --ftz=false -cubin variants.cu
// Every candidate must reproduce the engine kernel's bytes exactly (the harness compares against the kit chain).
#include <cuda_bf16.h>
#include <cstdint>

// ---- the engine's helpers, verbatim from dsv41_torch.cu ------------------------------------------------

__device__ __forceinline__ void pdl_enter(int pdl) {
    if (pdl) {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

__device__ __forceinline__ unsigned tf_ds_e2m1_code(float t) {
    const float a = fabsf(t);
    const unsigned idx = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f)
                       + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return idx | ((t < 0.0f && idx > 0u) ? 8u : 0u);
}

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

__device__ __forceinline__ float tf_ds_clamp6(float v) { return fminf(fmaxf(v, -6.f), 6.f); }

template<int BLOCK>
__device__ __forceinline__ float tf_ds_fp4_amax(float x) {
    float m = fmaxf(0.f, fabsf(x));
    #pragma unroll
    for (int delta = BLOCK / 2; delta; delta >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, delta, BLOCK));
    return m;
}

__device__ __forceinline__ unsigned tf_ds_pow2_exponent(float amax) {
    const unsigned bits = __float_as_uint(__fdiv_rn(fmaxf(amax, 6.f * 0x1p-126f), 6.f));
    return ((bits >> 23) & 255u) + ((bits & 0x7fffffu) != 0);
}

// ---- item 0: the decode query, one element a lane as the engine has it ---------------------------------

// One element's whole chain, as tf_ds_fused_rope_qd_kernel does it. `i` is the element's column in its head.
__device__ __forceinline__ void qd_element(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                          const float* sin, long long base, long long p, int i, int HD, int RD) {
    float v = __bfloat162float(x[base + i]);
    const float other = __shfl_xor_sync(0xffffffffu, v, 1);
    if (i >= HD - RD) {
        const int j = (i - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float rot = (i & 1) ? __fmaf_rn(other, sn, __fmul_rn(v, cs))
                                  : __fmaf_rn(v, cs, -__fmul_rn(other, sn));
        v = __bfloat162float(__float2bfloat16_rn(rot));
    }
    const float scale = __uint_as_float(tf_ds_pow2_exponent(tf_ds_fp4_amax<32>(v)) << 23);
    const float t = tf_ds_clamp6(__fdiv_rn(v, scale));
    const float q = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t) & 7u), t);
    out[base + i] = __float2bfloat16_rn(__fmul_rn(q, scale));
}

// The engine's form: grid = rows * H, block = 128, thread = one element of one head.
extern "C" __global__ void __launch_bounds__(128) v_q_cur(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x, h = blockIdx.y;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    for (int i = threadIdx.x; i < HD; i += blockDim.x) qd_element(x, out, cos, sin, base, p, i, HD, RD);
}

// HPB heads a block, 128 threads a head: `tid / 128` picks the head, `tid % 128` the column (warp aligned).
template<int HPB>
__device__ __forceinline__ void qd_heads(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                         const float* sin, const long long* pos, int H, int HD, int RD) {
    const int tid = threadIdx.x;
    const int hloc = tid / 128, i = tid % 128;
    const int blk = blockIdx.x, r = blk / (H / HPB), hg = blk % (H / HPB);
    const int h = hg * HPB + hloc;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * HD;
    qd_element(x, out, cos, sin, base, pos[r], i, HD, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_h2(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                         const float* cos, const float* sin, const long long* pos,
                                                         int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_heads<2>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(512) v_q_h4(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                         const float* cos, const float* sin, const long long* pos,
                                                         int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_heads<4>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(1024) v_q_h8(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_heads<8>(x, out, cos, sin, pos, H, HD, RD);
}

// One block a row, 128 threads, the head loop inside the block.
extern "C" __global__ void __launch_bounds__(128) v_q_row(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x;
    const long long p = pos[r];
    for (int h = 0; h < H; ++h) qd_element(x, out, cos, sin, ((long long)r * H + h) * HD, p, threadIdx.x, HD, RD);
}

// One block a row, 512 threads: four heads at a time, the head loop inside the block.
extern "C" __global__ void __launch_bounds__(512) v_q_row4(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                           const float* cos, const float* sin, const long long* pos,
                                                           int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x;
    const long long p = pos[r];
    const int hloc = threadIdx.x / 128, i = threadIdx.x % 128;
    for (int h = hloc; h < H; h += 4) qd_element(x, out, cos, sin, ((long long)r * H + h) * HD, p, i, HD, RD);
}

// ---- item 2: the scale from BF16 bits, the exact reciprocal, the warp integer maximum --------------------

// E8M0 scale byte of a sign-cleared bf16 magnitude: the kit's ceil(log2(amax/6)) + 127, floor 1 (6*2^-126).
// Derived from the exponent and the mantissa's comparison with 1.5, no FP32 division by six.
__device__ __forceinline__ unsigned tf_ds_scale_byte_bits(unsigned bits) {
    const unsigned exp = (bits >> 7) & 0xffu, mant = bits & 0x7fu;
    if (exp == 0xffu) return 255u;                       // infinity: the kit's scale is infinity
    const int b = (int)exp - 2 + (mant > 0x40u ? 1 : 0);  // (mant > 0x40) <=> amax > 1.5 * 2^E
    return (unsigned)(b < 1 ? 1 : b);
}

// 1 / 2^(byte - 127) exactly, zero for an infinite scale (the byte is never 255 otherwise).
__device__ __forceinline__ float tf_ds_recip_bits(unsigned byte) {
    return byte == 255u ? 0.f : __uint_as_float((254u - byte) << 23);
}

// |bf16| bits with NaNs mapped to zero (the float reduction's fmaxf drops them the same way).
__device__ __forceinline__ unsigned tf_ds_abs_bits(float v) {
    const unsigned b = __float_as_uint(v) >> 16;
    return ((b & 0x7fffu) > 0x7f80u) ? 0u : (b & 0x7fffu);
}

// The query element with the reciprocal multiply (bit-identical: both are exact scalings by a power of two).
__device__ __forceinline__ void qd_element_recip(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                                const float* sin, long long base, long long p, int i, int HD, int RD) {
    float v = __bfloat162float(x[base + i]);
    const float other = __shfl_xor_sync(0xffffffffu, v, 1);
    if (i >= HD - RD) {
        const int j = (i - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float rot = (i & 1) ? __fmaf_rn(other, sn, __fmul_rn(v, cs))
                                  : __fmaf_rn(v, cs, -__fmul_rn(other, sn));
        v = __bfloat162float(__float2bfloat16_rn(rot));
    }
    const float amax = tf_ds_fp4_amax<32>(v);
    const unsigned byte = tf_ds_pow2_exponent(amax);
    const float scale = __uint_as_float(byte << 23);
    const float t = tf_ds_clamp6(__fmul_rn(v, tf_ds_recip_bits(byte)));
    const float q = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t) & 7u), t);
    out[base + i] = __float2bfloat16_rn(__fmul_rn(q, scale));
}

// ... and with the scale byte from the BF16 bits and the group maximum as a warp integer max.
__device__ __forceinline__ void qd_element_bits(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                               const float* sin, long long base, long long p, int i, int HD, int RD) {
    float v = __bfloat162float(x[base + i]);
    const float other = __shfl_xor_sync(0xffffffffu, v, 1);
    if (i >= HD - RD) {
        const int j = (i - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float rot = (i & 1) ? __fmaf_rn(other, sn, __fmul_rn(v, cs))
                                  : __fmaf_rn(v, cs, -__fmul_rn(other, sn));
        v = __bfloat162float(__float2bfloat16_rn(rot));
    }
    const unsigned amax = __reduce_max_sync(0xffffffffu, tf_ds_abs_bits(v));
    const unsigned byte = tf_ds_scale_byte_bits(amax);
    const float scale = __uint_as_float(byte << 23);
    const float t = tf_ds_clamp6(__fmul_rn(v, tf_ds_recip_bits(byte)));
    const float q = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t) & 7u), t);
    out[base + i] = __float2bfloat16_rn(__fmul_rn(q, scale));
}

// The engine's geometry with the reciprocal body.
extern "C" __global__ void __launch_bounds__(128) v_q_recip(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                            const float* cos, const float* sin, const long long* pos,
                                                            int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x, h = blockIdx.y;
    qd_element_recip(x, out, cos, sin, ((long long)r * H + h) * HD, pos[r], threadIdx.x, HD, RD);
}

extern "C" __global__ void __launch_bounds__(128) v_q_bits(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                           const float* cos, const float* sin, const long long* pos,
                                                           int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x, h = blockIdx.y;
    qd_element_bits(x, out, cos, sin, ((long long)r * H + h) * HD, pos[r], threadIdx.x, HD, RD);
}

// A warp a head: four elements a lane (i = lane + 32k), HPB warps a block (HPB heads a block).
template<int HPB, bool BITS>
__device__ __forceinline__ void qd_warp(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                        const float* sin, const long long* pos, int H, int HD, int RD) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int blk = blockIdx.x, r = blk / (H / HPB), hg = blk % (H / HPB);
    const int h = hg * HPB + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    #pragma unroll
    for (int k = 0; k < HD / 32; ++k) {
        const int i = lane + 32 * k;
        if (BITS) qd_element_bits(x, out, cos, sin, base, p, i, HD, RD);
        else qd_element_recip(x, out, cos, sin, base, p, i, HD, RD);
    }
}

extern "C" __global__ void __launch_bounds__(128) v_q_w4r(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_warp<4, false>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(128) v_q_w4b(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_warp<4, true>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_w8b(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_warp<8, true>(x, out, cos, sin, pos, H, HD, RD);
}

// ---- item 0/1: the rotary pair inside one lane (no partner shuffle at all) ------------------------------

// A warp handles one head, one 64-element chunk an iteration: lane l loads the bf16 pair (2l, 2l+1) of the chunk
// as one 4-byte word, so the rotary partner is a register and the 32-element scale groups are the two half-warps.
template<bool BITS>
__device__ __forceinline__ void qd_pair_chunk(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                              const float* sin, long long base, long long p, int chunk, int lane,
                                              int HD, int RD) {
    const int lane2 = lane & 15;
    const unsigned mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
    const int i0 = chunk + 2 * lane;             // the pair's even element
    const uint32_t raw = *(const uint32_t*)(x + base + i0);
    const __nv_bfloat162 pr = *(__nv_bfloat162*)&raw;
    float v0 = __bfloat162float(pr.x), v1 = __bfloat162float(pr.y);
    if (i0 >= HD - RD) {
        const int j = (i0 - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float r0 = __fmaf_rn(v0, cs, -__fmul_rn(v1, sn));
        const float r1 = __fmaf_rn(v1, cs, __fmul_rn(v0, sn));
        v0 = __bfloat162float(__float2bfloat16_rn(r0));
        v1 = __bfloat162float(__float2bfloat16_rn(r1));
    }
    unsigned byte;
    if (BITS) {
        const unsigned amax = __reduce_max_sync(mask, max(tf_ds_abs_bits(v0), tf_ds_abs_bits(v1)));
        byte = tf_ds_scale_byte_bits(amax);
    } else {
        const float amax = tf_ds_fp4_amax<32>(fmaxf(v0, v1));
        byte = tf_ds_pow2_exponent(amax);
    }
    const float scale = __uint_as_float(byte << 23);
    const float recip = tf_ds_recip_bits(byte);
    const float t0 = tf_ds_clamp6(__fmul_rn(v0, recip)), t1 = tf_ds_clamp6(__fmul_rn(v1, recip));
    const float q0 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t0) & 7u), t0);
    const float q1 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t1) & 7u), t1);
    const __nv_bfloat162 o = {__float2bfloat16_rn(__fmul_rn(q0, scale)),
                              __float2bfloat16_rn(__fmul_rn(q1, scale))};
    *(uint32_t*)(out + base + i0) = *(const uint32_t*)&o;
    (void)lane2;
}

template<int WPB, bool BITS>
__device__ __forceinline__ void qd_pair_heads(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                              const float* sin, const long long* pos, int H, int HD, int RD) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.x, hg = blockIdx.y;
    const int h = hg * WPB + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    for (int chunk = 0; chunk < HD; chunk += 64) qd_pair_chunk<BITS>(x, out, cos, sin, base, p, chunk, lane, HD, RD);
}

extern "C" __global__ void __launch_bounds__(32) v_q_p1(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                        const float* cos, const float* sin, const long long* pos,
                                                        int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<1, true>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(64) v_q_p2(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                        const float* cos, const float* sin, const long long* pos,
                                                        int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<2, true>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(128) v_q_p4(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                         const float* cos, const float* sin, const long long* pos,
                                                         int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<4, true>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                         const float* cos, const float* sin, const long long* pos,
                                                         int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<8, true>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(128) v_q_p4f(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<4, false>(x, out, cos, sin, pos, H, HD, RD);
}

// ---- item 3: the CUDA FP4 conversion intrinsics, after tf_ds_clamp6 -------------------------------------

// |value| of an E2M1 code through the intrinsic (the sign comes from t, as the engine's copysignf does).
__device__ __forceinline__ float tf_ds_mag_cvt(unsigned code) {
    return fabsf(__half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)code, __NV_E2M1)));
}

__device__ __forceinline__ unsigned tf_ds_code_cvt(float t) {
    return (unsigned)__nv_cvt_float_to_fp4(t, __NV_E2M1, cudaRoundNearest);
}

// The pair-in-lane chunk through the paired intrinsic: one 2-way conversion in, one 2-way back out.
__device__ __forceinline__ void qd_pair_chunk_cvt(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                                  const float* sin, long long base, long long p, int chunk, int lane,
                                                  int HD, int RD) {
    const unsigned mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
    const int i0 = chunk + 2 * lane;
    const uint32_t raw = *(const uint32_t*)(x + base + i0);
    const __nv_bfloat162 pr = *(__nv_bfloat162*)&raw;
    float v0 = __bfloat162float(pr.x), v1 = __bfloat162float(pr.y);
    if (i0 >= HD - RD) {
        const int j = (i0 - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float r0 = __fmaf_rn(v0, cs, -__fmul_rn(v1, sn));
        const float r1 = __fmaf_rn(v1, cs, __fmul_rn(v0, sn));
        v0 = __bfloat162float(__float2bfloat16_rn(r0));
        v1 = __bfloat162float(__float2bfloat16_rn(r1));
    }
    const unsigned amax = __reduce_max_sync(mask, max(tf_ds_abs_bits(v0), tf_ds_abs_bits(v1)));
    const unsigned byte = tf_ds_scale_byte_bits(amax);
    const float scale = __uint_as_float(byte << 23);
    const float recip = tf_ds_recip_bits(byte);
    const float t0 = tf_ds_clamp6(__fmul_rn(v0, recip)), t1 = tf_ds_clamp6(__fmul_rn(v1, recip));
    const __nv_fp4x2_storage_t c = __nv_cvt_float2_to_fp4x2(make_float2(t0, t1), __NV_E2M1, cudaRoundNearest);
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(c, __NV_E2M1);
    const float m0 = fabsf(__half2float(*((const __half*)&h.x)));
    const float m1 = fabsf(__half2float(*((const __half*)&h.y)));
    const __nv_bfloat162 o = {__float2bfloat16_rn(__fmul_rn(copysignf(m0, t0), scale)),
                              __float2bfloat16_rn(__fmul_rn(copysignf(m1, t1), scale))};
    *(uint32_t*)(out + base + i0) = *(const uint32_t*)&o;
}

template<int WPB>
__device__ __forceinline__ void qd_pair_heads_cvt(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                                  const float* sin, const long long* pos, int H, int HD, int RD) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.x, hg = blockIdx.y;
    const int h = hg * WPB + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    for (int chunk = 0; chunk < HD; chunk += 64)
        qd_pair_chunk_cvt(x, out, cos, sin, base, p, chunk, lane, HD, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8i(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads_cvt<8>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(128) v_q_p4i(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads_cvt<4>(x, out, cos, sin, pos, H, HD, RD);
}

// The one-element-a-thread form through the intrinsics (the low-row geometry).
__device__ __forceinline__ void qd_element_cvt(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                               const float* sin, long long base, long long p, int i, int HD, int RD) {
    float v = __bfloat162float(x[base + i]);
    const float other = __shfl_xor_sync(0xffffffffu, v, 1);
    if (i >= HD - RD) {
        const int j = (i - (HD - RD)) / 2;
        const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
        const float rot = (i & 1) ? __fmaf_rn(other, sn, __fmul_rn(v, cs))
                                  : __fmaf_rn(v, cs, -__fmul_rn(other, sn));
        v = __bfloat162float(__float2bfloat16_rn(rot));
    }
    const unsigned amax = __reduce_max_sync(0xffffffffu, tf_ds_abs_bits(v));
    const unsigned byte = tf_ds_scale_byte_bits(amax);
    const float scale = __uint_as_float(byte << 23);
    const float t = tf_ds_clamp6(__fmul_rn(v, tf_ds_recip_bits(byte)));
    const float q = copysignf(tf_ds_mag_cvt(tf_ds_code_cvt(t)), t);
    out[base + i] = __float2bfloat16_rn(__fmul_rn(q, scale));
}

extern "C" __global__ void __launch_bounds__(128) v_q_bitsi(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                            const float* cos, const float* sin, const long long* pos,
                                                            int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x, h = blockIdx.y;
    qd_element_cvt(x, out, cos, sin, ((long long)r * H + h) * HD, pos[r], threadIdx.x, HD, RD);
}

// The pair form with the engine's current scale arithmetic (float 16-lane amax, /6, per-element division).
__device__ __forceinline__ void qd_pair_chunk_float(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
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
        const float r1 = __fmaf_rn(v1, cs, __fmul_rn(v0, sn));
        v0 = __bfloat162float(__float2bfloat16_rn(r0));
        v1 = __bfloat162float(__float2bfloat16_rn(r1));
    }
    const float amax = tf_ds_fp4_amax<16>(fmaxf(fabsf(v0), fabsf(v1)));
    const unsigned byte = tf_ds_pow2_exponent(amax);
    const float scale = __uint_as_float(byte << 23);
    const float t0 = tf_ds_clamp6(__fdiv_rn(v0, scale)), t1 = tf_ds_clamp6(__fdiv_rn(v1, scale));
    const float q0 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t0) & 7u), t0);
    const float q1 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t1) & 7u), t1);
    const __nv_bfloat162 o = {__float2bfloat16_rn(__fmul_rn(q0, scale)),
                              __float2bfloat16_rn(__fmul_rn(q1, scale))};
    *(uint32_t*)(out + base + i0) = *(const uint32_t*)&o;
}

template<int WPB>
__device__ __forceinline__ void qd_pair_heads_float(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                                    const float* sin, const long long* pos, int H, int HD, int RD) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.x, hg = blockIdx.y;
    const int h = hg * WPB + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    for (int chunk = 0; chunk < HD; chunk += 64)
        qd_pair_chunk_float(x, out, cos, sin, base, p, chunk, lane, HD, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8f(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads_float<8>(x, out, cos, sin, pos, H, HD, RD);
}

// Two elements a lane, 1 warp a block: the smallest pair geometry (grid = (rows, heads * 2)).
template<bool BITS>
__device__ __forceinline__ void qd_pair32(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                          const float* sin, const long long* pos, int H, int HD, int RD) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x, seg = blockIdx.y;          // seg = head * (HD / 64) + chunk
    const int h = seg / (HD / 64), chunk = (seg % (HD / 64)) * 64;
    const long long base = ((long long)r * H + h) * HD;
    const long long p = pos[r];
    if (BITS) qd_pair_chunk<true>(x, out, cos, sin, base, p, chunk, lane, HD, RD);
    else qd_pair_chunk_float(x, out, cos, sin, base, p, chunk, lane, HD, RD);
}

extern "C" __global__ void __launch_bounds__(32) v_q_s1(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                        const float* cos, const float* sin, const long long* pos,
                                                        int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair32<true>(x, out, cos, sin, pos, H, HD, RD);
}

// ---- HD = 128 (the served index head), both chunks in flight -------------------------------------------

// One chunk's pair arithmetic from already-loaded values (the loads are hoisted into the caller).
template<bool BITS>
__device__ __forceinline__ void qd_pair_apply(float v0, float v1, bool rot, const float* cs, const float* sn,
                                              unsigned mask, __nv_bfloat16* dst) {
    if (rot) {
        const float r0 = __fmaf_rn(v0, *cs, -__fmul_rn(v1, *sn));
        const float r1 = __fmaf_rn(v1, *cs, __fmul_rn(v0, *sn));
        v0 = __bfloat162float(__float2bfloat16_rn(r0));
        v1 = __bfloat162float(__float2bfloat16_rn(r1));
    }
    unsigned byte;
    if (BITS) {
        const unsigned amax = __reduce_max_sync(mask, max(tf_ds_abs_bits(v0), tf_ds_abs_bits(v1)));
        byte = tf_ds_scale_byte_bits(amax);
    } else {
        const float amax = tf_ds_fp4_amax<16>(fmaxf(fabsf(v0), fabsf(v1)));
        byte = tf_ds_pow2_exponent(amax);
    }
    const float scale = __uint_as_float(byte << 23);
    const float recip = BITS ? tf_ds_recip_bits(byte) : scale;
    const float t0 = tf_ds_clamp6(BITS ? __fmul_rn(v0, recip) : __fdiv_rn(v0, scale));
    const float t1 = tf_ds_clamp6(BITS ? __fmul_rn(v1, recip) : __fdiv_rn(v1, scale));
    const float q0 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t0) & 7u), t0);
    const float q1 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t1) & 7u), t1);
    const __nv_bfloat162 o = {__float2bfloat16_rn(__fmul_rn(q0, scale)),
                              __float2bfloat16_rn(__fmul_rn(q1, scale))};
    *(uint32_t*)dst = *(const uint32_t*)&o;
}

// HD = 128: two 64-element chunks, both loaded before either is quantized (the second chunk rotates).
template<int WPB, bool BITS>
__device__ __forceinline__ void qd_pair_128(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                            const float* sin, const long long* pos, int H, int RD) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.x, hg = blockIdx.y;
    const int h = hg * WPB + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * 128;
    const long long p = pos[r];
    const unsigned mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
    const int i0 = 2 * lane;                      // chunk 0 (unrotated when RD <= 64)
    const uint32_t raw0 = *(const uint32_t*)(x + base + i0);
    const uint32_t raw1 = *(const uint32_t*)(x + base + 64 + i0);
    const bool rot0 = i0 >= 128 - RD, rot1 = 64 + i0 >= 128 - RD;
    const float cs0 = rot0 ? cos[p * (RD / 2) + (i0 - (128 - RD)) / 2] : 0.f;
    const float sn0 = rot0 ? sin[p * (RD / 2) + (i0 - (128 - RD)) / 2] : 0.f;
    const float cs1 = rot1 ? cos[p * (RD / 2) + (64 + i0 - (128 - RD)) / 2] : 0.f;
    const float sn1 = rot1 ? sin[p * (RD / 2) + (64 + i0 - (128 - RD)) / 2] : 0.f;
    const __nv_bfloat162 a = *(__nv_bfloat162*)&raw0, b = *(__nv_bfloat162*)&raw1;
    qd_pair_apply<BITS>(__bfloat162float(a.x), __bfloat162float(a.y), rot0, &cs0, &sn0, mask, out + base + i0);
    qd_pair_apply<BITS>(__bfloat162float(b.x), __bfloat162float(b.y), rot1, &cs1, &sn1, mask, out + base + 64 + i0);
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8x(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_128<8, true>(x, out, cos, sin, pos, H, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8xf(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                           const float* cos, const float* sin, const long long* pos,
                                                           int H, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_128<8, false>(x, out, cos, sin, pos, H, RD);
}

extern "C" __global__ void __launch_bounds__(128) v_q_p4x(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_128<4, true>(x, out, cos, sin, pos, H, RD);
}

// HD = 128, two chunks loaded up front, the arithmetic inline (no pointers to locals, no ternaries on the loads).
template<int WPB, bool BITS>
__device__ __forceinline__ void qd_pair_128y(const __nv_bfloat16* x, __nv_bfloat16* out, const float* cos,
                                             const float* sin, const long long* pos, int H, int RD) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = blockIdx.x, hg = blockIdx.y;
    const int h = hg * WPB + warp;
    if (h >= H) return;
    const long long base = ((long long)r * H + h) * 128;
    const long long p = pos[r];
    const unsigned mask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
    const int i0 = 2 * lane;
    const uint32_t raw0 = *(const uint32_t*)(x + base + i0);
    const uint32_t raw1 = *(const uint32_t*)(x + base + 64 + i0);
    float v0[4] = {__bfloat162float((*(const __nv_bfloat162*)&raw0).x),
                   __bfloat162float((*(const __nv_bfloat162*)&raw0).y),
                   __bfloat162float((*(const __nv_bfloat162*)&raw1).x),
                   __bfloat162float((*(const __nv_bfloat162*)&raw1).y)};
    const int start = 128 - RD;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = 64 * k + i0;
        if (i >= start) {
            const int j = (i - start) / 2;
            const float cs = cos[p * (RD / 2) + j], sn = sin[p * (RD / 2) + j];
            const float r0 = __fmaf_rn(v0[2 * k], cs, -__fmul_rn(v0[2 * k + 1], sn));
            const float r1 = __fmaf_rn(v0[2 * k + 1], cs, __fmul_rn(v0[2 * k], sn));
            v0[2 * k] = __bfloat162float(__float2bfloat16_rn(r0));
            v0[2 * k + 1] = __bfloat162float(__float2bfloat16_rn(r1));
        }
        unsigned byte;
        if (BITS) {
            const unsigned amax = __reduce_max_sync(mask, max(tf_ds_abs_bits(v0[2 * k]), tf_ds_abs_bits(v0[2 * k + 1])));
            byte = tf_ds_scale_byte_bits(amax);
        } else {
            const float amax = tf_ds_fp4_amax<16>(fmaxf(fabsf(v0[2 * k]), fabsf(v0[2 * k + 1])));
            byte = tf_ds_pow2_exponent(amax);
        }
        const float scale = __uint_as_float(byte << 23);
        const float t0 = tf_ds_clamp6(BITS ? __fmul_rn(v0[2 * k], tf_ds_recip_bits(byte)) : __fdiv_rn(v0[2 * k], scale));
        const float t1 = tf_ds_clamp6(BITS ? __fmul_rn(v0[2 * k + 1], tf_ds_recip_bits(byte)) : __fdiv_rn(v0[2 * k + 1], scale));
        const float q0 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t0) & 7u), t0);
        const float q1 = copysignf(tf_ds_e2m1_mag(tf_ds_e2m1_code(t1) & 7u), t1);
        const __nv_bfloat162 o = {__float2bfloat16_rn(__fmul_rn(q0, scale)),
                                  __float2bfloat16_rn(__fmul_rn(q1, scale))};
        *(uint32_t*)(out + base + 64 * k + i0) = *(const uint32_t*)&o;
    }
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8y(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_128y<8, true>(x, out, cos, sin, pos, H, RD);
}

extern "C" __global__ void __launch_bounds__(256) v_q_p8yf(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                           const float* cos, const float* sin, const long long* pos,
                                                           int H, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_128y<8, false>(x, out, cos, sin, pos, H, RD);
}

extern "C" __global__ void __launch_bounds__(512) v_q_p16(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                          const float* cos, const float* sin, const long long* pos,
                                                          int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<16, true>(x, out, cos, sin, pos, H, HD, RD);
}

extern "C" __global__ void __launch_bounds__(64) v_q_p2b(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                         const float* cos, const float* sin, const long long* pos,
                                                         int H, int HD, int RD, int pdl) {
    pdl_enter(pdl);
    qd_pair_heads<2, true>(x, out, cos, sin, pos, H, HD, RD);
}

// ---- item 3: exhaustive parity of the FP4 conversion intrinsics against the engine's E2M1 chain ---------

// One thread per (BF16 input, reachable E8M0 scale) pair: the engine's code/dequant against the intrinsic's, with the
// same pre-clamp (tf_ds_clamp6 turns a NaN into -6, which the intrinsic then maps to -MAXNORM as the kit does).
extern "C" __global__ void v_fp4_probe(const unsigned short* table, int n, unsigned* diff) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)n * 255) return;
    const int vi = (int)(t / 255);
    const unsigned byte = 1u + (unsigned)(t % 255);      // 1..255 (255 is the infinite scale)
    const __nv_bfloat16 x = *(__nv_bfloat16*)&table[vi];
    const float v = __bfloat162float(x);
    const float scale = __uint_as_float(byte << 23);
    const float cl = tf_ds_clamp6(__fmul_rn(v, tf_ds_recip_bits(byte)));
    const unsigned c_cur = tf_ds_e2m1_code(cl);
    // the kit drops the sign bit when the magnitude rounds to zero; the intrinsic keeps it, so mask it
    unsigned c_int = (unsigned)__nv_cvt_float_to_fp4(cl, __NV_E2M1, cudaRoundNearest);
    c_int &= (c_int & 7u) ? 0xfu : 0x7u;
    const float q_cur = copysignf(tf_ds_e2m1_mag(c_cur & 7u), cl);
    const float m_int = fabsf(__half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)c_int, __NV_E2M1)));
    const float q_int = copysignf(m_int, cl);
    const __nv_bfloat16 o_cur = __float2bfloat16_rn(__fmul_rn(q_cur, scale));
    const __nv_bfloat16 o_int = __float2bfloat16_rn(__fmul_rn(q_int, scale));
    const unsigned bc = *(const unsigned short*)&o_cur, bi = *(const unsigned short*)&o_int;
    if (c_cur != c_int || bc != bi) atomicAdd(diff, 1u);
}

// The query body through the paired intrinsic, for the timing arm (the same chain as the engine's, one call each way).
extern "C" __global__ void __launch_bounds__(128) v_q_bitsi2(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                             const float* cos, const float* sin, const long long* pos,
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
        const unsigned byte = tf_ds_scale_byte_bits(__reduce_max_sync(0xffffffffu, tf_ds_abs_bits(v)));
        const float scale = __uint_as_float(byte << 23);
        const float t = tf_ds_clamp6(__fmul_rn(v, tf_ds_recip_bits(byte)));
        const float q = copysignf(tf_ds_mag_cvt(tf_ds_code_cvt(t)), t);
        out[base + i] = __float2bfloat16_rn(__fmul_rn(q, scale));
    }
}

// The codes both paths give for every BF16 input at one fixed scale byte (the probe's debugging view).
extern "C" __global__ void v_fp4_dump(const unsigned short* table, int n, unsigned byte, unsigned* cur, unsigned* intr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const __nv_bfloat16 x = *(__nv_bfloat16*)&table[i];
    const float cl = tf_ds_clamp6(__fmul_rn(__bfloat162float(x), tf_ds_recip_bits(byte)));
    cur[i] = tf_ds_e2m1_code(cl);
    const unsigned c = (unsigned)__nv_cvt_float_to_fp4(cl, __NV_E2M1, cudaRoundNearest);
    intr[i] = c & ((c & 7u) ? 0xfu : 0x7u);
}

// The differing (input, byte) pairs of the probe, for inspection.
extern "C" __global__ void v_fp4_diff(const unsigned short* table, int n, unsigned* pairs, unsigned* count) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)n * 255) return;
    const int vi = (int)(t / 255);
    const unsigned byte = 1u + (unsigned)(t % 255);
    const __nv_bfloat16 x = *(__nv_bfloat16*)&table[vi];
    const float cl = tf_ds_clamp6(__fmul_rn(__bfloat162float(x), tf_ds_recip_bits(byte)));
    const unsigned c_cur = tf_ds_e2m1_code(cl);
    unsigned c_int = (unsigned)__nv_cvt_float_to_fp4(cl, __NV_E2M1, cudaRoundNearest);
    c_int &= (c_int & 7u) ? 0xfu : 0x7u;
    if (c_cur != c_int) {
        const unsigned k = atomicAdd(count, 1u);
        if (k < 128u) pairs[k] = ((unsigned)vi << 8) | byte;
    }
}

// The (input, byte) pairs whose dequantized output byte differs, with both bytes.
extern "C" __global__ void v_fp4_diff2(const unsigned short* table, int n, unsigned* pairs, unsigned* count) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)n * 255) return;
    const int vi = (int)(t / 255);
    const unsigned byte = 1u + (unsigned)(t % 255);
    const __nv_bfloat16 x = *(__nv_bfloat16*)&table[vi];
    const float cl = tf_ds_clamp6(__fmul_rn(__bfloat162float(x), tf_ds_recip_bits(byte)));
    const float scale = __uint_as_float(byte << 23);
    const unsigned c_cur = tf_ds_e2m1_code(cl);
    unsigned c_int = (unsigned)__nv_cvt_float_to_fp4(cl, __NV_E2M1, cudaRoundNearest);
    c_int &= (c_int & 7u) ? 0xfu : 0x7u;
    const float q_cur = copysignf(tf_ds_e2m1_mag(c_cur & 7u), cl);
    const float q_int = copysignf(fabsf(__half2float(__nv_cvt_fp4_to_halfraw((__nv_fp4_storage_t)c_int, __NV_E2M1))), cl);
    const __nv_bfloat16 o_cur = __float2bfloat16_rn(__fmul_rn(q_cur, scale));
    const __nv_bfloat16 o_int = __float2bfloat16_rn(__fmul_rn(q_int, scale));
    if (*(const unsigned short*)&o_cur != *(const unsigned short*)&o_int) {
        const unsigned k = atomicAdd(count, 1u);
        if (k < 128u) pairs[k] = ((unsigned)vi << 24) | (byte << 16) | (*(const unsigned short*)&o_cur << 8) | 0u;
        if (k < 128u) pairs[k] |= (unsigned)(*(const unsigned short*)&o_int) << 8;
    }
}
