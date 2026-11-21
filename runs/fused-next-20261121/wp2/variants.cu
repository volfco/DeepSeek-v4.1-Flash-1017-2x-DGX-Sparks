// WP2 candidate forms of tf_ds_compress2x_norm_kernel (0197 decode compressor). Every candidate reproduces the
// engine kernel's arithmetic bit for bit; only the thread geometry / partner resolution / pair evaluation change.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ void pdl_enter(int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

__device__ __forceinline__ float v_pair_ref(float s0, float s1, float kv0, float kv1) {
    float m = -3.402823466e+38f;
    m = m < s0 ? s0 : m;
    m = m < s1 ? s1 : m;
    float sum = 0.f;
    sum = __fadd_rn(sum, expf(__fsub_rn(s0, m)));
    sum = __fadd_rn(sum, expf(__fsub_rn(s1, m)));
    const float p0 = __fdiv_rn(expf(__fsub_rn(s0, m)), sum);
    const float p1 = __fdiv_rn(expf(__fsub_rn(s1, m)), sum);
    const float y0 = __fmul_rn(kv0, p0);
    const float y1 = __fmul_rn(kv1, p1);
    float t = __fadd_rn(__fadd_rn(0.f, y0), __fadd_rn(0.f, y1));
    t = __fadd_rn(t, 0.f);
    t = __fadd_rn(t, 0.f);
    return t;
}

// One expf: the pair's max argument gives expf(0) = 1 exactly, so only the other side is evaluated. Branchless
// selects, no isfinite test.
__device__ __forceinline__ float v_pair_1e(float s0, float s1, float kv0, float kv1) {
    float m = -3.402823466e+38f;
    m = m < s0 ? s0 : m;
    m = m < s1 ? s1 : m;
    const float t0 = __fsub_rn(s0, m), t1 = __fsub_rn(s1, m);
    const bool z0 = t0 == 0.f, z1 = t1 == 0.f;
    const float e = expf(z0 ? t1 : t0);
    const float e0 = z0 ? 1.f : e;
    const float e1 = z1 ? 1.f : e;
    float sum = 0.f;
    sum = __fadd_rn(sum, e0);
    sum = __fadd_rn(sum, e1);
    const float p0 = __fdiv_rn(e0, sum);
    const float p1 = __fdiv_rn(e1, sum);
    const float y0 = __fmul_rn(kv0, p0);
    const float y1 = __fmul_rn(kv1, p1);
    float t = __fadd_rn(__fadd_rn(0.f, y0), __fadd_rn(0.f, y1));
    t = __fadd_rn(t, 0.f);
    t = __fadd_rn(t, 0.f);
    return t;
}

__device__ __forceinline__ float v_kit_squares4(float a, float b, float c, float d) {
    float sum = __fmul_rn(b, b);
    sum = __fmaf_rn(a, a, sum);
    sum = __fmaf_rn(c, c, sum);
    return __fmaf_rn(d, d, sum);
}

__device__ __forceinline__ float v_kit_warp_sum(float sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, offset));
    return sum;
}

template<int D>
__device__ __forceinline__ float v_kit_inverse_rms(float sum, float eps) {
    const float mean_eps = __fmaf_rn(sum, 1.0f / D, eps);
    float root, inv;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(mean_eps));
    asm("div.approx.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(root));
    return inv;
}

__device__ __forceinline__ int2 v_partners(const long long* slots, int rows,
                                           long long own, long long q0, long long q1) {
    if (rows == 1) return make_int2(own == q0 ? 0 : -1, own == q1 ? 0 : -1);
    int2 found = make_int2(-1, -1);
    for (int first = 0; first < rows; first += 32) {
        const int r = first + (threadIdx.x & 31);
        const long long slot = r < rows ? slots[r] : -1;
        const unsigned m0 = __ballot_sync(0xffffffffu, r < rows && slot == q0);
        const unsigned m1 = __ballot_sync(0xffffffffu, r < rows && slot == q1);
        if (found.x < 0 && m0) found.x = first + __ffs(m0) - 1;
        if (found.y < 0 && m1) found.y = first + __ffs(m1) - 1;
        if (found.x >= 0 && found.y >= 0) break;
    }
    return found;
}

// The engine's current decode kernel, verbatim (the baseline arm inside this module).
template<int D>
__device__ __forceinline__ void v_cur_decode(float value, const __nv_bfloat16* w, __nv_bfloat16* dst, float eps) {
    const int tid = threadIdx.x;
    const float x = __bfloat162float(__float2bfloat16_rn(value));
    __shared__ float row[D == 512 ? 512 : 1];
    if (D == 512) {
        row[tid] = x;
        __syncthreads();
    }
    __shared__ float warp_sum[4];
    if (tid < 128) {
        float sum;
        if (D == 512) {
            const int base = tid * 4;
            sum = v_kit_squares4(row[base], row[base + 1], row[base + 2], row[base + 3]);
        } else {
            sum = __fmul_rn(x, x);
        }
        sum = v_kit_warp_sum(sum);
        if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    }
    __syncthreads();
    const float sum = __fadd_rn(__fadd_rn(warp_sum[0], warp_sum[2]), __fadd_rn(warp_sum[1], warp_sum[3]));
    const float inv = v_kit_inverse_rms<D>(sum, eps);
    const float y = __fmul_rn(x, inv);
    dst[tid] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[tid]), y));
}

extern "C" __global__ void __launch_bounds__(512) v_cur(const float* ckv, long long ckv_ld,
                                                        const float* cgate, long long cgate_ld,
                                                        const long long* rslot, float* ring_kv, float* ring_score,
                                                        const long long* gi, long long src_rows,
                                                        const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                        int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    const int g = blockIdx.x;
    const long long own = rslot[g];
    const int c = threadIdx.x;
    const float own_kv = ckv[g * ckv_ld + c], own_score = cgate[g * cgate_ld + c];
    ring_kv[own * D + c] = own_kv;
    ring_score[own * D + c] = own_score;
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const int2 pr = v_partners(rslot, G, own, q0, q1);
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if ((!ok0 || !ok1) && threadIdx.x == 0 && invalid) atomicExch(invalid, 1u);
    float kv[2], score[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool ok = i == 0 ? ok0 : ok1;
        const int row = i == 0 ? pr.x : pr.y;
        const long long q = i == 0 ? q0 : q1;
        if (!ok) {
            kv[i] = score[i] = 0.f;
        } else if (row == g) {
            kv[i] = own_kv;
            score[i] = own_score;
        } else if (row >= 0) {
            kv[i] = ckv[row * ckv_ld + c];
            score[i] = cgate[row * cgate_ld + c];
        } else {
            kv[i] = ring_kv[q * D + c];
            score[i] = ring_score[q * D + c];
        }
    }
    const float value = v_pair_ref(score[0], score[1], kv[0], kv[1]);
    if (D == 512) v_cur_decode<512>(value, w, out + g * ldo, eps);
    else v_cur_decode<128>(value, w, out + g * ldo, eps);
}

// -------------------------------------------------------------------------------------------------------------------
// Candidate: 128 threads, ITEMS channels a thread (4 at D = 512, 1 at D = 128). No shared row exchange, no barrier
// before the sum: each thread holds its own contiguous channels, which is the kit's squares4 order.
//   RES_BLOCK: resolve both partners once, by warp 0, into shared memory (one barrier) instead of in every warp.
//   VEC: 16-byte global loads / 8-byte bf16x4 stores when the strides allow it, scalar otherwise.
//   PAIR: 0 = the engine's pair arithmetic, 1 = the branchless one-exp form.
template<int D, int ITEMS, bool RES_BLOCK, bool VEC, int PAIR>
__device__ __forceinline__ void v_row(const float* ckv, long long ckv_ld, const float* cgate, long long cgate_ld,
                                      const long long* rslot, float* ring_kv, float* ring_score,
                                      const long long* gi, long long src_rows, const __nv_bfloat16* w,
                                      __nv_bfloat16* out, long long ldo, int G, float eps,
                                      unsigned* invalid) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = tid * ITEMS;
    const long long own = rslot[g];
    __shared__ int2 sh_pair;
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    if (RES_BLOCK) {
        if (tid < 32) sh_pair = v_partners(rslot, G, own, q0, q1);
    }
    constexpr bool V4 = VEC && (ITEMS == 4);
    const bool va = V4 && ((ckv_ld & 3) == 0) && ((cgate_ld & 3) == 0);
    float own_kv[ITEMS], own_score[ITEMS];
    if (va) {
        const float4 a = *(const float4*)(ckv + g * ckv_ld + base);
        const float4 b = *(const float4*)(cgate + g * cgate_ld + base);
        own_kv[0] = a.x; own_kv[1] = a.y; own_kv[2] = a.z; own_kv[3] = a.w;
        own_score[0] = b.x; own_score[1] = b.y; own_score[2] = b.z; own_score[3] = b.w;
        *(float4*)(ring_kv + own * D + base) = a;
        *(float4*)(ring_score + own * D + base) = b;
    } else {
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            own_kv[j] = ckv[g * ckv_ld + base + j];
            own_score[j] = cgate[g * cgate_ld + base + j];
            ring_kv[own * D + base + j] = own_kv[j];
            ring_score[own * D + base + j] = own_score[j];
        }
    }
    int2 pr;
    if (RES_BLOCK) {
        __syncthreads();
        pr = sh_pair;
    } else {
        pr = v_partners(rslot, G, own, q0, q1);
    }
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if ((!ok0 || !ok1) && tid == 0 && invalid) atomicExch(invalid, 1u);
    float kv[2][ITEMS], score[2][ITEMS];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool ok = i == 0 ? ok0 : ok1;
        const int row = i == 0 ? pr.x : pr.y;
        const long long q = i == 0 ? q0 : q1;
        if (!ok) {
            #pragma unroll
            for (int j = 0; j < ITEMS; ++j) kv[i][j] = score[i][j] = 0.f;
        } else if (row == g) {
            #pragma unroll
            for (int j = 0; j < ITEMS; ++j) { kv[i][j] = own_kv[j]; score[i][j] = own_score[j]; }
        } else if (row >= 0) {
            if (va) {
                const float4 a = *(const float4*)(ckv + row * ckv_ld + base);
                const float4 b = *(const float4*)(cgate + row * cgate_ld + base);
                kv[i][0] = a.x; kv[i][1] = a.y; kv[i][2] = a.z; kv[i][3] = a.w;
                score[i][0] = b.x; score[i][1] = b.y; score[i][2] = b.z; score[i][3] = b.w;
            } else {
                #pragma unroll
                for (int j = 0; j < ITEMS; ++j) {
                    kv[i][j] = ckv[row * ckv_ld + base + j];
                    score[i][j] = cgate[row * cgate_ld + base + j];
                }
            }
        } else {
            if (V4) {
                const float4 a = *(const float4*)(ring_kv + q * D + base);
                const float4 b = *(const float4*)(ring_score + q * D + base);
                kv[i][0] = a.x; kv[i][1] = a.y; kv[i][2] = a.z; kv[i][3] = a.w;
                score[i][0] = b.x; score[i][1] = b.y; score[i][2] = b.z; score[i][3] = b.w;
            } else {
                #pragma unroll
                for (int j = 0; j < ITEMS; ++j) {
                    kv[i][j] = ring_kv[q * D + base + j];
                    score[i][j] = ring_score[q * D + base + j];
                }
            }
        }
    }
    float x[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const float value = PAIR == 0 ? v_pair_ref(score[0][j], score[1][j], kv[0][j], kv[1][j])
                          : PAIR == 1 ? v_pair_1e(score[0][j], score[1][j], kv[0][j], kv[1][j])
                                      : kv[0][j];
        x[j] = __bfloat162float(__float2bfloat16_rn(value));
    }
    __shared__ float warp_sum[4];
    float sum = ITEMS == 4 ? v_kit_squares4(x[0], x[1], x[2], x[3]) : __fmul_rn(x[0], x[0]);
    sum = v_kit_warp_sum(sum);
    if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(warp_sum[0], warp_sum[2]), __fadd_rn(warp_sum[1], warp_sum[3]));
    const float inv = v_kit_inverse_rms<D>(total, eps);
    __nv_bfloat16* dst = out + g * ldo;
    if (V4 && ((ldo & 3) == 0)) {
        __nv_bfloat16 v4[ITEMS];
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const float y = __fmul_rn(x[j], inv);
            v4[j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[base + j]), y));
        }
        const __nv_bfloat162 a = __halves2bfloat162(v4[0], v4[1]);
        const __nv_bfloat162 b = __halves2bfloat162(v4[2], v4[3]);
        *(__nv_bfloat162*)(dst + base) = a;
        *(__nv_bfloat162*)(dst + base + 2) = b;
    } else {
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const float y = __fmul_rn(x[j], inv);
            dst[base + j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[base + j]), y));
        }
    }
}

#define V_LAUNCH(name, RES, VEC, PAIR)                                                                          \
    extern "C" __global__ void __launch_bounds__(128) name(const float* ckv, long long ckv_ld,                          \
        const float* cgate, long long cgate_ld, const long long* rslot, float* ring_kv, float* ring_score,               \
        const long long* gi, long long src_rows, const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,              \
        int G, int D, float eps, unsigned* invalid, int pdl) {                                                          \
        pdl_enter(pdl);                                                                                                 \
        if (D == 512) v_row<512, 4, RES, VEC, PAIR>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi,      \
                                                    src_rows, w, out, ldo, G, eps, invalid);                           \
        else v_row<128, 1, RES, VEC, PAIR>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi,               \
                                          src_rows, w, out, ldo, G, eps, invalid);                                     \
    }

// D=512: 4 channels a thread. D=128: 1 channel a thread (the engine's own 128-lane form).
V_LAUNCH(v_128x4, false, false, 0)
V_LAUNCH(v_128x4v, false, true, 0)
V_LAUNCH(v_128x4_r1, true, false, 0)
V_LAUNCH(v_128x4v_r1, true, true, 0)
V_LAUNCH(v_128x4v_1e, false, true, 1)
V_LAUNCH(v_128x4v_r1_1e, true, true, 1)

// Item 2's exhaustive pair check: every (s0, s1, kv0, kv1) combination from a value table, ref vs one-exp, compared
// after the bf16 rounding the kernel stores.
extern "C" __global__ void v_pair_probe(const float* vals, int n, unsigned* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n * n * n;
    if (i >= total) return;
    const int a = i / (n * n * n), b = (i / (n * n)) % n, c = (i / n) % n, d = i % n;
    const float s0 = vals[a], s1 = vals[b], kv0 = vals[c], kv1 = vals[d];
    const unsigned r = __bfloat16_as_ushort(__float2bfloat16_rn(v_pair_ref(s0, s1, kv0, kv1)));
    const unsigned o = __bfloat16_as_ushort(__float2bfloat16_rn(v_pair_1e(s0, s1, kv0, kv1)));
    out[i] = r != o;
}

// The engine geometry (512 threads, one channel a thread, shared row exchange) with the item-2/item-3 policies.
template<bool RES_BLOCK, int PAIR>
__device__ __forceinline__ void v_cur_row(const float* ckv, long long ckv_ld, const float* cgate, long long cgate_ld,
                                          const long long* rslot, float* ring_kv, float* ring_score,
                                          const long long* gi, long long src_rows, const __nv_bfloat16* w,
                                          __nv_bfloat16* out, long long ldo, int G, int D, float eps,
                                          unsigned* invalid) {
    const int g = blockIdx.x;
    const long long own = rslot[g];
    const int c = threadIdx.x;
    const float own_kv = ckv[g * ckv_ld + c], own_score = cgate[g * cgate_ld + c];
    ring_kv[own * D + c] = own_kv;
    ring_score[own * D + c] = own_score;
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    __shared__ int2 sh_pair;
    int2 pr;
    if (RES_BLOCK) {
        if (threadIdx.x < 32) sh_pair = v_partners(rslot, G, own, q0, q1);
        __syncthreads();
        pr = sh_pair;
    } else {
        pr = v_partners(rslot, G, own, q0, q1);
    }
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if ((!ok0 || !ok1) && threadIdx.x == 0 && invalid) atomicExch(invalid, 1u);
    float kv[2], score[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool ok = i == 0 ? ok0 : ok1;
        const int row = i == 0 ? pr.x : pr.y;
        const long long q = i == 0 ? q0 : q1;
        if (!ok) {
            kv[i] = score[i] = 0.f;
        } else if (row == g) {
            kv[i] = own_kv;
            score[i] = own_score;
        } else if (row >= 0) {
            kv[i] = ckv[row * ckv_ld + c];
            score[i] = cgate[row * cgate_ld + c];
        } else {
            kv[i] = ring_kv[q * D + c];
            score[i] = ring_score[q * D + c];
        }
    }
    const float value = PAIR == 0 ? v_pair_ref(score[0], score[1], kv[0], kv[1])
                                  : v_pair_1e(score[0], score[1], kv[0], kv[1]);
    if (D == 512) v_cur_decode<512>(value, w, out + g * ldo, eps);
    else v_cur_decode<128>(value, w, out + g * ldo, eps);
}

#define V_CUR_LAUNCH(name, RES, PAIR)                                                                                  \
    extern "C" __global__ void __launch_bounds__(512) name(const float* ckv, long long ckv_ld,                         \
        const float* cgate, long long cgate_ld, const long long* rslot, float* ring_kv, float* ring_score,               \
        const long long* gi, long long src_rows, const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,              \
        int G, int D, float eps, unsigned* invalid, int pdl) {                                                          \
        pdl_enter(pdl);                                                                                                 \
        v_cur_row<RES, PAIR>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo, G, D,   \
                             eps, invalid);                                                                             \
    }

V_CUR_LAUNCH(v_cur_r1, true, 0)
V_CUR_LAUNCH(v_cur_1e, false, 1)
V_CUR_LAUNCH(v_cur_r1_1e, true, 1)

// Launch-floor probes: an empty kernel, and the same with the engine kernel's static shared allocation.
extern "C" __global__ void __launch_bounds__(512) v_empty() {}
extern "C" __global__ void __launch_bounds__(512) v_empty2k(unsigned* sink) {
    __shared__ float buf[512];
    buf[threadIdx.x & 511] = 1.0f;
    __syncthreads();
    if (threadIdx.x == 0 && buf[0] == 2.0f) *sink = 1u;
}

// A kernel dedicated to one width: D compile-time, no runtime dispatch, block = D threads (the engine's geometry).
template<int D>
__device__ __forceinline__ void v_fixed_row(const float* ckv, long long ckv_ld, const float* cgate, long long cgate_ld,
                                            const long long* rslot, float* ring_kv, float* ring_score,
                                            const long long* gi, long long src_rows, const __nv_bfloat16* w,
                                            __nv_bfloat16* out, long long ldo, int G, float eps, unsigned* invalid) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const long long own = rslot[g];
    const float own_kv = ckv[g * ckv_ld + tid], own_score = cgate[g * cgate_ld + tid];
    ring_kv[own * D + tid] = own_kv;
    ring_score[own * D + tid] = own_score;
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const int2 pr = v_partners(rslot, G, own, q0, q1);
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if ((!ok0 || !ok1) && tid == 0 && invalid) atomicExch(invalid, 1u);
    float kv[2], score[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool ok = i == 0 ? ok0 : ok1;
        const int row = i == 0 ? pr.x : pr.y;
        const long long q = i == 0 ? q0 : q1;
        if (!ok) {
            kv[i] = score[i] = 0.f;
        } else if (row == g) {
            kv[i] = own_kv;
            score[i] = own_score;
        } else if (row >= 0) {
            kv[i] = ckv[row * ckv_ld + tid];
            score[i] = cgate[row * cgate_ld + tid];
        } else {
            kv[i] = ring_kv[q * D + tid];
            score[i] = ring_score[q * D + tid];
        }
    }
    const float value = v_pair_ref(score[0], score[1], kv[0], kv[1]);
    const float x = __bfloat162float(__float2bfloat16_rn(value));
    __shared__ float row[D == 512 ? 512 : 1];
    if (D == 512) {
        row[tid] = x;
        __syncthreads();
    }
    __shared__ float warp_sum[4];
    if (tid < 128) {
        float sum;
        if (D == 512) {
            const int base = tid * 4;
            sum = v_kit_squares4(row[base], row[base + 1], row[base + 2], row[base + 3]);
        } else {
            sum = __fmul_rn(x, x);
        }
        sum = v_kit_warp_sum(sum);
        if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(warp_sum[0], warp_sum[2]), __fadd_rn(warp_sum[1], warp_sum[3]));
    const float inv = v_kit_inverse_rms<D>(total, eps);
    const float y = __fmul_rn(x, inv);
    out[g * ldo + tid] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[tid]), y));
}

extern "C" __global__ void __launch_bounds__(512) v_512only(const float* ckv, long long ckv_ld,
                                                            const float* cgate, long long cgate_ld,
                                                            const long long* rslot, float* ring_kv, float* ring_score,
                                                            const long long* gi, long long src_rows,
                                                            const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                            int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    v_fixed_row<512>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo, G, eps,
                     invalid);
}

extern "C" __global__ void __launch_bounds__(128) v_128only(const float* ckv, long long ckv_ld,
                                                            const float* cgate, long long cgate_ld,
                                                            const long long* rslot, float* ring_kv, float* ring_score,
                                                            const long long* gi, long long src_rows,
                                                            const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                            int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    v_fixed_row<128>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo, G, eps,
                     invalid);
}

// The same D=128 kernel with a live 2 KiB shared array (the engine kernel's row[512] allocation) to price the
// allocation alone.
extern "C" __global__ void __launch_bounds__(128) v_128only_2k(const float* ckv, long long ckv_ld,
                                                               const float* cgate, long long cgate_ld,
                                                               const long long* rslot, float* ring_kv, float* ring_score,
                                                               const long long* gi, long long src_rows,
                                                               const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                               int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    __shared__ float big[512];
    big[threadIdx.x] = eps;
    v_fixed_row<128>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo, G, eps,
                     invalid);
    if (threadIdx.x == 0 && big[0] == 2.0f && invalid) *invalid = 2u;  // keep the array live
}

// Pair parity over arbitrary (s0, s1, kv0, kv1) rows: out[i] = 1 when the bf16-rounded pair sum differs.
extern "C" __global__ void v_pair_probe3(const float* s0, const float* s1, const float* kv0, const float* kv1, int n,
                                         unsigned* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned r = __bfloat16_as_ushort(__float2bfloat16_rn(v_pair_ref(s0[i], s1[i], kv0[i], kv1[i])));
    const unsigned o = __bfloat16_as_ushort(__float2bfloat16_rn(v_pair_1e(s0[i], s1[i], kv0[i], kv1[i])));
    out[i] = r != o;
}

// Dedicated one-width kernels for the landing: D compile-time, no runtime width dispatch.
extern "C" __global__ void __launch_bounds__(128) v_final128(const float* ckv, long long ckv_ld,
                                                             const float* cgate, long long cgate_ld,
                                                             const long long* rslot, float* ring_kv, float* ring_score,
                                                             const long long* gi, long long src_rows,
                                                             const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                             int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    v_row<128, 1, false, true, 0>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo,
                                  G, eps, invalid);
}

extern "C" __global__ void __launch_bounds__(128) v_final512x4(const float* ckv, long long ckv_ld,
                                                               const float* cgate, long long cgate_ld,
                                                               const long long* rslot, float* ring_kv, float* ring_score,
                                                               const long long* gi, long long src_rows,
                                                               const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                               int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    v_row<512, 4, false, true, 0>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo,
                                  G, eps, invalid);
}

extern "C" __global__ void __launch_bounds__(128) v_final128_1e(const float* ckv, long long ckv_ld,
                                                                const float* cgate, long long cgate_ld,
                                                                const long long* rslot, float* ring_kv, float* ring_score,
                                                                const long long* gi, long long src_rows,
                                                                const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                                int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    v_row<128, 1, false, true, 1>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo,
                                  G, eps, invalid);
}

extern "C" __global__ void __launch_bounds__(128) v_final128_nopair(const float* ckv, long long ckv_ld,
                                                                    const float* cgate, long long cgate_ld,
                                                                    const long long* rslot, float* ring_kv, float* ring_score,
                                                                    const long long* gi, long long src_rows,
                                                                    const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                                    int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    v_row<128, 1, false, true, 2>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo,
                                  G, eps, invalid);
}

// v_final128 with the engine kernel's `if (tid < 128)` guard around the sum, to price the guard.
extern "C" __global__ void __launch_bounds__(128) v_final128g(const float* ckv, long long ckv_ld,
                                                              const float* cgate, long long cgate_ld,
                                                              const long long* rslot, float* ring_kv, float* ring_score,
                                                              const long long* gi, long long src_rows,
                                                              const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                              int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    const int g = blockIdx.x, tid = threadIdx.x;
    const long long own = rslot[g];
    const float own_kv = ckv[g * ckv_ld + tid], own_score = cgate[g * cgate_ld + tid];
    ring_kv[own * 128 + tid] = own_kv;
    ring_score[own * 128 + tid] = own_score;
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const int2 pr = v_partners(rslot, G, own, q0, q1);
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if ((!ok0 || !ok1) && tid == 0 && invalid) atomicExch(invalid, 1u);
    float kv[2], score[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool ok = i == 0 ? ok0 : ok1;
        const int row = i == 0 ? pr.x : pr.y;
        const long long q = i == 0 ? q0 : q1;
        if (!ok) kv[i] = score[i] = 0.f;
        else if (row == g) { kv[i] = own_kv; score[i] = own_score; }
        else if (row >= 0) { kv[i] = ckv[row * ckv_ld + tid]; score[i] = cgate[row * cgate_ld + tid]; }
        else { kv[i] = ring_kv[q * 128 + tid]; score[i] = ring_score[q * 128 + tid]; }
    }
    const float x = __bfloat162float(__float2bfloat16_rn(v_pair_ref(score[0], score[1], kv[0], kv[1])));
    __shared__ float warp_sum[4];
    if (tid < 128) {
        float sum = __fmul_rn(x, x);
        sum = v_kit_warp_sum(sum);
        if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(warp_sum[0], warp_sum[2]), __fadd_rn(warp_sum[1], warp_sum[3]));
    const float inv = v_kit_inverse_rms<128>(total, eps);
    out[g * ldo + tid] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[tid]), __fmul_rn(x, inv)));
}

// 256 threads, two channels a thread, at width 512: the pair pass is spread over 256 threads and the kit's four-channel
// reduction still runs on 128 logical lanes out of the staged row (the 512-thread form's structure, half the threads).
extern "C" __global__ void __launch_bounds__(256) v_256x2(const float* ckv, long long ckv_ld,
                                                          const float* cgate, long long cgate_ld,
                                                          const long long* rslot, float* ring_kv, float* ring_score,
                                                          const long long* gi, long long src_rows,
                                                          const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,
                                                          int G, int D, float eps, unsigned* invalid, int pdl) {
    pdl_enter(pdl);
    const int g = blockIdx.x, tid = threadIdx.x, base = tid * 2;
    const long long own = rslot[g];
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const bool va = ((ckv_ld & 1) == 0) && ((cgate_ld & 1) == 0);
    float okv[2], osc[2];
    if (va) {
        const float2 a = *(const float2*)(ckv + g * ckv_ld + base);
        const float2 b = *(const float2*)(cgate + g * cgate_ld + base);
        okv[0] = a.x; okv[1] = a.y; osc[0] = b.x; osc[1] = b.y;
        *(float2*)(ring_kv + own * 512 + base) = a;
        *(float2*)(ring_score + own * 512 + base) = b;
    } else {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            okv[j] = ckv[g * ckv_ld + base + j];
            osc[j] = cgate[g * cgate_ld + base + j];
            ring_kv[own * 512 + base + j] = okv[j];
            ring_score[own * 512 + base + j] = osc[j];
        }
    }
    const int2 pr = v_partners(rslot, G, own, q0, q1);
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if ((!ok0 || !ok1) && tid == 0 && invalid) atomicExch(invalid, 1u);
    float kv[2][2], sc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool ok = i == 0 ? ok0 : ok1;
        const int row = i == 0 ? pr.x : pr.y;
        const long long q = i == 0 ? q0 : q1;
        if (!ok) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) kv[i][j] = sc[i][j] = 0.f;
        } else if (row == g) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) { kv[i][j] = okv[j]; sc[i][j] = osc[j]; }
        } else if (row >= 0) {
            if (va) {
                const float2 a = *(const float2*)(ckv + row * ckv_ld + base);
                const float2 b = *(const float2*)(cgate + row * cgate_ld + base);
                kv[i][0] = a.x; kv[i][1] = a.y; sc[i][0] = b.x; sc[i][1] = b.y;
            } else {
                #pragma unroll
                for (int j = 0; j < 2; ++j) { kv[i][j] = ckv[row * ckv_ld + base + j]; sc[i][j] = cgate[row * cgate_ld + base + j]; }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 2; ++j) { kv[i][j] = ring_kv[q * 512 + base + j]; sc[i][j] = ring_score[q * 512 + base + j]; }
        }
    }
    float x[2];
    #pragma unroll
    for (int j = 0; j < 2; ++j) x[j] = __bfloat162float(__float2bfloat16_rn(v_pair_ref(sc[0][j], sc[1][j], kv[0][j], kv[1][j])));
    __shared__ float row[512];
    row[base] = x[0];
    row[base + 1] = x[1];
    __syncthreads();
    __shared__ float warp_sum[4];
    if (tid < 128) {
        const int b4 = tid * 4;
        float sum = v_kit_squares4(row[b4], row[b4 + 1], row[b4 + 2], row[b4 + 3]);
        sum = v_kit_warp_sum(sum);
        if ((tid & 31) == 0) warp_sum[tid >> 5] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(warp_sum[0], warp_sum[2]), __fadd_rn(warp_sum[1], warp_sum[3]));
    const float inv = v_kit_inverse_rms<512>(total, eps);
    #pragma unroll
    for (int j = 0; j < 2; ++j)
        out[g * ldo + base + j] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[base + j]), __fmul_rn(x[j], inv)));
}

// Item 4: one block per speculative window of W rows; rows that resolve to the same ordered source pair share the
// softmax, the normalized latent and the norm, while every row still writes its own raw-ring slot and its own output
// row. Duplicates copy the representative's bf16 latent out of shared memory.
template<int W, bool DEDUPE>
__device__ __forceinline__ void v_win_body(const float* ckv, long long ckv_ld, const float* cgate, long long cgate_ld,
                                           const long long* rslot, float* ring_kv, float* ring_score,
                                           const long long* gi, long long src_rows, const __nv_bfloat16* w,
                                           __nv_bfloat16* out, long long ldo, int G, float eps, unsigned* invalid) {
    const int lane = threadIdx.x, y = threadIdx.y;
    const int slot = blockIdx.x * W + y;
    const bool live = slot < G;
    const int g = live ? slot : G - 1;
    const long long own = rslot[g];
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const float own_kv = ckv[g * ckv_ld + lane], own_score = cgate[g * cgate_ld + lane];
    if (live) {
        ring_kv[own * 128 + lane] = own_kv;
        ring_score[own * 128 + lane] = own_score;
    }
    const bool ok0 = q0 >= 0 && q0 < src_rows, ok1 = q1 >= 0 && q1 < src_rows;
    if (live && (!ok0 || !ok1) && lane == 0 && y == 0 && invalid) atomicExch(invalid, 1u);
    __shared__ int2 sp[W];
    __shared__ int rep[W];
    __shared__ float wsum[W][4];
    __shared__ __nv_bfloat16 lat[W][128];
    if (lane == 0) sp[y] = make_int2((int)q0, (int)q1);
    __syncthreads();
    if (lane == 0 && y == 0) {
        #pragma unroll
        for (int a = 0; a < W; ++a) {
            rep[a] = a;
            for (int b = 0; b < a; ++b)
                if (DEDUPE && sp[b].x == sp[a].x && sp[b].y == sp[a].y) { rep[a] = b; break; }
        }
    }
    __syncthreads();
    const int r = rep[y];
    float x = 0.f;
    if (r == y) {
        const int2 pr = v_partners(rslot, G, own, q0, q1);
        float kv[2], score[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const bool ok = i == 0 ? ok0 : ok1;
            const int row = i == 0 ? pr.x : pr.y;
            const long long q = i == 0 ? q0 : q1;
            if (!ok) {
                kv[i] = score[i] = 0.f;
            } else if (row == g) {
                kv[i] = own_kv;
                score[i] = own_score;
            } else if (row >= 0) {
                kv[i] = ckv[row * ckv_ld + lane];
                score[i] = cgate[row * cgate_ld + lane];
            } else {
                kv[i] = ring_kv[q * 128 + lane];
                score[i] = ring_score[q * 128 + lane];
            }
        }
        x = __bfloat162float(__float2bfloat16_rn(v_pair_ref(score[0], score[1], kv[0], kv[1])));
        float sum = __fmul_rn(x, x);
        sum = v_kit_warp_sum(sum);
        if ((lane & 31) == 0) wsum[y][lane >> 5] = sum;
    }
    __syncthreads();
    if (r == y) {
        const float total = __fadd_rn(__fadd_rn(wsum[y][0], wsum[y][2]), __fadd_rn(wsum[y][1], wsum[y][3]));
        const float inv = v_kit_inverse_rms<128>(total, eps);
        lat[y][lane] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(w[lane]), __fmul_rn(x, inv)));
    }
    __syncthreads();
    if (live) out[g * ldo + lane] = lat[r][lane];
}

#define V_WIN_LAUNCH(name, W, DEDUPE)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128 * W) name(const float* ckv, long long ckv_ld,                     \
        const float* cgate, long long cgate_ld, const long long* rslot, float* ring_kv, float* ring_score,              \
        const long long* gi, long long src_rows, const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo,             \
        int G, int D, float eps, unsigned* invalid, int pdl) {                                                         \
        pdl_enter(pdl);                                                                                                \
        v_win_body<W, DEDUPE>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, w, out, ldo, G,   \
                              eps, invalid);                                                                           \
    }

V_WIN_LAUNCH(v_win7, 7, true)
V_WIN_LAUNCH(v_win4, 4, true)
V_WIN_LAUNCH(v_win1, 1, true)
V_WIN_LAUNCH(v_win7n, 7, false)
