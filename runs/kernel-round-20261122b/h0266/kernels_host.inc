// The pair-sum arithmetic of tf_ds_compress2_kernel / tf_ds_compress2x_kernel, verbatim (kv/score addressed by the
// caller): the pair's softmax (max, the two expf, the sum from 0), the products, the pair's sum as compress2 adds it.
__device__ __forceinline__ float tf_ds_compress2_pair(float s0, float s1, float kv0, float kv1) {
    float m = -3.402823466e+38f;   // numeric_limits<float>::lowest()
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


// These three steps mirror the instructions and reduction order of the pinned kit.
__device__ __forceinline__ float tf_ds_kit_squares4(float a, float b, float c, float d) {
    float sum = __fmul_rn(b, b);
    sum = __fmaf_rn(a, a, sum);
    sum = __fmaf_rn(c, c, sum);
    return __fmaf_rn(d, d, sum);
}

__device__ __forceinline__ float tf_ds_kit_warp_sum(float sum) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, offset));
    return sum;
}

template<int D>
__device__ __forceinline__ float tf_ds_kit_inverse_rms(float sum, float eps) {
    const float mean_eps = __fmaf_rn(sum, 1.0f / D, eps);
    float root, inv;
    root = kit_sqrt_approx(mean_eps);
    inv = kit_div_approx(1.0f, root);
    return inv;
}


// Each warp resolves both partners before reading any ring data. Use the first
// matching current row, including when that row belongs to another block.
__device__ __forceinline__ int2 tf_ds_compress2_partners(const long long* slots, int rows,
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


// Decode ratio-2 scatter, pair compression and kit RMSNorm. Launch exactly D threads per row, one channel a thread.
// The width is a compile-time parameter with one kernel exported per served width (decodeOk: 128 or 512), so neither
// kernel carries the other's code or its static shared allocation: at the served decode rows a dedicated 128-wide
// kernel measures 1.30/1.54/1.54/1.78 us (R = 1/7/32/48) against 1.50/1.73/1.73/1.95 for the shared one, and the
// 512-wide form 1.69/1.92/1.95/2.20 against 1.73/1.95/1.99/2.22. The 128-wide block is exactly the reduction's 128
// lanes, so it needs no `tid < 128` guard (that guard alone costs ~0.12 us: it puts the warp-sum store and the
// shuffles behind a runtime branch ptxas cannot fold away) and no shared `row`.
template<int D>
__device__ __forceinline__ void tf_ds_compress2x_norm_body(const float* ckv, long long ckv_ld,
                                                          const float* cgate, long long cgate_ld,
                                                          const long long* rslot, float* ring_kv,
                                                          float* ring_score, const long long* gi, long long src_rows,
                                                          __nv_bfloat16 wv, __nv_bfloat16* out, long long ldo,
                                                          int G, float eps, unsigned* invalid) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const long long own = rslot[g];
    // The partner slots are read before the current row: ptxas then issues both 64-bit slot loads ahead of the pair
    // loads, which hides their latency behind the scatter and the ballot scan (the other order measures 1.46 against
    // 1.32 us at R = 1, width 128, with otherwise identical SASS).
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const float own_kv = ckv[g * ckv_ld + tid], own_score = cgate[g * cgate_ld + tid];
    ring_kv[own * D + tid] = own_kv;
    ring_score[own * D + tid] = own_score;
    const int2 pr = tf_ds_compress2_partners(rslot, G, own, q0, q1);
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
    const float value = tf_ds_compress2_pair(score[0], score[1], kv[0], kv[1]);
    const float x = __bfloat162float(__float2bfloat16_rn(value));
    // At width 512 the kit first reduces four adjacent channels per logical lane, so the block stages the rounded row.
    if (D == 512) {
        blk->row[tid] = x;
        __syncthreads();
    }
    if (D == 512) {
        if (tid < 128) {
            const int base = tid * 4;
            float sum = tf_ds_kit_squares4(blk->row[base], blk->row[base + 1], blk->row[base + 2], blk->row[base + 3]);
            sum = tf_ds_kit_warp_sum(sum);
            if ((tid & 31) == 0) blk->warp_sum[tid >> 5] = sum;
        }
    } else {
        float sum = __fmul_rn(x, x);
        sum = tf_ds_kit_warp_sum(sum);
        if ((tid & 31) == 0) blk->warp_sum[tid >> 5] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(blk->warp_sum[0], blk->warp_sum[2]), __fadd_rn(blk->warp_sum[1], blk->warp_sum[3]));
    const float inv = tf_ds_kit_inverse_rms<D>(total, eps);
    const float y = __fmul_rn(x, inv);
    out[g * ldo + tid] = __float2bfloat16_rn(__fmul_rn(__bfloat162float(wv), y));
}

#define TF_DS_COMPRESS2X_NORM_KERNEL(name, D)                                                                          \
    __global__ void __launch_bounds__(D) name(const float* ckv, long long ckv_ld, const float* cgate,        \
        long long cgate_ld, const long long* rslot, float* ring_kv, float* ring_score, const long long* gi,             \
        long long src_rows, const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo, int G, int D_, float eps,         \
        unsigned* invalid, int pdl) {                                                                                   \
        (void)D_;                                                                                                       \
        if (pdl) {                                                                                                      \
            /* TF_DS_PDL: the successor may be scheduled now; the comp_norm weight is a fixed model weight (no writer \
               in the round), so it is read here, before the wait for the predecessor's bytes. */                        \
            pdl_launch(1);                                                                                              \
            const __nv_bfloat16 wv = w[threadIdx.x];                                                                    \
            pdl_wait(1);                                                                                                \
            tf_ds_compress2x_norm_body<D>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows, wv,  \
                                          out, ldo, G, eps, invalid);                                                   \
        } else {                                                                                                        \
            tf_ds_compress2x_norm_body<D>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows,       \
                                          w[threadIdx.x], out, ldo, G, eps, invalid);                                   \
        }                                                                                                               \
    }

TF_DS_COMPRESS2X_NORM_KERNEL(tf_ds_compress2x_norm128_kernel, 128)
TF_DS_COMPRESS2X_NORM_KERNEL(tf_ds_compress2x_norm512_kernel, 512)


// -- TF_DS_COMPRESS2_PAIR_ONCE (compress2_pair.zig, off by default) -------------------------------------------------
//
// Rows whose ordered pair of source slots (gi[2 g], gi[2 g + 1]) matches produce byte-identical latent rows: the pair
// sum and the kit norm read only the pair's resolved bytes, and equal keys resolve to the same bytes whichever row
// resolves them (the own-row and the current-row arm read the same ckv/cgate element). So the key's first row computes
// the pair and broadcasts the result by writing every matching row's output row directly; a later row's block stops
// after its own raw-ring scatter and never reads another block's result, so no ordering rule and no barrier is needed
// and the computing row's arithmetic is tf_ds_compress2x_norm_body's, unchanged. Keys carry the request's raw-slot
// base (setGlue's gi), so rows from different requests never share; a duplicate key inside one round can only be the
// row pair (2k, 2k + 1) of one stream (the fused scatter's distinct-rslot invariant bounds the repeats).
// The grid stays one block a row: the served timings are flat in the row count (1.75/2.00/2.01 us at R = 1/7/32 x 512,
// continue.md's 0197 table), so blocks already run concurrently and fewer, fatter blocks would serialise the pairs.
__device__ __forceinline__ int tf_ds_compress2_first_pair(const long long* gi, int rows, long long q0, long long q1) {
    for (int first = 0; first < rows; first += 32) {
        const int r = first + (threadIdx.x & 31);
        const long long a = r < rows ? gi[2 * r] : -1;
        const long long b = r < rows ? gi[2 * r + 1] : -1;
        const unsigned m = __ballot_sync(0xffffffffu, r < rows && a == q0 && b == q1);
        if (m) return first + __ffs(m) - 1;
    }
    return -1;
}

template<int D>
__device__ __forceinline__ void tf_ds_compress2x_norm_pair_body(const float* ckv, long long ckv_ld,
                                                               const float* cgate, long long cgate_ld,
                                                               const long long* rslot, float* ring_kv,
                                                               float* ring_score, const long long* gi, long long src_rows,
                                                               __nv_bfloat16 wv, __nv_bfloat16* out, long long ldo,
                                                               int G, float eps, unsigned* invalid) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const long long own = rslot[g];
    // The partner slots are read before the current row: ptxas then issues both 64-bit slot loads ahead of the pair
    // loads (tf_ds_compress2x_norm_body's order, kept verbatim).
    const long long q0 = gi[2 * g], q1 = gi[2 * g + 1];
    const float own_kv = ckv[g * ckv_ld + tid], own_score = cgate[g * cgate_ld + tid];
    ring_kv[own * D + tid] = own_kv;
    ring_score[own * D + tid] = own_score;
    if (tf_ds_compress2_first_pair(gi, G, q0, q1) != g) return; // this row's output row is its key's first row's bytes
    const int2 pr = tf_ds_compress2_partners(rslot, G, own, q0, q1);
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
    const float value = tf_ds_compress2_pair(score[0], score[1], kv[0], kv[1]);
    const float x = __bfloat162float(__float2bfloat16_rn(value));
    // At width 512 the kit first reduces four adjacent channels per logical lane, so the block stages the rounded row.
    if (D == 512) {
        blk->row[tid] = x;
        __syncthreads();
    }
    if (D == 512) {
        if (tid < 128) {
            const int base = tid * 4;
            float sum = tf_ds_kit_squares4(blk->row[base], blk->row[base + 1], blk->row[base + 2], blk->row[base + 3]);
            sum = tf_ds_kit_warp_sum(sum);
            if ((tid & 31) == 0) blk->warp_sum[tid >> 5] = sum;
        }
    } else {
        float sum = __fmul_rn(x, x);
        sum = tf_ds_kit_warp_sum(sum);
        if ((tid & 31) == 0) blk->warp_sum[tid >> 5] = sum;
    }
    __syncthreads();
    const float total = __fadd_rn(__fadd_rn(blk->warp_sum[0], blk->warp_sum[2]), __fadd_rn(blk->warp_sum[1], blk->warp_sum[3]));
    const float inv = tf_ds_kit_inverse_rms<D>(total, eps);
    const float y = __fmul_rn(x, inv);
    const __nv_bfloat16 yv = __float2bfloat16_rn(__fmul_rn(__bfloat162float(wv), y));
    // The broadcast: every row of this key gets the computing row's bytes (equal keys: identical values by the rule
    // above), stored straight to its output row so no row reads a staged result.
    for (int j = 0; j < G; ++j)
        if (gi[2 * j] == q0 && gi[2 * j + 1] == q1) out[(long long)j * ldo + tid] = yv;
}

#define TF_DS_COMPRESS2X_NORM_PAIR_KERNEL(name, D)                                                                   \
    __global__ void __launch_bounds__(D) name(const float* ckv, long long ckv_ld, const float* cgate,       \
        long long cgate_ld, const long long* rslot, float* ring_kv, float* ring_score, const long long* gi,            \
        long long src_rows, const __nv_bfloat16* w, __nv_bfloat16* out, long long ldo, int G, int D_, float eps,        \
        unsigned* invalid, int pdl) {                                                                                  \
        (void)D_;                                                                                                      \
        if (pdl) {                                                                                                     \
            /* TF_DS_PDL: the successor may be scheduled now; the comp_norm weight is a fixed model weight (no writer  \
               in the round), so it is read here, before the wait for the predecessor's bytes. (0197's split.) */       \
            pdl_launch(1);                                                                                             \
            const __nv_bfloat16 wv = w[threadIdx.x];                                                                   \
            pdl_wait(1);                                                                                               \
            tf_ds_compress2x_norm_pair_body<D>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows,  \
                                               wv, out, ldo, G, eps, invalid);                                          \
        } else {                                                                                                       \
            tf_ds_compress2x_norm_pair_body<D>(ckv, ckv_ld, cgate, cgate_ld, rslot, ring_kv, ring_score, gi, src_rows,  \
                                               w[threadIdx.x], out, ldo, G, eps, invalid);                             \
        }                                                                                                              \
    }

TF_DS_COMPRESS2X_NORM_PAIR_KERNEL(tf_ds_compress2x_norm128_pair_kernel, 128)
TF_DS_COMPRESS2X_NORM_PAIR_KERNEL(tf_ds_compress2x_norm512_pair_kernel, 512)

// ===================================================================================================================

