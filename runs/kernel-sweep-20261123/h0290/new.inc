// extracted from /home/colum/Workspace/tfref-ksweep/zig/kernels/cuda/dsv41_ops.cu by emu/extract.py - do not edit

__device__ unsigned tf_ds_block_argmax_v(const float* row, int n, float* out_v) {
    const float ninf = -__int_as_float(0x7f800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        if (v == v && (v > bv || (v == bv && i < bi))) {
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return 0;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : ninf;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    *out_v = bv;
    return (unsigned)bi;
}

__device__ unsigned tf_ds_block_argmax(const float* row, int n) {
    float v;
    const unsigned bi = tf_ds_block_argmax_v(row, n, &v);
    return row[0] != row[0] ? 0u : bi;
}


__device__ __forceinline__ uint64_t tf_ds_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}


struct TfDsDrawRow {
    uint64_t seed;
    double temp;
    double top_p;
    double min_p;
    uint64_t pos;
    uint32_t top_k;  // 1..56 when temp > 0
    uint32_t pad;
};

__device__ __forceinline__ uint64_t tf_ds_draw_key(float v, int id) {
    uint32_t b = __float_as_uint(v);
    if ((b << 1) == 0) b = 0;
    const uint32_t hi = v != v ? 0u : ((b & 0x80000000u) ? ~b : (b | 0x80000000u));
    return (uint64_t(hi) << 32) | uint64_t(0xFFFFFFFFu - uint32_t(id));
}

// TF_DS_DRAW_CUT (draw_cut.zig; `Cut`): the radix select stops at a complete bin. Once the bins above the cutoff are
// consumed and the cutoff bin holds exactly the keys still wanted, every key at or above the partial prefix (its lower
// digits zero) is a candidate: exactly k keys, the same set the remaining digits would reach (keys are unique), so the
// compaction below and the rank sort see the same candidates. Thread 0 flags the bin in bit 8 of `digit`. The keys,
// the ids and the order are tf_ds_draw_key's as before; no digit is skipped on a constant byte (a rank's global ids
// straddle 65,536).
template <bool Cut>
__device__ __forceinline__ void tf_ds_draw_rows(const float* logits, unsigned long long ld, int n,
                                                const TfDsDrawRow* rules, unsigned* out, int pdl) {
    pdl_enter(pdl);
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    const TfDsDrawRow rule = rules[blockIdx.x];
    if (rule.temp <= 0.0) {
        const unsigned token = tf_ds_block_argmax(row, n);
        if (threadIdx.x == 0) out[blockIdx.x] = token;
        return;
    }
    __shared__ unsigned hist[256];
    __shared__ uint64_t sel[64];
    __shared__ unsigned sid[64];
    __shared__ double scaled[64];
    __shared__ double score[64];
    __shared__ unsigned count, digit, remain;
    const int tid = threadIdx.x;
    const unsigned k = min(rule.top_k, (unsigned)n);

    uint64_t prefix = 0;
    unsigned want = k;
    for (int d = 0; d < 8; ++d) {
        const int shift = 56 - 8 * d;
        for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            const uint64_t key = tf_ds_draw_key(row[i], i);
            if (d == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned above = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (above + hist[b] >= want) break;
                above += hist[b];
            }
            digit = b;
            remain = want - above;
            if (Cut && want != 0 && hist[b] == want - above) digit |= 0x100u;   // the bin completes the quota
        }
        __syncthreads();
        const unsigned dg = digit;
        prefix |= uint64_t(Cut ? (dg & 0xFFu) : dg) << shift;
        want = remain;
        if (Cut && (dg >> 8) != 0) break;
    }
    // prefix is the k-th largest key (Cut: or the complete bin's prefix, which the same k keys reach)
    if (tid == 0) count = 0;
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) {
        const uint64_t key = tf_ds_draw_key(row[i], i);
        if (key >= prefix) {
            const unsigned at = atomicAdd(&count, 1u);
            if (at < 64) sel[at] = key;
        }
    }
    __syncthreads();
    if (tid < (int)k) {
        unsigned rank = 0;
        for (unsigned m = 0; m < k; ++m) rank += sel[m] > sel[tid] ? 1u : 0u;
        sid[rank] = ~uint32_t(sel[tid]);
    }
    __syncthreads();
    const double t = fmax(rule.temp, 1e-6);
    if (tid < (int)k) {
        const unsigned id = sid[tid];
        scaled[tid] = double(row[id]) / t;
        uint64_t x = tf_ds_mix64(rule.seed + 0x9E3779B97F4A7C15ull);
        x = tf_ds_mix64(x ^ (rule.pos * 0xD1B54A32D192ED03ull));
        x = tf_ds_mix64(x ^ uint64_t(id));
        const double u = double(x >> 11) * 0x1p-53 + 0x1p-54;
        score[tid] = scaled[tid] - log(-log(u));
    }
    __syncthreads();
    if (tid != 0) return;
    const double ninf = -__longlong_as_double(0x7ff0000000000000ll);
    if (0.0 < rule.top_p && rule.top_p < 1.0) {
        double mx = scaled[0];
        for (unsigned j = 1; j < k; ++j) mx = fmax(mx, scaled[j]);
        double probs[64];
        for (unsigned j = 0; j < k; ++j) probs[j] = exp(scaled[j] - mx);
        // numpy's pairwise sum of k <= 56 values (sampling.pairwiseSum)
        double sum;
        if (k < 8) {
            sum = 0.0;
            for (unsigned j = 0; j < k; ++j) sum += probs[j];
        } else {
            double r[8];
            for (int j = 0; j < 8; ++j) r[j] = probs[j];
            unsigned i = 8;
            for (; i < k - k % 8; i += 8) {
                for (int j = 0; j < 8; ++j) r[j] += probs[i + j];
            }
            sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < k; ++i) sum += probs[i];
        }
        double acc = 0.0;
        unsigned below = 0;
        for (unsigned j = 0; j < k; ++j) {
            acc += probs[j] / sum;
            if (acc < rule.top_p) below += 1;
        }
        for (unsigned j = below + 1; j < k; ++j) score[j] = ninf;
    }
    if (rule.min_p > 0.0) {
        const double floor_ = scaled[0] + log(rule.min_p);
        for (unsigned j = 0; j < k; ++j) {
            if (scaled[j] < floor_) score[j] = ninf;
        }
    }
    unsigned best = 0;
    for (unsigned j = 0; j < k; ++j) {
        if (score[j] > score[best]) best = j;
    }
    out[blockIdx.x] = sid[best];
}

__global__ void __launch_bounds__(1024) tf_ds_draw_rows_kernel(const float* logits, unsigned long long ld, int n,
                                                                           const TfDsDrawRow* rules, unsigned* out, int pdl) {
    tf_ds_draw_rows<false>(logits, ld, n, rules, out, pdl);
}

__global__ void __launch_bounds__(1024) tf_ds_draw_rows_cut_kernel(const float* logits, unsigned long long ld,
                                                                               int n, const TfDsDrawRow* rules,
                                                                               unsigned* out, int pdl) {
    tf_ds_draw_rows<true>(logits, ld, n, rules, out, pdl);
}

// TF_DS_TOPK_MERGE (topk_merge.zig): a rank's best kmax candidates a row of its own head columns, for the exchange that
// replaces the vocabulary halves' gather in a device-drawn round. logits [rows, n] fp32 (row stride ld) is this rank's
// head output; its column c is the whole row's col0 + c. out [rows][kmax][2] u32: the value's bits (as read: -0 and NaN
// unchanged) and the global id, in tf_ds_draw_key order (descending), selected by tf_ds_draw_rows_kernel's radix select
// and rank sort. Bit 31 of a row's first id says this rank's column 0 is NaN (only rank 0's is read: the whole row's
// row[0] rule, as tf_ds_argmax_pair_kernel carries it). One block (1024 threads) a row; kmax <= 56 <= n.
// `Cut` (TF_DS_DRAW_CUT): the same complete-bin stop as tf_ds_draw_rows'.
template <bool Cut>
__device__ __forceinline__ void tf_ds_topk_select(const float* logits, unsigned long long ld, int n, unsigned col0,
                                                  int kmax, unsigned* out, int pdl) {
    pdl_enter(pdl);
    const float* row = logits + (unsigned long long)blockIdx.x * ld;
    unsigned* o = out + (unsigned long long)blockIdx.x * kmax * 2;
    __shared__ unsigned hist[256];
    __shared__ uint64_t sel[64];
    __shared__ unsigned count, digit, remain;
    const int tid = threadIdx.x;
    const unsigned k = min((unsigned)kmax, (unsigned)n);

    uint64_t prefix = 0;
    unsigned want = k;
    for (int d = 0; d < 8; ++d) {
        const int shift = 56 - 8 * d;
        for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            const uint64_t key = tf_ds_draw_key(row[i], (int)(col0 + (unsigned)i));
            if (d == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned above = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (above + hist[b] >= want) break;
                above += hist[b];
            }
            digit = b;
            remain = want - above;
            if (Cut && want != 0 && hist[b] == want - above) digit |= 0x100u;   // the bin completes the quota
        }
        __syncthreads();
        const unsigned dg = digit;
        prefix |= uint64_t(Cut ? (dg & 0xFFu) : dg) << shift;
        want = remain;
        if (Cut && (dg >> 8) != 0) break;
    }
    // prefix is the k-th largest key, or under Cut the complete bin's prefix (keys are unique: exactly k at or above
    // either)
    if (tid == 0) count = 0;
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) {
        const uint64_t key = tf_ds_draw_key(row[i], (int)(col0 + (unsigned)i));
        if (key >= prefix) {
            const unsigned at = atomicAdd(&count, 1u);
            if (at < 64) sel[at] = key;
        }
    }
    __syncthreads();
    if (tid < (int)k) {
        unsigned rank = 0;
        for (unsigned m = 0; m < k; ++m) rank += sel[m] > sel[tid] ? 1u : 0u;
        const unsigned id = ~uint32_t(sel[tid]);
        o[2 * rank] = __float_as_uint(row[id - col0]);
        o[2 * rank + 1] = id | (rank == 0 && row[0] != row[0] ? 0x80000000u : 0u);
    }
}

__global__ void __launch_bounds__(1024) tf_ds_topk_select_kernel(const float* logits, unsigned long long ld, int n,
                                                                             unsigned col0, int kmax, unsigned* out, int pdl) {
    tf_ds_topk_select<false>(logits, ld, n, col0, kmax, out, pdl);
}

__global__ void __launch_bounds__(1024) tf_ds_topk_select_cut_kernel(const float* logits, unsigned long long ld,
                                                                                 int n, unsigned col0, int kmax,
                                                                                 unsigned* out, int pdl) {
    tf_ds_topk_select<true>(logits, ld, n, col0, kmax, out, pdl);
}

