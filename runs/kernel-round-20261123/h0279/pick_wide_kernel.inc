// pick_wide_kernel.inc: extracted from zig/kernels/cuda/dsv41_ops.cu by extract.py - do not edit
// topk_indices' key of score x at column j as an unsigned integer in the keys' signed order: x's bits (-0 as +0, as
// score + 0.0 makes it), ordered as signed integers, above the inverted column.
__device__ __forceinline__ unsigned long long tf_ds_score_key(float x, int j) {
    unsigned int bits = __float_as_uint(x);
    if (bits == 0x80000000u) bits = 0u;                        // + 0.0
    const int sb = (int)bits;
    const int ordered = sb < 0 ? (sb ^ 0x7FFFFFFF) : sb;
    const unsigned long long key = ((unsigned long long)(long long)ordered << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned int)j);
    return key ^ 0x8000000000000000ull;                       // signed order as unsigned
}

// An inclusive cutoff on the scores for a pool of at most 8192 blocks: the radix scan runs the score's four bytes only
// (digits 7-4). The column bytes are not scanned: when the cutoff bin holds more columns than the slots left, `want_s`
// is that many tied columns, and the compaction takes the bin's lowest columns in ascending order (the same total order
// as scanning the column digits, without two more full passes over the pool). `want_s` 0 means the bin completed
// exactly, so every column in it is picked.
__device__ __forceinline__ void tf_ds_pool_cut(const float* row, int n, unsigned long long* prefix_s,
                                               unsigned long long* mask_s, int* want_s) {
    __shared__ unsigned int hist[256];
    for (int d = 7; d >= 4; --d) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned long long prefix = *prefix_s, mask = *mask_s;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned long long u = tf_ds_score_key(row[i], i);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> (8 * d)) & 0xff], 1u);
        }
        __syncthreads();
        // Each lane covers eight descending bins. Only the lane containing the target rank scans its eight counts.
        if (threadIdx.x < 32) {
            const int lane = threadIdx.x;
            const int hi = 255 - 8 * lane;
            const int target = *want_s;
            int count = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) count += hist[hi - j];
            int end = count;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int prev = __shfl_up_sync(0xffffffffu, end, off);
                if (lane >= off) end += prev;
            }
            int rank = target - (end - count);
            if (rank > 0 && rank <= count) {
                int digit = hi;
                while (rank > (int)hist[digit]) {
                    rank -= hist[digit];
                    --digit;
                }
                *want_s = rank == (int)hist[digit] ? 0 : rank;
                *prefix_s = prefix | ((unsigned long long)digit << (8 * d));
                *mask_s = mask | (0xffull << (8 * d));
            }
        }
        __syncthreads();
        if (*want_s == 0) break;
    }
}

// The prefix count of `flag` over this tile's lanes in ascending column order (the block's whole mask reaches the ballot
// and the barrier), and the tile's total.
__device__ __forceinline__ int tf_ds_tile_prefix(bool flag, int* warp_end, int* total, int lane, int warp) {
    const unsigned bits = __ballot_sync(0xffffffffu, flag);
    if (lane == 0) warp_end[warp] = __popc(bits);
    if (threadIdx.x < 32 && threadIdx.x >= ((blockDim.x + 31) >> 5)) warp_end[threadIdx.x] = 0;  // (a block below 1024)
    __syncthreads();
    if (warp == 0) {
        int t = warp_end[lane];
        #pragma unroll
        for (int step = 1; step < 32; step <<= 1) {
            const int prev = __shfl_up_sync(0xffffffffu, t, step);
            if (lane >= step) t += prev;
        }
        warp_end[lane] = t;
    }
    __syncthreads();
    *total = warp_end[31];
    return (warp ? warp_end[warp - 1] : 0) + __popc(bits & ((1u << lane) - 1));
}

// The pool's selection and its two outputs: the cutoff on the score digits with the tied columns resolved during
// compaction (tf_ds_pool_cut), then either the mask alone (the prompt arm: no ballot, prefix sum or compaction) or the
// decode arm's ascending block list. `maxima` is one block a column - on chip for the fused/pick forms, the global
// maxima row for the wide pick (read only here either way), `limit` the row's `every`.
__device__ __forceinline__ void tf_ds_idx_select(const float* maxima, int nb, int k, uint8_t* mask, long long ms,
                                                 int* cblk, int r, long long limit) {
    __shared__ unsigned long long prefix_s, mask_s;
    __shared__ int want_s;
    __shared__ int warp_end[32];
    if (threadIdx.x == 0) {
        // k == nb picks every block: mask 0 makes every key "tied", and want 0 picks the whole bin
        const bool all = k == nb;
        prefix_s = all ? 0ull : 0xffff0000ull;   // (inverted columns below 65,536 have two constant high bytes)
        mask_s = all ? 0ull : 0xffff0000ull;
        want_s = all ? 0 : k;
    }
    __syncthreads();
    if (k != nb) tf_ds_pool_cut(maxima, nb, &prefix_s, &mask_s, &want_s);
    __syncthreads();
    const unsigned long long prefix = prefix_s, cmask = mask_s;
    const int want = want_s;
    const float ninf = -__int_as_float(0x7f800000);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (cblk == nullptr) {
        // the prompt arm reads only the mask: each block's bit straight from the cutoff, no ballot, prefix sum or
        // compaction unless the cutoff straddles ties (then the tied columns' order needs the prefix)
        int tied_base = 0;
        for (int first = 0; first < nb; first += blockDim.x) {
            const int col = first + threadIdx.x;
            const bool in = col < nb;
            const float value = in ? maxima[col] : ninf;
            const unsigned long long u = tf_ds_score_key(value, col);
            bool picked;
            if (want == 0) picked = in && (u >= prefix);   // (the whole cutoff bin, no tie split)
            else {
                const bool above = in && ((u & cmask) > prefix);
                const bool tied = in && ((u & cmask) == prefix);
                int before, total;
                before = tf_ds_tile_prefix(tied, warp_end, &total, lane, warp);
                picked = above || (tied && (tied_base + before) < want);
                tied_base += total;
                __syncthreads();   // the tile's total is read before warp_end is reused
            }
            if (mask != nullptr && in) mask[(long long)r * ms + col] = (uint8_t)(picked && col < limit && value > ninf);
        }
    } else {
        // decode: compact the selected blocks in ascending column order (the tied cutoff's slots go to its lowest
        // columns, which is the ascending scan's first `want` of them)
        int base = 0, tied_base = 0, ttotal = 0;
        for (int first = 0; first < nb; first += blockDim.x) {
            const int col = first + threadIdx.x;
            const bool in = col < nb;
            const float value = in ? maxima[col] : ninf;
            const unsigned long long u = tf_ds_score_key(value, col);
            bool picked;
            if (want == 0) picked = in && (u >= prefix);   // (the whole cutoff bin, no tie split)
            else {
                const bool above = in && ((u & cmask) > prefix);
                const bool tied = in && ((u & cmask) == prefix);
                const int tbefore = tf_ds_tile_prefix(tied, warp_end, &ttotal, lane, warp);
                picked = above || (tied && (tied_base + tbefore) < want);
                tied_base += ttotal;
                __syncthreads();   // the tied total is read before warp_end is reused
            }
            int before, total;
            before = tf_ds_tile_prefix(picked, warp_end, &total, lane, warp);
            const bool valid = picked && col < limit && value > ninf;
            if (mask != nullptr && in) mask[(long long)r * ms + col] = (uint8_t)valid;
            if (picked) cblk[(long long)r * k + base + before] = valid ? col : -1;
            base += total;
            __syncthreads();   // all warps consumed the offsets before the next tile
        }
    }
}

// TF_DS_IDX_PICK_WIDE: tf_ds_idx_pick_kernel's selection without the on-chip staging - the row's maxima are read from
// global memory (block_max's output, L2-resident: one fp32 maximum a block, <= 256 KiB a row at 65,536 blocks), so the
// selection reaches past the shared-memory staging bound (101,376 B opt-in) up to 65,536 blocks: the radix shortcut's
// bound, where the inverted column's two high bytes stay 0xffff (a column's inverted id leaves 0xFFFE in them at
// 65,537, which keeps the unfused fallback). Same block list as the unfused sequence, same order, same ties.
extern "C" __global__ void __launch_bounds__(1024) tf_ds_idx_pick_wide_kernel(const float* bmax, long long bs, int nbk,
                                                                             int k, const long long* every, int* cblk,
                                                                             int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.x;
    tf_ds_idx_select(bmax + (long long)r * bs, nbk, k, nullptr, 0, cblk, r, every[r]);
}
