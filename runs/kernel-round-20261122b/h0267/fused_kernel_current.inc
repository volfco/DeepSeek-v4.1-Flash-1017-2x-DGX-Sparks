__device__ __forceinline__ int tf_ds_markov_better(float v1, int i1, float v2, int i2) {
    const int n1 = v1 != v1, n2 = v2 != v2;
    const int gt = (v1 > v2) | ((v1 == v2) & (i1 < i2));
    return (n1 & (!n2 | (i1 < i2))) | (!n1 & !n2 & gt);
}

extern "C" __global__ void __launch_bounds__(256) tf_ds_markov_fused_kernel(
    const float* LG, uint64_t l_seg, uint64_t l_row, const int64_t* TOK, uint64_t t_stride, const int32_t* SLOT,
    const __half* CACHE, uint64_t c_row, const __nv_bfloat16* EMB, const __half* HEAD, int rank, float* PV, int* PI,
    int n_rows, int n_cols, int seg0, int n_part, int pdl) {
    pdl_enter(pdl);
    SimBlock& shm = *blk;
    const int t = (int)blockIdx.x, s = (int)blockIdx.y;
    const int p = s * (int)gridDim.x + t;
    const long long v0 = (long long)(seg0 + s) * n_cols; // the segment's first global id
    const int col0 = t * 1024;
    unsigned long long tokv[16];
    int slotv[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        tokv[r] = 0;
        slotv[r] = -1;
        if (r < n_rows) {
            const int64_t tok = TOK[(long long)r * t_stride];
            tokv[r] = (unsigned long long)tok;
            slotv[r] = SLOT[tok];
        }
    }
    float bv[16];
    int bi[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        bv[r] = -INFINITY; // the masked columns' (-inf, sentinel) identity
        bi[r] = 1 << 30;
    }
    for (int q = 0; q < 4; ++q) {
        const int c = col0 + (int)threadIdx.x + 256 * q;
        if (c >= n_cols) continue;
        float acc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
        for (int k = 0; k < rank; ++k) {
            const float wk = __half2float(HEAD[(size_t)(v0 + c) * (unsigned)rank + k]);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r >= n_rows || slotv[r] >= 0) continue; // the row's bias is cached: no dot
                const float ek = __half2float(__float2half_rn(__bfloat162float(EMB[tokv[r] * (unsigned)rank + k])));
                acc[r] = __fadd_rn(acc[r], __fmul_rn(wk, ek));
            }
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= n_rows) continue;
            const float b = slotv[r] >= 0
                ? __half2float(CACHE[(uint64_t)slotv[r] * c_row + (uint64_t)s * n_cols + c])
                : __half2float(__float2half_rn(acc[r]));
            const float sc = __fadd_rn(LG[(uint64_t)s * l_seg + (uint64_t)r * l_row + c], b);
            const int v = (int)(v0 + c);
            if (tf_ds_markov_better(sc, v, bv[r], bi[r])) {
                bv[r] = sc;
                bi[r] = v;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= n_rows) continue; // uniform in the block: every thread reaches the same barriers
        shm.sv[threadIdx.x] = bv[r];
        shm.si[threadIdx.x] = bi[r];
        __syncthreads();
        if (threadIdx.x == 0) {
            float ev = shm.sv[0];
            int ei = shm.si[0];
            for (int i = 1; i < 256; ++i)
                if (tf_ds_markov_better(shm.sv[i], shm.si[i], ev, ei)) {
                    ev = shm.sv[i];
                    ei = shm.si[i];
                }
            PV[(size_t)r * n_part + p] = ev;
            PI[(size_t)r * n_part + p] = ei;
        }
        __syncthreads();
    }
}

