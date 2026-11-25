// extracted from /home/colum/Workspace/tfref-ksweep/zig/kernels/cuda/dsv41_ops.cu by emu/extract.py - do not edit

__global__ void tf_ds_block_max_kernel(const float* score, long long ss, int width, int bsize,
                                                  const long long* vis, float* out, long long os, int nb, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.y;
    const long long t = vis[r] - 1;
    const long long last = t >= 0 ? t / bsize : -((-t + bsize - 1) / bsize);
    const float ninf = -__int_as_float(0x7f800000);
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        const float* s = score + (long long)r * ss + (long long)b * bsize;
        float m = ninf;
        bool nan = false;
        for (int j = 0; j < bsize && b * bsize + j < width; ++j) {
            const float x = s[j];
            if (x != x) nan = true;
            else if (x > m) m = x;
        }
        float v = nan ? __int_as_float(0x7fc00000) : m;
        if (b == last) v = __int_as_float(0x7f800000);
        out[(long long)r * os + b] = v;
    }
}

// TF_DS_BLOCK_MAX8 (block_max8.zig): tf_ds_block_max_kernel at the served block size over 16-byte-aligned rows
// (block_max8.ok: bsize 8, the score address and the row stride multiples of 16 bytes). A whole block's eight scores
// come in two aligned 16-byte loads, then the scalar loop's comparison order (a NaN wins; otherwise a greater score
// replaces the maximum); the row's last partial block keeps the guarded scalar loop. tf_ds_block_max_t<8>'s body
// (0213) over tf_ds_block_max_kernel's grid: the same bytes out.
__global__ void tf_ds_block_max8_kernel(const float* score, long long ss, int width, int /*bsize*/,
                                                   const long long* vis, float* out, long long os, int nb, int pdl) {
    pdl_enter(pdl);
    const int r = blockIdx.y;
    const long long t = vis[r] - 1;
    const long long last = t >= 0 ? t / 8 : -((-t + 8 - 1) / 8);
    const float ninf = -__int_as_float(0x7f800000);
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
        const float* s = score + (long long)r * ss + (long long)b * 8;
        float m = ninf;
        bool nan = false;
        if (b * 8 + 8 <= width) {
            const float4 lo = *reinterpret_cast<const float4*>(s);
            const float4 hi = *reinterpret_cast<const float4*>(s + 4);
            const float x[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (x[j] != x[j]) nan = true;
                else if (x[j] > m) m = x[j];
            }
        } else {
            for (int j = 0; j < 8 && b * 8 + j < width; ++j) {
                const float x = s[j];
                if (x != x) nan = true;
                else if (x > m) m = x;
            }
        }
        float v = nan ? __int_as_float(0x7fc00000) : m;
        if (b == last) v = __int_as_float(0x7f800000);
        out[(long long)r * os + b] = v;
    }
}

