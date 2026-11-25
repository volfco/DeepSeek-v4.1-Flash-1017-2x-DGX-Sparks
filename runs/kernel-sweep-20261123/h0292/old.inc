// extracted from /home/colum/Workspace/tfref-kstack4/zig/kernels/cuda/dsv41_ops.cu by emu/extract.py - do not edit

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

