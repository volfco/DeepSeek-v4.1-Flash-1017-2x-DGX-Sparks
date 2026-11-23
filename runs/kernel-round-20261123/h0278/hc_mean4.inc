// A DSpark tap as the prompt forward takes it, h.to(fp32).mean(1).to(bf16) over the 4 streams: torch's reduce_kernel
// over the middle dim (a thread an output: four accumulators from 0, one value each, combined in order; too few
// values a thread to split them over the block), times MeanOps' factor 0.25, rounded to bf16. h [rows, 4, D] bf16,
// out [rows, D] (row stride os).
extern "C" __global__ void tf_ds_hc_mean4_kernel(const __nv_bfloat16* h, __nv_bfloat16* out, long long os, int rows,
                                                 int D, int pdl) {
    pdl_enter(pdl);
    const long long total = (long long)rows * D;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / D, d = i % D;
        const __nv_bfloat16* x = h + r * 4 * D + d;
        const float a0 = __fadd_rn(0.f, __bfloat162float(x[0]));
        const float a1 = __fadd_rn(0.f, __bfloat162float(x[D]));
        const float a2 = __fadd_rn(0.f, __bfloat162float(x[2 * D]));
        const float a3 = __fadd_rn(0.f, __bfloat162float(x[3 * D]));
        const float s = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
        out[r * os + d] = __float2bfloat16_rn(__fmul_rn(s, 0.25f));
    }
}
