// TF_DS_RESUME_ASSEMBLE (resume_assemble.zig): an exact-resume prepend's rows and positions in one launch
// (prompt.zig's replayCut). Output row i of n takes its residual (row_h16 16-byte words) and its mixing weights
// (row_p16 words) from the tail ring's slot (replay + i) % window (i < pre: the restored prefix, the wrap in the
// slot) or from fresh row first + i - pre, and writes its position replay + i: replayCut's two fresh copyRows,
// tailRowsCopy's one or two ring segments a field and the host position upload, one launch. Only called under the
// switch with resume_assemble.ok's shape (16-byte words, every buffer word-aligned).
extern "C" __global__ void tf_ds_resume_assemble_kernel(const uint4* tail_h, const uint4* tail_p,
                                                       const uint4* fresh_h, const uint4* fresh_p, uint4* out_h,
                                                       uint4* out_p, long long* pos, unsigned long long row_h16,
                                                       unsigned long long row_p16, unsigned long long window,
                                                       int pre, int first, long long replay, int pdl) {
    pdl_enter(pdl);
    const int i = blockIdx.x; // the output row: a block a row, a 16-byte word a thread
    const unsigned long long p = (unsigned long long)replay + i;
    const bool from_tail = i < pre;
    const unsigned long long r = from_tail ? p % window : (unsigned long long)(first + i - pre);
    const uint4* sh = (from_tail ? tail_h : fresh_h) + r * row_h16;
    const uint4* sp = (from_tail ? tail_p : fresh_p) + r * row_p16;
    for (unsigned long long w = threadIdx.x; w < row_h16; w += blockDim.x)
        out_h[(unsigned long long)i * row_h16 + w] = sh[w];
    for (unsigned long long w = threadIdx.x; w < row_p16; w += blockDim.x)
        out_p[(unsigned long long)i * row_p16 + w] = sp[w];
    if (threadIdx.x == 0) pos[i] = (long long)p;
}

