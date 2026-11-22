// TF_DS_CONF_PACK (conf_pack.zig): the DSpark confidence head's input, conf_in fp32 [streams * steps, d + rank], in one
// launch. Output row j = i * steps + s (stream i, step s) reads token ids[i * (steps + 1) + s]: its first d floats are xc
// row i * n + s widened, the next rank floats the token's markov_embed row widened, which is also copied into embs row j
// (the chain's gather buffer). An id outside [0, vocab) sets the flag (tf_gather_rows_kernel's rule) and leaves embs row
// j as it was, and that stale row is what gets widened: the chain widened the gather's destination either way. One
// block a row, threads striding the row's columns.
extern "C" __global__ void tf_ds_conf_pack_kernel(const __nv_bfloat16* xc, const __nv_bfloat16* embed, const int64_t* ids,
                                                  __nv_bfloat16* embs, float* conf_in, uint32_t* invalid, int n, int steps,
                                                  int d, int rank, unsigned long long vocab, int pdl) {
    pdl_enter(pdl);
    const int j = blockIdx.x;
    const int i = j / steps, s = j - i * steps;
    const int64_t id = ids[(size_t)i * (steps + 1) + s];
    const bool ok = id >= 0 && (unsigned long long)id < vocab;
    if (!ok && threadIdx.x == 0) atomicExch(invalid, 1u);
    const __nv_bfloat16* x = xc + ((size_t)i * n + s) * d;
    __nv_bfloat16* e = embs + (size_t)j * rank;
    const __nv_bfloat16* src = ok ? embed + (size_t)id * rank : e;
    float* out = conf_in + (size_t)j * (d + rank);
    for (int c = threadIdx.x; c < d + rank; c += blockDim.x) {
        if (c < d) {
            out[c] = __bfloat162float(x[c]);
        } else {
            const __nv_bfloat16 v = src[c - d];
            if (ok) e[c - d] = v;
            out[c] = __bfloat162float(v);
        }
    }
}
