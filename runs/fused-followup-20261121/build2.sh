#!/usr/bin/env bash
# Compile the previous fused bundle (prune-six: production source + 0195/0197/0198/0199) and the 0200-0212 tree
# from their own CUDA sources into one scratch directory, plus three audit kernels that expose the changed
# arithmetic to the host: audit_norm (0197/0202's torch norm), audit_pair (0207's pair softmax) and audit_drop
# (0211's six-slot drop rule). Usage: bash build2.sh <old-engine> <new-engine> <scratch-binaries>
set -euo pipefail
old=$(realpath "$1")
new=$(realpath "$2")
mkdir -p "$3"
cd "$3"
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
for version in old new; do
    src=$old
    [[ $version == new ]] && src=$new
    cp "$src/dsv41_torch.cu" "$version-torch.cu"
    cat >> "$version-torch.cu" <<'CUDA'

extern "C" __global__ void audit_norm(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                      __nv_bfloat16* dst, int d, float eps) {
    x += (long long)blockIdx.x * d;
    dst += (long long)blockIdx.x * d;
CUDA
    # Both trees carry the repaired 0197 norm, templated on the kit's 128/512 layout.
    echo '    if (d == 512) tf_ds_triton_rmsnorm_row<512>(x,w,dst,eps);' >> "$version-torch.cu"
    echo '    else tf_ds_triton_rmsnorm_row<128>(x,w,dst,eps);' >> "$version-torch.cu"
    echo '}' >> "$version-torch.cu"
    # 0207: the ratio-2 pair, one row of (s0, s1, kv0, kv1) a lane.
    cat >> "$version-torch.cu" <<'CUDA'

extern "C" __global__ void audit_pair(const float* s0, const float* s1, const float* kv0, const float* kv1,
                                      float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tf_ds_compress2_pair(s0[i], s1[i], kv0[i], kv1[i]);
}
CUDA
    "$nvcc_bin" -O3 -arch=sm_121 --fmad=false --ftz=false -cubin "$version-torch.cu" -o "$version-torch.cubin"
    cp "$src/dsv41_ops.cu" "$version-ops.cu"
    # 0211: the six-slot trailing-ranks rule, six shares and ids a lane.
    cat >> "$version-ops.cu" <<'CUDA'

extern "C" __global__ void audit_drop(const float* shares, const int* ids, unsigned* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s[6];
        int d[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) { s[j] = shares[i * 6 + j]; d[j] = ids[i * 6 + j]; }
        out[i] = tf_ds_drop_six(s, d);
    }
}
CUDA
    "$nvcc_bin" -O3 -arch=sm_121 --fmad=false --ftz=false -cubin "$version-ops.cu" -o "$version-ops.cubin"
done
"$nvcc_bin" -O3 -arch=sm_121 --fmad=false --ftz=false -cubin "$old/torch_ops/movement.cu" -o movement.cubin
