#!/usr/bin/env bash
# Build the fused-kernel bench cubins on the head, with the engine's own flags (never fast math):
#   build.sh OLD_CUDA_DIR NEW_CUDA_DIR OUT
# OLD_CUDA_DIR / NEW_CUDA_DIR each hold dsv41_ops.cu and dsv41_torch.cu (e.g. two engine trees). OUT receives
# {old,new}-{ops,torch}.cubin plus floor.cubin (empty_kernel and copy_kernel: the launch floor and the bandwidth
# probe). Run it on the head; the workstation has no nvcc.
set -euo pipefail
old=$(realpath "$1")
new=$(realpath "$2")
out=$(realpath -m "$3")
mkdir -p "$out"
cd "$out"
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
flags=(-O3 -arch=sm_121 --fmad=false --ftz=false)
for version in old new; do
    src=$old
    [[ $version == new ]] && src=$new
    for unit in ops torch; do
        "$nvcc_bin" "${flags[@]}" -cubin "$src/dsv41_$unit.cu" -o "$version-$unit.cubin"
    done
done
cat > floor.cu <<'CUDA'
// The launch floor's node: an empty kernel at any launch shape.
extern "C" __global__ void empty_kernel() {}

// The bandwidth probe's node: a 16-byte copy, grid-stride.
extern "C" __global__ void copy_kernel(const uint4* src, uint4* dst, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}
CUDA
"$nvcc_bin" "${flags[@]}" -cubin floor.cu -o floor.cubin
echo "built $out"
