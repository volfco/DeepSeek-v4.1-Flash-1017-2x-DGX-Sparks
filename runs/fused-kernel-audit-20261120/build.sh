#!/usr/bin/env bash
# Compile from original and patched engine CUDA source directories in an isolated build directory.
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
    if [[ $version == old ]]; then
        echo '    tf_ds_triton_rmsnorm_row(x,w,dst,d,eps);' >> "$version-torch.cu"
    else
        echo '    if (d == 512) tf_ds_triton_rmsnorm_row<512>(x,w,dst,eps);' >> "$version-torch.cu"
        echo '    else tf_ds_triton_rmsnorm_row<128>(x,w,dst,eps);' >> "$version-torch.cu"
    fi
    echo '}' >> "$version-torch.cu"
    if [[ $version == old ]]; then
        # Diagnostic variant overriding only the old sqrt/div pair. Keep its sum, mean and epsilon rounding.
        python3 - "$version-torch.cu" <<'PYTHON'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
start = s.index("__device__ __forceinline__ void tf_ds_triton_rmsnorm_row")
end = s.index("\n}", start) + 2
helper = s[start:end].replace("tf_ds_triton_rmsnorm_row", "audit_root_only")
old = "rinv_s = __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(v, (float)D), eps)));"
new = """{
            float mean = __fadd_rn(__fdiv_rn(v, (float)D), eps), root, inv;
            asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(mean));
            asm("div.approx.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.f), "f"(root));
            rinv_s = inv;
        }"""
assert old in helper
wrapper = s[s.rindex('extern "C" __global__ void audit_norm'):]
p.write_text(s + helper.replace(old, new) + "\n" + wrapper.replace("audit_norm", "audit_root").replace("tf_ds_triton_rmsnorm_row", "audit_root_only"))
PYTHON
    fi
    "$nvcc_bin" -O3 -arch=sm_121 --fmad=false --ftz=false -cubin "$version-torch.cu" -o "$version-torch.cubin"
    "$nvcc_bin" -O3 -arch=sm_121 --fmad=false --ftz=false -cubin "$src/dsv41_ops.cu" -o "$version-ops.cubin"
done
"$nvcc_bin" -O3 -arch=sm_121 --fmad=false --ftz=false -cubin "$old/torch_ops/movement.cu" -o movement.cubin
