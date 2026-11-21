#!/usr/bin/env bash
# Rebuild only the optional-prefetch CUDA module with the pinned engine's flags.
set -euo pipefail
source_file=$1
destination=$2
/usr/local/cuda/bin/nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
  -std=c++20 -O3 --fmad=false --ftz=false -gencode=arch=compute_121,code=sm_121 \
  -o "$destination" "$source_file"
/usr/local/cuda/bin/cuobjdump --dump-elf-symbols "$destination" | awk '/tf_ds_l2_prefetch_plan_kernel/ { seen=1 } END { exit !seen }'
sha256sum "$source_file" "$destination"
