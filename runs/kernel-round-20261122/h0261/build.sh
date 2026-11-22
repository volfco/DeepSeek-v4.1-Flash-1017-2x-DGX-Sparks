#!/usr/bin/env bash
# h0261: tf_ds_topk_select / tf_ds_topk_merge / tf_ds_draw_rows_topk kernels' exact source on the CPU (std::thread a
# CUDA thread, std::barrier for __syncthreads and the warp shuffles), against independent references and against the
# untouched full-row tf_ds_draw_rows_kernel (also extracted and emulated) on the concatenated rows.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cu=${CU:-/home/colum/Workspace/tfref-k260/zig/kernels/cuda/dsv41_ops.cu}
python3 "$here/extract.py" "$cu" "$here/kernels.inc" \
  "__device__ unsigned tf_ds_block_argmax_v(const float* row, int n, float* out_v) {" "// logits [rows, n] fp32 (row stride ld), out [rows] u32; a row a block" \
  "__device__ __forceinline__ uint64_t tf_ds_mix64(" "// grid (ceil(width / blockDim), streams)" \
  "struct TfDsDrawRow {" "// 0179 (TF_DS_EXPERT_HOT_CACHE=hot)"
sed 's/extern "C" //' "$here/kernels.inc" > "$here/kernels_host.inc"
g++ -std=c++20 -O2 -ffp-contract=off -pthread -Wall -Wno-unused-function -Wno-unused-variable -o "$here/topk_run" "$here/topk_run.cpp"
"$here/topk_run"
