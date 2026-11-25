#!/usr/bin/env bash
# h0266: the 0197 decode compressor and the TF_DS_COMPRESS2_PAIR_ONCE variant's exact source on the CPU (std::thread a
# CUDA thread, barriers for __syncthreads / warp shuffles / ballots), against each other and an independent per-row
# reference. Re-run after kernel edits: the .inc is re-extracted from the worktree every time.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cu=${CU:-/home/colum/Workspace/tfref-k0266/zig/kernels/cuda/dsv41_torch.cu}
python3 "$here/extract.py" "$cu" "$here/kernels.inc"
sed 's/extern "C" //' "$here/kernels.inc" > "$here/kernels_host.inc"
g++ -std=c++20 -O2 -ffp-contract=off -pthread -Wall -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-variable \
    -o "$here/compress2_pair_run" "$here/harness.cpp"
"$here/compress2_pair_run"
