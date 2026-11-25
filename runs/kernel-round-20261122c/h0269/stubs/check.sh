#!/usr/bin/env bash
# Host syntax check of the hand-written CUDA files in $1 (a zig/kernels/cuda dir) with clang and stub headers.
# Catches C++ mistakes and argument-shape typos, not codegen or PTX validity. See check.cu.
set -uo pipefail
dir=$(cd "${1:?usage: check.sh <zig/kernels/cuda dir>}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
rc=0
for f in "$dir"/dsv41_torch.cu "$dir"/dsv41_ops.cu "$dir"/dsv41_shard.cu; do
    [[ -f $f ]] || continue
    out=$(clang++ -x c++ -std=c++17 -fsyntax-only -Wno-array-bounds -I "$here" \
          -DKERNEL_FILE="\"$f\"" "$here/check.cu" 2>&1)
    if [[ -n $out ]]; then
        printf '%s\n' "$out" | head -20 | sed "s|^|$(basename "$f"): |"
        rc=1
    fi
done
[[ $rc == 0 ]] && echo "cuda syntax check: ok"
exit $rc
