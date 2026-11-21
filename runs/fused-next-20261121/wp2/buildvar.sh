#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_121 --fmad=false --ftz=false -cubin variants.cu -o bin/variants.cubin -Xptxas -v 2>&1 | grep -E "Function|Used|spill" | head -60
