#!/usr/bin/env bash
# h0260: tf_ds_conf_pack_kernel's exact source on the CPU against an independent transcription of the served chain.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cu=${CU:-/home/colum/Workspace/tfref-k260/zig/kernels/cuda/dsv41_ops.cu}
python3 "$here/extract.py" "$cu" "// TF_DS_CONF_PACK (conf_pack.zig)" tf_ds_conf_pack_kernel "$here/kernel.inc"
g++ -std=c++17 -O1 -ffp-contract=off -Wall -Wno-unused-function -o "$here/conf_pack_run" "$here/conf_pack_run.cpp"
"$here/conf_pack_run"
