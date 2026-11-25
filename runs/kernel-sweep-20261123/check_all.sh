#!/usr/bin/env bash
# check_all.sh <zig/kernels/cuda dir>: check.sh's host syntax check (clang, /home/colum/Workspace/cuda-syntax-stubs)
# over EVERY .cu under the directory, not only check.sh's three dsv41 files, with no error limit. Prints one line an
# error as "<file>: <message> | <the source line, trimmed>" - no line numbers, so two trees compare with diff however
# their lines moved. The stubs leave false positives (types and intrinsics they do not define); the gate is that a
# patched tree prints the same set as the base tree, apart from lines the patch itself is expected to add.
# EXTRA=<header>: force-include it first (stubs-extra/extra.h fills the stock stubs' gaps: the strict pass).
set -uo pipefail
dir=$(cd "${1:?usage: check_all.sh <zig/kernels/cuda dir>}" && pwd)
stubs=/home/colum/Workspace/cuda-syntax-stubs
cd "$dir"
for f in $(find . -name '*.cu' | sort); do
    clang++ -x c++ -std=c++17 -fsyntax-only -ferror-limit=0 -Wno-array-bounds -I "$stubs" -I "$(dirname "$f")" -I . \
        ${EXTRA:+-include "$EXTRA"} \
        -DKERNEL_FILE="\"$dir/$f\"" "$stubs/check.cu" 2>&1 |
        awk -v f="$f" -v d="$dir/" '/error:/ { msg = $0; sub(/^.*error: /, "", msg); getline src; sub(/^[ 0-9]*\| */, "", src); gsub(/[ \t]+/, " ", src); i = index(src, d); if (i) src = substr(src, 1, i - 1) substr(src, i + length(d)); gsub(/:[0-9]+:[0-9]+:/, ":", src); print f ": " msg " | " src }'
done
