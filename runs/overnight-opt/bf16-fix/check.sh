#!/usr/bin/env bash
set -euo pipefail
# No CUDA context, board, or network connection is used by this test.
source_root=$(realpath "${1:?provide the benchmark-only source root}")
report_dir=$(cd "$(dirname "$0")" && pwd)
manifest=${2:-$report_dir/../engineering02/bf16-partials/diagnostics/aot.json}
actual_hash=$(sha256sum "$manifest" | cut -d' ' -f1)
[[ $actual_hash == 29dccbcfdfc327a2ca83f40eb88c2bad83f9f61128e0cd27c29a1243d6fcd603 ]]
fixture_root=$(mktemp -d)
trap 'rm -rf -- "$fixture_root"' EXIT
cp -a "$source_root/zig/src" "$fixture_root/src"
src=$fixture_root/src
cp "$report_dir/selector-test.zig" "$src/families/dsv41/selector-test.zig"
cp "$manifest" "$src/families/dsv41/kit-bf16-diagnostic.json"
"${ZIG:-zig}" test -lc -O ReleaseSafe --dep cuda --dep core --dep engine_api --dep lanes \
    "-Mroot=$src/families/dsv41/selector-test.zig" "-Mcuda=$src/cuda/root.zig" \
    "-Mcore=$src/core/root.zig" "-Mengine_api=$src/core/engine_api.zig" "-Mlanes=$src/core/lanes/lanes.zig" \
    --test-filter 'deployed TP2 wo_b kernel coverage' --cache-dir /tmp/dsv41-bf16-host-zig-cache
