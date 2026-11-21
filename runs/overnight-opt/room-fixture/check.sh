#!/usr/bin/env bash
# Clean 0125 or 0127 input. All fixture edits and compiler output stay in a new temporary directory.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: ZIG=/path/to/zig check.sh /path/to/clean-engine-source" >&2; exit 2; fi
source_tree="$(realpath "$1")"
patch_path="$(cd "$(dirname "$0")/.." && pwd)/room-fixture.patch"
work="$(mktemp -d)"
cp -a "$source_tree/zig" "$work/"
patch --batch --fuzz=0 -d "$work" -p1 < "$patch_path"
zig="${ZIG:-zig}"
for mode in ReleaseSafe Debug; do
    "$zig" test -O "$mode" -lc --dep lanes -Mroot="$work/zig/src/core/engine_api.zig" \
        -O "$mode" -Mlanes="$work/zig/src/core/lanes/lanes.zig" \
        --cache-dir "$work/.zig-cache" --global-cache-dir "${ZIG_GLOBAL_CACHE_DIR:-$work/.global-cache}" \
        -femit-bin="$work/engine-api-$mode"
done
