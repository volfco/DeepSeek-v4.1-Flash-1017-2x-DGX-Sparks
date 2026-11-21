#!/usr/bin/env bash
set -euo pipefail
# Supply the benchmark-only source tree with candidate.patch and tail-start.patch applied.
src_dir=$(realpath "${1:?expected patched source root}")
here=$(cd "$(dirname "$0")" && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
cp -a "$src_dir/zig" "$work_dir/zig"
src="$work_dir/zig/src"
cat "$here/stream-test.zig" >> "$src/families/dsv41/model.zig"
zig_bin=${ZIG:-zig}
for fixture in 'lanes:identical prompt follower snapshots' 'lanes:TF_DS_TAIL_BATCH: an end state' 'model:prompt end copies'; do
    "$zig_bin" test -lc -O ReleaseSafe --dep core --dep cuda --dep engine_api --dep lanes \
        "-Mroot=$src/families/dsv41/${fixture%%:*}.zig" "-Mcore=$src/core/root.zig" \
        "-Mcuda=$src/cuda/root.zig" "-Mengine_api=$src/core/engine_api.zig" \
        "-Mlanes=$src/core/lanes/lanes.zig" --test-filter "${fixture#*:}" \
        --cache-dir "$work_dir/cache" "${@:2}"
done
