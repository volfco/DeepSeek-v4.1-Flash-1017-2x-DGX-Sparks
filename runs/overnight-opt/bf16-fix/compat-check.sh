#!/usr/bin/env bash
set -euo pipefail
baseline=$(realpath "${1:?supply the unchanged odds or fallback source}")
output=$(realpath "${2:?supply a writable directory for the four logs}")
here=$(cd "$(dirname "$0")" && pwd)
working=$(mktemp -d)
trap 'rm -rf -- "$working"' EXIT
cp -a "$baseline/zig" "$working/zig"
src=$working/zig/src
cat "$here/compat-loopback.zig" >> "$src/families/dsv41/rank_wire.zig"
cat "$here/compat-switches.zig" >> "$src/families/dsv41/lanes.zig"
for version in original corrected; do
    if [[ $version = corrected ]]; then patch -s -d "$working" -p1 < "$here/compat.patch"; fi
    for suite in 'rank_wire:BF16 pair compatibility' 'lanes:effective BF16 mode'; do
        unit=${suite%%:*}
        "${ZIG:-zig}" test -lc -O ReleaseSafe --dep core --dep cuda --dep engine_api --dep lanes \
            "-Mroot=$src/families/dsv41/$unit.zig" "-Mcore=$src/core/root.zig" \
            "-Mcuda=$src/cuda/root.zig" "-Mengine_api=$src/core/engine_api.zig" "-Mlanes=$src/core/lanes/lanes.zig" \
            --test-filter "${suite#*:}" --cache-dir /tmp/dsv41-bf16-host-zig-cache > "$output/$version-$unit.log" 2>&1
    done
done
awk '/^mode=0 /' "$output/original-rank_wire.log" > "$working/off-original"
awk '/^mode=0 /' "$output/corrected-rank_wire.log" > "$working/off-corrected"
[[ $(wc -l < "$working/off-original") = 3 ]]
cmp "$working/off-original" "$working/off-corrected"
