#!/usr/bin/env bash
# The zstd recompression check (docs/benchmark-gaps-and-moe-review.md, section 5's "a lossless recompression of the
# trellis" row): one slice of the rank cache through zstd -19 settles whether losslessly recompressing the trellis
# bytes is worth designing - "under 3% saving: close". Lossless is checked, not assumed: every level's output is
# decompressed and compared byte for byte.
# Read-only against the rank cache; scratch in mktemp -d, removed on exit. Not run by us: the operator (or the
# benchmark engine) runs it on the head, where the rank cache file lives (`just zstd-check`).
# Usage: check.sh [RANK_CACHE] [SIZE_MIB=1024] [OFFSET_MIB=middle of the file]
#   RANK_CACHE  the weight file, or a directory (searched for its biggest file); empty: look under
#               ~/.cache/dsv41-tensorfold/rank-cache-* (the recipe's default)
set -euo pipefail
arg=${1:-}
size=${2:-1024}
if [[ -z "$arg" ]]; then
  arg=$(ls -S "$HOME"/.cache/dsv41-tensorfold/rank-cache-*/* 2>/dev/null | head -1 || true)
fi
if [[ -d "$arg" ]]; then
  arg=$(ls -S "$arg"/* 2>/dev/null | head -1 || true)
fi
[[ -f "$arg" ]] || { echo "zstd-check: no rank cache file at '${1:-<default>}'; pass its path" >&2; exit 1; }
command -v zstd >/dev/null || { echo "zstd-check: zstd is not installed here" >&2; exit 1; }

total=$(stat -c%s "$arg")
off=${3:-$(( (total - size * 1048576) / 2 / 1048576 ))}
(( off >= 0 )) || off=0
tmp=$(mktemp -d "${TMPDIR:-/tmp}/zstd-check.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
echo "rank cache: $arg ($((total / 1048576)) MiB); slice: ${size} MiB at ${off} MiB"
dd if="$arg" of="$tmp/slice" bs=1M count="$size" skip="$off" status=none
orig=$(stat -c%s "$tmp/slice")
(( orig > 0 )) || { echo "zstd-check: empty slice" >&2; exit 1; }

printf '%-6s %9s %8s %9s %9s %s\n' level zst_MiB saved enc_MiBs dec_MiBs lossless
best=0
for lvl in 1 6 19; do
  t0=$(date +%s.%N)
  zstd -"$lvl" -q -f -o "$tmp/slice.zst" "$tmp/slice"
  t1=$(date +%s.%N)
  zstd -q -d -f -o "$tmp/slice.out" "$tmp/slice.zst"
  t2=$(date +%s.%N)
  cmp -s "$tmp/slice" "$tmp/slice.out" || { echo "zstd-check: level -$lvl is NOT lossless; stop" >&2; exit 1; }
  c=$(stat -c%s "$tmp/slice.zst")
  saved=$(awk -v o="$orig" -v c="$c" 'BEGIN{printf "%.2f", (1 - c / o) * 100}')
  enc=$(awk -v o="$orig" -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", o / 1048576 / (b - a)}')
  dec=$(awk -v o="$orig" -v a="$t1" -v b="$t2" 'BEGIN{printf "%.1f", o / 1048576 / (b - a)}')
  printf '%-6s %9s %7s%% %9s %9s %s\n' "-$lvl" "$((c / 1048576))" "$saved" "$enc" "$dec" yes
  best=$(awk -v x="$saved" -v y="$best" 'BEGIN{print (x + 0 > y + 0) ? x : y}')
done
awk -v b="$best" 'BEGIN{
  if (b + 0 >= 3) print "VERDICT: best saving " b "% at or above 3%: a lossless trellis recompression is worth designing (it is fewer bytes on the memory bus every round)"
  else print "VERDICT: best saving " b "% under 3%: close (docs/benchmark-gaps-and-moe-review.md, section 5)"
}'
