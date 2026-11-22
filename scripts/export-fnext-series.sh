#!/usr/bin/env bash
# Export the fused-next round (`0213`–`0258`) into zig/patches/ as NNNN-<slug>.patch files, numbered from each
# commit's own subject (the round has gaps: 0218–0229 and 0232–0239 are unused), then prove the chain: a clean
# checkout of TF_REF with every zig/patches/*.patch applied in filename order must equal the branch.
#
# Why not scripts/port-patches.sh: that tool applies every hand-written patch before every exported one, and this
# recipe's tree interleaves them (exported 0129–0171, hand-written 0182–0211, then this round), so it cannot reach
# the base. Ordering here is by filename alone, exactly as zig/Dockerfile applies them.
#
# Usage: scripts/export-fnext-series.sh --engine DIR --base REF --branch REF [--dry-run]
#   --engine DIR   the engine repository holding the rebased commits (default ~/Workspace/tfref-fnext-rebase)
#   --base REF     the tree the recipe's zig/patches already reach (default c3b3859 = TF_REF + 0001–0211)
#   --branch REF   the rebased tip to export (e.g. the worktree's HEAD)
#   --dry-run      emit into a temporary directory, verify, write nothing
# Env: TF_REF (default: scripts/config.sh's pin)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
log() { printf '[export-fnext] %s\n' "$*"; }
die() { printf '[export-fnext] ERROR: %s\n' "$*" >&2; exit 1; }

engine="$HOME/Workspace/tfref-fnext-rebase"; base="c3b3859"; branch=""; dry=0
while (( $# )); do
  case "$1" in
    --engine) engine=${2:?}; shift ;;
    --base) base=${2:?}; shift ;;
    --branch) branch=${2:?}; shift ;;
    --dry-run) dry=1 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done
[[ -n "$branch" ]] || die "--branch names the rebased tip to export"
git -C "$engine" rev-parse --git-dir >/dev/null 2>&1 || die "$engine is not a git repository"
base_sha=$(git -C "$engine" rev-parse --verify -q "$base^{commit}") || die "no $base in $engine"
branch_sha=$(git -C "$engine" rev-parse --verify -q "$branch^{commit}") || die "no $branch in $engine"
git -C "$engine" merge-base --is-ancestor "$base_sha" "$branch_sha" || die "$base is not an ancestor of $branch"
TF_REF=${TF_REF:-$(sed -n 's/^TF_REF="${TF_REF:-\([^}]*\)}".*/\1/p' "$root/scripts/config.sh" | head -1)}
[[ "$TF_REF" =~ ^[0-9a-f]{40}$ ]] || die "TF_REF is not a full commit hash: '$TF_REF'"
git -C "$engine" cat-file -e "$TF_REF^{commit}" 2>/dev/null || die "TF_REF $TF_REF is not in $engine"

slug() { tr '[:upper:]' '[:lower:]' <<<"$1" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' | cut -c1-48 | sed -E 's/-+$//'; }
out=$(mktemp -d "${TMPDIR:-/tmp}/export-fnext.XXXXXX")
trap 'rm -rf -- "$out"' EXIT

last=0; n=0
for c in $(git -C "$engine" rev-list --first-parent --reverse "$base_sha..$branch_sha"); do
  s=$(git -C "$engine" log -1 --format=%s "$c")
  [[ "$s" =~ ^0([0-9]{3})\  ]] || die "commit ${c:0:12} has no leading patch number: '$s'"
  num=$((10#${BASH_REMATCH[1]}))
  (( num > last )) || die "commit ${c:0:12} is numbered $num, not above $last: '$s'"
  last=$num
  f=$(printf "%04d-%s.patch" "$num" "$(slug "${s#[0-9][0-9][0-9][0-9] }")")
  {
    printf '# export-fnext-series.sh: %s @ %s over %s @ %s (commit %s)\n' "$branch" "${branch_sha:0:12}" "$base" "${base_sha:0:12}" "${c:0:12}"
    printf '# %s\n' "$s"
    printf '# Applied by zig/Dockerfile with git apply in filename order; regenerate with scripts/export-fnext-series.sh, do not edit.\n'
    git -C "$engine" diff --binary --full-index --no-renames --no-color --no-ext-diff \
      --src-prefix=a/ --dst-prefix=b/ "$c^1" "$c" -- . ':(exclude)PORT.md' ':(exclude)docs/port'
  } > "$out/$f"
  grep -q '^diff --git ' "$out/$f" || die "$f has no diff"
  n=$((n + 1))
done
(( n > 0 )) || die "$base..$branch changes nothing"
log "emitted $n patches: $(ls "$out" | head -1) … $(ls "$out" | tail -1)"

git clone -q --no-checkout --shared "$engine" "$out/tf"
git -C "$out/tf" -c advice.detachedHead=false checkout -q --detach "$TF_REF"
for p in "$root"/zig/patches/*.patch; do
  ( cd "$out/tf" && git apply --check "$p" && git apply "$p" ) 2>"$out/err" ||
    { sed 's/^/    /' "$out/err" >&2; die "$(basename "$p") does not apply on a clean TF_REF in filename order"; }
done
for p in "$out"/*.patch; do
  ( cd "$out/tf" && git apply --check "$p" && git apply "$p" ) 2>"$out/err" ||
    { sed 's/^/    /' "$out/err" >&2; die "$(basename "$p") does not apply after the recipe's patches"; }
  log "  $(basename "$p") applies ($(grep -c '^diff --git ' "$p") files)"
done
git -C "$out/tf" add -A -f >/dev/null
if ! git -C "$out/tf" diff --cached --quiet "$branch_sha" -- . ':(exclude)PORT.md' ':(exclude)docs/port'; then
  git -C "$out/tf" diff --cached --stat "$branch_sha" -- . ':(exclude)PORT.md' ':(exclude)docs/port' | tail -15 >&2
  die "TF_REF + zig/patches does not reproduce $branch (stat above)"
fi
log "TF_REF + zig/patches = $branch (${branch_sha:0:12}): verified"
if (( dry )); then log "dry run: nothing written ($out)"; trap - EXIT; rm -rf -- "$out"; exit 0; fi
cp -f "$out"/*.patch "$root/zig/patches/"
log "wrote $n patches into zig/patches/"
