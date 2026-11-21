#!/usr/bin/env bash
# Build the fused-next bench image on the head and ship it to the worker: the tfref-fused-next tree at fused-next
# (prune-six = production's image source, + 0200-0205/0209-0212, + this round's 0213-0253), the same two-stage
# Dockerfile as runs/fused-followup-20261121. Usage: bash build.sh [commit]
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=colum@100.95.49.21
W=colum@10.50.0.11
rev=${1:-fused-next}
sha=$(git -C ~/Workspace/tfref-fused-next rev-parse --short "$rev")
tag=tensorfold-dsv41:bench-fused-next-$sha
src=$(mktemp -d -p "$HOME/.cache")
trap 'rm -rf "$src"' EXIT
git -C ~/Workspace/tfref-fused-next archive "$rev" | tar -xf - -C "$src"
echo "== build $tag ($rev = $(git -C ~/Workspace/tfref-fused-next rev-parse "$rev")) $(date -u +%T)"
bash runs/fused-followup-20261121/build-context.sh "$src" | ssh -o BatchMode=yes $H "docker build --progress=plain -t $tag -"
echo "== ship $tag $(date -u +%T)"
ssh -o BatchMode=yes $H "docker save $tag | ssh -o BatchMode=yes $W docker load"
for r in "$H" "$H ssh -o BatchMode=yes $W"; do ssh -o BatchMode=yes $r "docker run --rm --entrypoint sha256sum $tag /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes"; done
echo "== done $tag $(date -u +%T)"
