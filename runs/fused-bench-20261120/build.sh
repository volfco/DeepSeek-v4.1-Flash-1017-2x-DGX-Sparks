#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
H=colum@100.95.49.21
W=colum@10.50.0.11
tag=tensorfold-dsv41:bench-fused-rework-20261120
echo "== build $tag $(date -u +%T)"; bash runs/fused-bench-20261120/build-context.sh | ssh -o BatchMode=yes $H "docker build --progress=plain -t $tag -"
echo "== ship $tag $(date -u +%T)"; ssh -o BatchMode=yes $H "docker save $tag | ssh -o BatchMode=yes $W docker load"
for r in "$H" "$H ssh -o BatchMode=yes $W"; do ssh -o BatchMode=yes $r "docker run --rm --entrypoint sha256sum $tag /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes"; done
echo "== done $(date -u +%T)"
