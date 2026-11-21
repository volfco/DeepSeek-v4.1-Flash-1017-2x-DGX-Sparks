#!/usr/bin/env bash
# Write a Docker build context to stdout from the exact production source.
# The retest runner expects tensorfold-dsv41:bench-fused-route-0193-0194-fixed
# on both ranks.
set -euo pipefail
recipe=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_repo=${1:-"$recipe/../tfref-draftblock"}
context=$(mktemp -d -p "$HOME/.cache")
trap 'rm -rf "$context"' EXIT

git -C "$source_repo" archive 2039a60 | tar -xf - -C "$context"
# Reproduce the historical A/B image from the archived patches used in that run.
patch="$recipe/runs/fused-bench-20261119/stacked-series/0001-0193-0194-recipe-patches-fixed-versions.patch"
(cd "$context" && git apply --check "$patch" && git apply "$patch")

cat > "$context/Dockerfile" <<'DOCKERFILE'
FROM tensorfold-dsv41:exl3-tp2-ea522d353626 AS build
COPY . /tensorfold
WORKDIR /tensorfold
RUN /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast --prefix /stage -j4 install native

FROM tensorfold-dsv41:exl3-tp2-ea522d353626
COPY --from=build /stage/native/bin/tensorfold-native /opt/tensorfold/native/bin/tensorfold-native
COPY --from=build /stage/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-lanes
DOCKERFILE
tar -C "$context" -cf - .
