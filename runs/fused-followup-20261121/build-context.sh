#!/usr/bin/env bash
# Docker build context for the 0200-0212 bench image: the prune-six tree (production source + the reworked fused
# kernels + 0199) with zig/patches 0200-0212 applied, and nothing else. That base is the image the previous two
# campaigns measured (bench-prune-six-20261120), so every arm differs from the previous fused bundle by these 13
# patches alone -- not by 0182-0190, which the full recipe series carries and production does not.
set -euo pipefail
src=${1:-$HOME/Workspace/tfref-recipe-bench}
context=$(mktemp -d -p "$HOME/.cache")
trap 'rm -rf "$context"' EXIT
tar -C "$src" --exclude='.git' -cf - . | tar -xf - -C "$context"
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
