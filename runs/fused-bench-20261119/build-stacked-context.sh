#!/usr/bin/env bash
# Docker build context for the stacked 0193-0198 bench image: ~/Workspace/tfref-fusedbench branch fused-bench
# (production source 2039a60 + 0193/0194 + 0195-0198 cherry-picked + a drafter-gate merge fix).
set -euo pipefail
src=${1:-$HOME/Workspace/tfref-fusedbench}
context=$(mktemp -d -p "$HOME/.cache")
trap 'rm -rf "$context"' EXIT
git -C "$src" archive fused-bench | tar -xf - -C "$context"
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
