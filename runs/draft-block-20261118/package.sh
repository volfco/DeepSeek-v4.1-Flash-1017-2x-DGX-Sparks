#!/usr/bin/env bash
# Layer the two cross-built executables over the pinned image on both ranks, from the recipe root on the head.
#   runs/draft-block-20261118/package.sh <dir with tensorfold-native, tf-dsv41-lanes, SHA256SUMS>
# The pinned image's kernels, kit mounts and everything else stay as they are; only the two executables change.
# Prints the image tag. Nothing is started or stopped.
set -euo pipefail
dir=$(realpath "$1")
results=runs/draft-block-20261118/package
patch=runs/draft-block-20261118/draft-block.patch
base=tensorfold-dsv41:exl3-tp2-234f9080b326
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
hash=$(sha256sum "$patch" | cut -c 1-12)
tag="tensorfold-dsv41:bench-draft-block-$hash"
mkdir -p "$results"
(cd "$dir" && sha256sum -c SHA256SUMS)
cat > "$dir/Dockerfile" <<DOCKER
FROM $base
COPY tensorfold-native /opt/tensorfold/native/bin/tensorfold-native
COPY tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-lanes
LABEL tf.patches="234f9080b326-draft-block-$hash"
DOCKER
docker build --network none -q -t "$tag" "$dir" > "$results/head.image"
tar -C "$dir" -cf - Dockerfile tensorfold-native tf-dsv41-lanes | ssh -o BatchMode=yes "$worker" "docker build --network none -q -t '$tag' -" > "$results/worker.image"
for place in head worker; do
  cmd=(docker run --rm --network none --entrypoint sha256sum "$tag" /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes)
  if [[ "$place" == head ]]; then "${cmd[@]}"; else ssh -n -o BatchMode=yes "$worker" "${cmd[@]}"; fi > "$results/$place.binary"
done
cmp "$results/head.binary" "$results/worker.binary"
awk '{f=$2; sub(/^.*\//, "", f); print $1 "  " f}' "$results/head.binary" | sort > "$results/executables.sha256"
sort "$dir/SHA256SUMS" | cmp - "$results/executables.sha256"
printf '%s\n' "$tag" | tee "$results/tag.txt"
