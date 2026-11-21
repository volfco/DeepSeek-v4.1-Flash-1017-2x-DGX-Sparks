#!/usr/bin/env bash
# Run from the recipe root on the head, with the two cross-built executables
# and their local SHA256SUMS placed in the same private staging directory.
set -euo pipefail
case "$1" in
    committed)
        name=committed
        patch=runs/overnight-opt/kv/committed.patch
        results=runs/overnight-opt/kv/committed-build
        ;;
    eligible)
        name=eligible
        patch=runs/overnight-opt/kv/eligible.patch
        results=runs/overnight-opt/kv/eligible-build
        ;;
    prune)
        name=prune-report
        patch=runs/overnight-opt/prune/marker.patch
        results=runs/overnight-opt/prune/cross-build
        ;;
    *) exit 2 ;;
esac
dir=$(realpath "$2")
hash=$(sha256sum "$patch" | cut -c 1-12)
tag="tensorfold-dsv41:bench-$name-$hash"
mkdir -p "$results"
(cd "$dir" && sha256sum -c SHA256SUMS)
cat > "$dir/Dockerfile" <<DOCKER
FROM tensorfold-dsv41:exl3-tp2-234f9080b326
COPY tensorfold-native /opt/tensorfold/native/bin/tensorfold-native
COPY tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-lanes
LABEL tf.patches="234f9080b326-$name-$hash"
DOCKER
docker build --network none -q -t "$tag" "$dir" > "$results/head.image"
tar -C "$dir" -cf - Dockerfile tensorfold-native tf-dsv41-lanes | ssh -o BatchMode=yes 10.50.0.11 "docker build --network none -q -t '$tag' -" > "$results/worker.image"
for place in head worker; do
    cmd=(docker run --rm --network none --entrypoint sha256sum "$tag" /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes)
    if [[ "$place" == head ]]; then "${cmd[@]}"; else ssh -n -o BatchMode=yes 10.50.0.11 "${cmd[@]}"; fi > "$results/$place.binary"
done
cmp "$results/head.binary" "$results/worker.binary"
awk '{f=$2;sub(/^.*\//,"",f);print $1 "  " f}' "$results/head.binary" | sort > "$results/executables.sha256"
sort "$dir/SHA256SUMS" | cmp - "$results/executables.sha256"
printf '%s\n' "$tag"
