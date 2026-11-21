#!/usr/bin/env bash
# Run from the recipe root on the head after the live KV workloads have finished.
set -euo pipefail
here=runs/engine-followup
base=tensorfold-dsv41:exl3-tp2-234f9080b326
tag=tensorfold-dsv41:bench-adaptive-f7e3ead1eb9f
mkdir -p "$here/build" "$here/binary"
test "$(sha256sum "$here/adaptive.patch" | cut -d' ' -f1)" = f7e3ead1eb9f790832960b2d76435c8f47fac2df09aec7ea2bed8db21bb215b6
cp runs/admission-grace/build-01/source.tar.gz "$here/build/source.tar.gz"
cp "$here/adaptive.patch" "$here/build/finish-grace.patch"
cp runs/admission-grace/build-01/build.Dockerfile "$here/build/Dockerfile"
docker build --progress plain --network none --output "type=local,dest=$here/binary" "$here/build" > "$here/build.log" 2>&1
printf 'FROM %s\nCOPY tensorfold-native /opt/tensorfold/native/bin/tensorfold-native\nLABEL tf.patches="234f9080b326-adaptive-f7e3ead1eb9f"\n' "$base" > "$here/binary/Dockerfile"
docker build --network none -q -t "$tag" "$here/binary" > "$here/head.image"
tar -C "$here/binary" -cf - Dockerfile tensorfold-native | ssh -o BatchMode=yes 10.50.0.11 "docker build --network none -q -t '$tag' -" > "$here/worker.image"
docker run --rm --network none --entrypoint sha256sum "$tag" /opt/tensorfold/native/bin/tensorfold-native > "$here/head.binary"
ssh -n -o BatchMode=yes 10.50.0.11 "docker run --rm --network none --entrypoint sha256sum '$tag' /opt/tensorfold/native/bin/tensorfold-native" > "$here/worker.binary"
cmp "$here/head.binary" "$here/worker.binary"
cat "$here/head.binary"
