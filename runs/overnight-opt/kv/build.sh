#!/usr/bin/env bash
# Build separately from runtime sampling, from the recipe root on the head.
set -euo pipefail
here=runs/overnight-opt/kv
base=tensorfold-dsv41:exl3-tp2-234f9080b326
label=234f9080b326-held-adapt-9063970bbd02
tag=tensorfold-dsv41:bench-held-adapt-9063970bbd02
cat runs/engine-followup/kv/held-wait.patch runs/engine-followup/adaptive.patch > "$here/combined.patch"
test "$(sha256sum "$here/combined.patch" | cut -d' ' -f1)" = 9063970bbd02a32adf1669f88dc10dc0258b357a340bf08958a74bf206a03c58
mkdir -p "$here/build" "$here/binary"
cp runs/admission-grace/build-01/source.tar.gz "$here/build/source.tar.gz"
cp "$here/combined.patch" "$here/build/finish-grace.patch"
cp runs/admission-grace/build-01/build.Dockerfile "$here/build/Dockerfile"
docker build --progress plain --network none --output "type=local,dest=$here/binary" "$here/build" > "$here/build.log" 2>&1
printf 'FROM %s\nCOPY tensorfold-native /opt/tensorfold/native/bin/tensorfold-native\nLABEL tf.patches="%s"\n' "$base" "$label" > "$here/binary/Dockerfile"
docker build --network none -q -t "$tag" "$here/binary" > "$here/head.image"
tar -C "$here/binary" -cf - Dockerfile tensorfold-native | ssh -o BatchMode=yes 10.50.0.11 "docker build --network none -q -t '$tag' -" > "$here/worker.image"
for place in head worker; do
  check=(docker run --rm --network none --entrypoint sha256sum "$tag" /opt/tensorfold/native/bin/tensorfold-native)
  if [[ "$place" == head ]]; then "${check[@]}"; else ssh -n -o BatchMode=yes 10.50.0.11 "${check[@]}"; fi > "$here/$place.binary"
done
cmp "$here/head.binary" "$here/worker.binary"
cat "$here/head.binary"
