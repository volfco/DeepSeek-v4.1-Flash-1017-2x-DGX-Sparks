#!/usr/bin/env bash
# Build separately from runtime sampling, from the recipe root on the head.
set -euo pipefail
here=runs/engine-followup/kv
base=tensorfold-dsv41:exl3-tp2-234f9080b326
label=234f9080b326-held-9a5e38c42edd
tag=tensorfold-dsv41:bench-held-9a5e38c42edd
test "$(sha256sum "$here/held-wait.patch" | cut -d' ' -f1)" = 9a5e38c42edd58ff1b4d8e2e982a7d1e06b65bae1c45eb2eb5f163729500a294
mkdir -p "$here/build" "$here/binary"
cp runs/admission-grace/build-01/source.tar.gz "$here/build/source.tar.gz"
cp "$here/held-wait.patch" "$here/build/finish-grace.patch"
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
