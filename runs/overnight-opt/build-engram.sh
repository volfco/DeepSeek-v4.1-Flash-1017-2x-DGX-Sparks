#!/usr/bin/env bash
set -euo pipefail
dir=runs/overnight-opt
base=tensorfold-dsv41:exl3-tp2-234f9080b326
patches=("$dir/engram-fixes.patch" "$dir/aligned-index.patch")
if [[ ${1:-} = stages ]]; then patches+=("$dir/resume-fixes.patch" "$dir/phase-windows.patch"); fi
hash=$(cat "${patches[@]}" | sha256sum | cut -c1-12)
tag="tensorfold-dsv41:bench-engram-$hash"
work="$dir/build-engram-$hash"
mkdir -p "$work/source" "$work/binary"
cp runs/admission-grace/build-01/source.tar.gz "$work/source"
cp runs/admission-grace/build-01/build.Dockerfile "$work/source/Dockerfile"
cat "${patches[@]}" > "$work/source/finish-grace.patch"
docker build --progress plain --network none --output "type=local,dest=$work/binary" "$work/source" > "$work/build.log" 2>&1
printf 'FROM %s\nCOPY tensorfold-native /opt/tensorfold/native/bin/tensorfold-native\n' "$base" > "$work/binary/Dockerfile"
docker build --network none -q -t "$tag" "$work/binary" > "$work/head.image"
tar -C "$work/binary" -cf - Dockerfile tensorfold-native | ssh -o BatchMode=yes 10.50.0.11 "docker build --network none -q -t '$tag' -" > "$work/worker.image"
for rank in head worker; do
    cmd=()
    if [[ $rank = worker ]]; then cmd=(ssh -n -o BatchMode=yes 10.50.0.11); fi
    "${cmd[@]}" docker run --rm --network none --entrypoint sha256sum "$tag" /opt/tensorfold/native/bin/tensorfold-native > "$work/$rank.binary"
done
cmp "$work/head.binary" "$work/worker.binary"
echo "$tag"
