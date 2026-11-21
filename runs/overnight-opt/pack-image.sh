#!/usr/bin/env bash
# Run on the head with a tar archive containing only the two cross-built executables and Dockerfile.
set -euo pipefail
package=$1
digest=$2
[[ $digest =~ ^[a-f0-9]{12}$ ]]
tag="tensorfold-dsv41:bench-opt-$digest"
proof="runs/overnight-opt/image-$digest"
mkdir -p "$proof"
docker build --network none -q -t "$tag" - < "$package" > "$proof/head.image"
ssh -o BatchMode=yes 10.50.0.11 "docker build --network none -q -t '$tag' -" < "$package" > "$proof/worker.image"
for rank in head worker; do
    cmd=()
    if [[ $rank = worker ]]; then cmd=(ssh -n -o BatchMode=yes 10.50.0.11); fi
    "${cmd[@]}" docker run --rm --network none --entrypoint sha256sum "$tag" /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes > "$proof/$rank.sha256"
done
cmp "$proof/head.sha256" "$proof/worker.sha256"
cat "$proof/head.sha256"
echo "$tag"
