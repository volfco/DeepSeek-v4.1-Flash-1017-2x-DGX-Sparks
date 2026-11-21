#!/usr/bin/env bash
# Promote the gated draft-block image into the recipe on the head, from the recipe root. Starts and stops nothing.
#   runs/draft-block-20261118/promote.sh
# 1. the final patch joins zig/patches as 0172 (it applies last, onto the pinned series), which gives a new image hash;
# 2. the gated executables (the bench image the gate arm ran) are tagged as that hash's image, with the patch beside the
#    others in /opt/tf-patches and the hash as the tf.patches label, on both ranks, layer for layer the same;
# 3. checks: the label is the recipe's hash, both ranks hold the same image, the executables are the gated ones, and
#    the image's weight-file tool is the pinned image's (setup names the weight files with it).
# The recipe then starts this image by itself: no IMAGE or PREPARE setting is needed.
set -euo pipefail
D=runs/draft-block-20261118
gated=$(cat $D/package/tag.txt)
pinned=tensorfold-dsv41:exl3-tp2-234f9080b326
name=0172-dsv41-tf-ds-draft-block.patch
out=$D/promote; mkdir -p "$out"
before=$(bash -c 'source ./scripts/config.sh; image_hash')
[[ "$before" == 234f9080b326 ]] || { echo "the head's recipe is not the pinned series (hash $before)"; exit 2; }
cp $D/draft-block.patch zig/patches/$name
hash=$(bash -c 'source ./scripts/config.sh; image_hash')
image=tensorfold-dsv41:exl3-tp2-$hash
echo "recipe hash $before -> $hash; image $image" | tee "$out/hash.txt"
ctx=$(mktemp -d); trap 'rm -rf -- "$ctx"' EXIT
cp zig/patches/$name "$ctx/"
printf 'FROM %s\nCOPY %s /opt/tf-patches/%s\nLABEL tf.patches="%s"\n' "$gated" "$name" "$name" "$hash" > "$ctx/Dockerfile"
docker build --network none -q -t "$image" "$ctx" > "$out/head.image"
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
docker save "$image" | ssh -o BatchMode=yes "$worker" 'docker load' > "$out/worker.load" 2>&1
# checks
bash -c "source ./scripts/config.sh; source ./scripts/nodes.sh
  echo \"label head:   \$(docker image inspect -f '{{index .Config.Labels \"tf.patches\"}}' $image)\"
  echo \"label worker: \$(worker 1 \"docker image inspect -f '{{index .Config.Labels \\\"tf.patches\\\"}}' $image\")\"
  echo \"ident head:   \$(image_ident $image)\"
  echo \"ident worker: \$(worker_image_ident 1 $image)\"
  echo \"default IMAGE: \$IMAGE\"" | tee "$out/checks.txt"
grep -q "label head:   $hash" "$out/checks.txt" && grep -q "label worker: $hash" "$out/checks.txt"
[[ "$(grep 'ident head' "$out/checks.txt" | awk '{print $3}')" == "$(grep 'ident worker' "$out/checks.txt" | awk '{print $3}')" ]]
grep -q "default IMAGE: $image" "$out/checks.txt"
sums() { docker run --rm --network none --entrypoint sha256sum "$1" /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-rank-cache; }
sums "$image" > "$out/promoted.sha256"; sums "$gated" > "$out/gated.sha256"; sums "$pinned" > "$out/pinned.sha256"
ssh -n -o BatchMode=yes "$worker" "docker run --rm --network none --entrypoint sha256sum '$image' /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-rank-cache" > "$out/promoted-worker.sha256"
cmp "$out/promoted.sha256" "$out/gated.sha256"; cmp "$out/promoted.sha256" "$out/promoted-worker.sha256"
[[ "$(grep rank-cache "$out/promoted.sha256" | cut -d' ' -f1)" == "$(grep rank-cache "$out/pinned.sha256" | cut -d' ' -f1)" ]]
awk '{f=$2; sub(/^.*\//, "", f); print $1 "  " f}' "$out/promoted.sha256" | grep -v rank-cache | sort | cmp - <(sort $D/executables.sha256)
docker run --rm --network none --entrypoint sh "$image" -c 'ls /opt/tf-patches/*.patch | wc -l; ls /opt/tf-patches | tail -2' | tee "$out/patches-in-image.txt"
echo "$image" > "$out/tag.txt"
echo "promoted: $image"
