#!/usr/bin/env bash
# Drafter D1 (ursuciprian/DeepSeek-V4.1-Flash-1017-DSpark-D1 @ ca1a99a8) laid over production's checkpoint in a SIBLING
# data dir (<DATA_SUB>-d1); production's dir, index and weight files are never touched. From the recipe root on the head:
#   bash runs/d1-20261123/build.sh links    # checkpoint hard links + overlay + remapped index + served folder (light)
#   bash runs/d1-20261123/build.sh caches   # both ranks' weight files (~106 GB each; heavy: run with the engine idle)
# Served with DATA_SUFFIX=-d1 (scripts/config.sh appends it to DATA_SUB and WORKER_CACHE_DIR).
set -euo pipefail
source scripts/config.sh >/dev/null
step=${1:?links|caches}
SRC_SUB=$DATA_SUB; [[ "$SRC_SUB" != *-d1 ]] || { echo "run without DATA_SUFFIX"; exit 1; }
SUB="$SRC_SUB-d1"; SRC="$HF_CACHE/$SRC_SUB"; D="$HF_CACHE/$SUB"
REV=ca1a99a8f07e4ec45addc71d77f5fc73ee3b1c5f; OV=ov0-overlay-000.safetensors
IMAGE=$(grep -m1 '^IMAGE=' .env | cut -d= -f2-)
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
in_hf() { docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HF_CACHE:/hf" "$@"; }
if [[ $step == links ]]; then
  mkdir -p "$D"
  if [[ ! -f "$D/checkpoint/.d1-ready" ]]; then
    rm -rf "$D/checkpoint.new"; cp -al "$SRC/checkpoint" "$D/checkpoint.new"
    rm -f "$D/checkpoint.new/model.safetensors.index.json"   # a hard link to production's index: replace, never edit
    curl -sfL -o "$D/checkpoint.new/$OV" "https://huggingface.co/ursuciprian/DeepSeek-V4.1-Flash-1017-DSpark-D1/resolve/$REV/$OV"
    echo "85bd27f53eade78b4e81ef317668bc94aaa9c71992e0b795fe1e7793a358bedc  $D/checkpoint.new/$OV" | sha256sum -c -
    python3 -I - "$SRC/checkpoint/model.safetensors.index.json" "$D/checkpoint.new/$OV" "$D/checkpoint.new/model.safetensors.index.json" <<'PY'
import json, struct, sys
src, ov, dst = sys.argv[1:]
idx = json.load(open(src)); f = open(ov, "rb"); h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0])); h.pop("__metadata__", None)
miss = [k for k in h if k not in idx["weight_map"] or not k.startswith("mtp.")]
assert not miss, miss
for k in h: idx["weight_map"][k] = ov.rsplit("/", 1)[1]
json.dump(idx, open(dst, "w"), indent=2); print("remapped", len(h), "mtp tensors")
PY
    touch "$D/checkpoint.new/.d1-ready"; mv "$D/checkpoint.new" "$D/checkpoint"
  fi
  cp -n "$SRC/dsv41_token_map.json" "$D/"
  V="$D/model"; rm -rf "$V.new"; mkdir -p "$V.new"
  for f in "$D/checkpoint"/*; do ln -s "$(realpath -s --relative-to="$V.new" "$f")" "$V.new/"; done
  rm -f "$V.new/chat_template.jinja"; cp model/chat_template.jinja "$V.new/chat_template.jinja"
  rm -rf "$V"; mv "$V.new" "$V"; mkdir -p "$D/rank-cache"
  ls -la "$D" "$D/checkpoint" | head -12
elif [[ $step == caches ]]; then
  C="/hf/$SUB/checkpoint"
  want=$(in_hf --entrypoint bash "$IMAGE" -c "tf-dsv41-rank-cache $C /tmp/x 0 2 --index-only | sha256sum | cut -c1-20")
  [[ -f "$D/rank-cache/rank0of2-$want.bin" ]] || in_hf --entrypoint tf-dsv41-rank-cache "$IMAGE" "$C" "/hf/$SUB/rank-cache" 0 2 >/dev/null
  ls -la "$D/rank-cache"
  WC="\$HOME/.cache/dsv41-tensorfold/rank-cache-${MODEL_REVISION:0:8}-ablit-${ABLIT_REVISION:0:8}-d1"
  ssh -n -o BatchMode=yes "$worker" "mkdir -p $WC && df -h $WC | tail -1 && docker run --rm --network none --user \$(id -u):\$(id -g) -e HOME=/tmp -v '$NFS_VOLUME:/hf:ro' -v $WC:/out --entrypoint tf-dsv41-rank-cache '$IMAGE' '$C' /out 1 2 >/dev/null && ls -la $WC && df -h $WC | tail -1"
fi
