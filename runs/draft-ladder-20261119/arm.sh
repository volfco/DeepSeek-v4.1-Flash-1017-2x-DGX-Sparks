#!/usr/bin/env bash
# Start one arm of the draft-ladder comparison on the pair, from the recipe root on the head.
#   runs/draft-ladder-20261119/arm.sh <label> <image> [KEY=VALUE ...]
# The arm runs <image> on both ranks with a private spill directory (the production tier is not touched), the
# production .env otherwise, and the given switches. It refuses to restart a pair that has clients.
# "production" as the image restores the pair from .env alone (production's own spill tier).
set -euo pipefail
label=$1; image=$2; shift 2
out=runs/draft-ladder-20261119/$label
mkdir -p "$out"
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PREPARE=0

live=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null || echo down)
if [[ "$live" != "0 0" && "$live" != down ]]; then echo "pair has clients ($live): not restarting"; exit 3; fi
if docker ps --format '{{.Names}}' | grep -qx dsv41-1017-exl3-tf; then
  # anything but loopback since this container started means real traffic: stop and say so
  if docker logs dsv41-1017-exl3-tf 2>&1 | grep 'POST /v1/' | grep -v '\[tensorfold\] 127\.0\.0\.1 ' | grep -q .; then
    echo "non-loopback requests in the running container's log: not restarting"; exit 4
  fi
fi

if [[ "$image" != production ]]; then
  spill=$HOME/.cache/dsv41-ladder-$label
  mkdir -p "$spill"; ssh -o BatchMode=yes "$worker" "mkdir -p '$spill'"
  export IMAGE=$image SPILL_DIR=$spill WORKER_SPILL_DIR=$spill
fi
for kv in "$@"; do export "$kv"; done
date -u +%FT%TZ > "$out/started.txt"
printf '%s\n' "image=$image" "$@" > "$out/arm.txt"
rc=0; ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
echo "$rc" > "$out/restart.rc"
if (( rc != 0 )); then
  # the known first-request failure after a start: one retry decides
  grep -q 'the smoke test request failed' "$out/restart.log" || { echo "restart failed rc=$rc"; tail -5 "$out/restart.log"; exit 5; }
fi
reply=$(curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16}')
echo "$reply" > "$out/smoke.json"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert "error" not in r and r["usage"]["completion_tokens"] > 0, r; print("smoke ok:", r["choices"][0]["message"].get("content"))' "$out/smoke.json"

allow='^(TENSORFOLD_TOKEN_CACHE_MB|TF_DS_DRAFT_BLOCK|TF_DS_T26_KEYED_DRAFT|TF_DS_T13_EXPERT_PRUNE|TF_DS_T27_DEVICE_DRAW|TF_DS_PREFILL_SLICE|TF_DS_L2_PREFETCH|TF_DS_KV_SHARD|TF_DS_SPILL_GIB|TF_DS_STREAMS|TF_DS_POOL|TF_DS_GRAPHS|TF_DS_GRAPH_CAP|TF_DS_RECIPE_PATCHES|TF_DS_T13_EXPERT_CAP|TF_DS_ENGRAM_LEAD|TF_DS_SCORE|TF_DS_DRAFT_BLOCK_KEPT|TF_DS_DRAFT_LADDER|TF_DS_DRAFT_LADDER_LOG|TF_DS_ENGRAM_AIO)='
{
  echo "rank0 $(docker inspect -f '{{.Config.Image}} {{.Image}}' dsv41-1017-exl3-tf)"
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf | grep -E "$allow" | sort | tr '\n' ' '; echo
  docker inspect -f '{{range .Mounts}}{{if eq .Destination "/spill"}}spill {{.Source}}{{end}}{{end}}' dsv41-1017-exl3-tf
  echo "rank1 $(ssh -o BatchMode=yes "$worker" "docker inspect -f '{{.Config.Image}} {{.Image}}' dsv41-1017-exl3-tf")"
  ssh -o BatchMode=yes "$worker" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf" | grep -E "$allow" | sort | tr '\n' ' '; echo
  ssh -o BatchMode=yes "$worker" "docker inspect -f '{{range .Mounts}}{{if eq .Destination \"/spill\"}}spill {{.Source}}{{end}}{{end}}' dsv41-1017-exl3-tf"
} > "$out/armed.txt"
cat "$out/armed.txt"
docker logs dsv41-1017-exl3-tf 2>&1 | grep -i 'TF_DS_DRAFT_BLOCK\|TF_DS_DRAFT_LADDER\|engram-lead' | head -3 || true
