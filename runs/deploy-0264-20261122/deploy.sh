#!/usr/bin/env bash
# Production onto 0264 TF_DS_PLACE_BOUND_ALL: IMAGE= from bench-fused-next-086e8cb (recipe-0261) to
# bench-fused-next-27baa93 (recipe-0264 = recipe-0261 + 0264, the switch off by default) and TF_DS_PLACE_BOUND_ALL=1
# appended; every other switch stays. PREPARE=0, nothing built.
# Evidence: runs/overnight-20261122/placebound (ABBA on 27baa93: exact 39/39 in all arms; rounds -0.41% mean, the
# four-stream cells -0.25/-0.91/-2.15%, single-stream flat within drift).
#   IDLE_MIN=1 bash runs/deploy-0264-20261122/deploy.sh > runs/deploy-0264-20261122/run.log 2>&1
# Undo: cp ~/dsv41-env-backups/.env.bak-20261122-pre-0264 .env && ./start.sh restart
# The image change starts a fresh spill namespace: production's kept states go cold once.
set -euo pipefail
out=runs/deploy-0264-20261122; mkdir -p "$out"
C=dsv41-1017-exl3-tf
OLD=tensorfold-dsv41:bench-fused-next-086e8cb
IMG=tensorfold-dsv41:bench-fused-next-27baa93
IDLE_MIN=${IDLE_MIN:-30}
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1

grep -qx "IMAGE=$OLD" .env || { echo ".env does not name IMAGE=$OLD"; exit 5; }
grep -qx 'PREPARE=0' .env || { echo ".env lacks PREPARE=0"; exit 5; }
grep -qE '^(TF_DS_FUSED_TRACE|TF_DS_PDL|TF_DS_ROUND_PROFILE|TF_DS_PLACE_BOUND_ALL)=' .env && { echo ".env sets a trace/PDL/profile/new switch"; exit 5; }
for kv in TF_DS_KV_FETCH_REUSE=1 TF_DS_CONF_PACK=1 TF_DS_TOPK_MERGE=1; do grep -qx "$kv" .env || { echo ".env lacks $kv"; exit 5; }; done
docker image inspect "$IMG" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$IMG' >/dev/null" || { echo "$IMG missing on a rank"; exit 6; }

idle_once() {
  local live
  live=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null || echo down)
  [[ "$live" == "0 0" ]] || return 1
  local n
  n=$(docker logs --since "${IDLE_MIN}m" $C 2>&1 | grep 'POST /v1/' | grep -vc '\[tensorfold\] 127\.0\.0\.1 ' || true)
  [[ "$n" == 0 ]]
}
echo "$(date -u +%FT%TZ) waiting for $IDLE_MIN external-idle minutes"
until idle_once; do sleep 60; done
echo "$(date -u +%FT%TZ) idle"

mkdir -p "$HOME/dsv41-env-backups"
bak=$HOME/dsv41-env-backups/.env.bak-20261122-pre-0264
[[ -e "$bak" ]] || install -m 600 .env "$bak"
cmp -s .env "$bak" || { echo "backup differs from .env"; exit 7; }
sed -i "s|^IMAGE=$OLD\$|IMAGE=$IMG|" .env
sed -i "/^IMAGE=$IMG\$/i # 2026-11-22: 0259 TF_DS_KV_FETCH_REUSE adopted on bench-fused-next-69335b8 (runs/kvreuse-ab-20261122, runs/deploy-0264-20261122); undo: restore $bak" .env
for kv in TF_DS_PLACE_BOUND_ALL=1; do grep -q "^${kv%%=*}=" .env || echo "$kv" >> .env; done
diff "$bak" .env || true

DRY_RUN=1 ./start.sh restart > "$out/dry-run.log" 2>&1 || true
grep -q "$IMG" "$out/dry-run.log" || { echo "dry run does not name $IMG: restoring .env"; cp "$bak" .env; tail -5 "$out/dry-run.log"; exit 8; }
if grep -qiE 'docker build|prepare\.sh|building' "$out/dry-run.log"; then echo "dry run would build: restoring .env"; cp "$bak" .env; exit 9; fi
idle_once || { echo "a client arrived during the dry run: restoring .env"; cp "$bak" .env; exit 3; }

date -u +%FT%TZ > "$out/started.txt"
rc=0; ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
echo "$rc" > "$out/restart.rc"
if (( rc != 0 )); then
  grep -q 'the smoke test request failed' "$out/restart.log" || { echo "restart failed rc=$rc"; tail -5 "$out/restart.log"; exit 10; }
fi
reply=$(curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16}')
echo "$reply" > "$out/smoke.json"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert "error" not in r and r["usage"]["completion_tokens"] > 0, r; print("smoke ok:", r["choices"][0]["message"].get("content"))' "$out/smoke.json"

insp() { if [[ $1 == 0 ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$worker" "$2"; fi; }
{
  for r in 0 1; do
    echo "rank$r $(insp $r "docker inspect -f '{{.Config.Image}} {{.Image}} started {{.State.StartedAt}}' $C")"
    insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -E '^(TF_DS_|TENSORFOLD_)' | grep -vE '_(KEY|TOKEN)=' | sort | tr '\n' ' '; echo
    insp $r "docker exec $C sha256sum /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes"
  done
} > "$out/armed.txt"
for r in 0 1; do
  [[ "$(insp $r "docker inspect -f '{{.Config.Image}}' $C")" == "$IMG" ]] || { echo "rank$r is not on $IMG"; exit 11; }
  for kv in TF_DS_IDX_FUSED=1 TF_DS_FUSED_COMPRESS2_NORM=1 TF_DS_FUSED_ROPE_STORE=1 TF_DS_FUSED_ROUTE=1 TF_DS_DRAFT_BLOCK=6 TF_DS_ENGRAM_LEAD=1 TF_DS_T26_KEYED_DRAFT=1 TENSORFOLD_TOKEN_CACHE_MB=32 TF_DS_KV_FETCH_REUSE=1 TF_DS_CONF_PACK=1 TF_DS_TOPK_MERGE=1 TF_DS_T27_DEVICE_DRAW=1 TF_DS_PLACE_BOUND_ALL=1; do
    insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -qx "$kv" || { echo "rank$r missing $kv"; exit 12; }
  done
  insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -qE '^(TF_DS_FUSED_TRACE|TF_DS_PDL)=' && { echo "rank$r sets trace/PDL"; exit 12; }
done
echo "both ranks on $IMG with production's switches"

python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" \
  --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1 && echo "validate rc=0" || { echo "validate FAILED"; tail -5 "$out/validate.log"; exit 13; }
tail -3 "$out/validate.log" | cut -c1-160
date -u +%FT%TZ > "$out/finished.txt"
