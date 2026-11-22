#!/usr/bin/env bash
# Production: append TF_DS_DRAFT_BLOCK_KEPT=2 (widen to six drafts after a round that kept two rows, default 4).
# Image unchanged (bench-fused-next-27baa93). Evidence: runs/overnight-20261122/kept (ABBA, exact 39/39 all arms,
# decode tok/s +0.32% mean, json +2.5%).
#   IDLE_MIN=1 bash runs/deploy-kept2-20261122/deploy.sh > runs/deploy-kept2-20261122/run.log 2>&1
# Undo: cp ~/dsv41-env-backups/.env.bak-20261122-pre-kept2 .env && ./start.sh restart
set -euo pipefail
out=runs/deploy-kept2-20261122; mkdir -p "$out"
C=dsv41-1017-exl3-tf
IMG=tensorfold-dsv41:bench-fused-next-27baa93
IDLE_MIN=${IDLE_MIN:-30}
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1

grep -qx "IMAGE=$IMG" .env || { echo ".env does not name IMAGE=$IMG"; exit 5; }
grep -qx 'TF_DS_DRAFT_BLOCK=6' .env || { echo ".env lacks TF_DS_DRAFT_BLOCK=6"; exit 5; }
grep -qE '^TF_DS_DRAFT_BLOCK_KEPT=' .env && { echo ".env already sets DRAFT_BLOCK_KEPT"; exit 5; }

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
bak=$HOME/dsv41-env-backups/.env.bak-20261122-pre-kept2
[[ -e "$bak" ]] || install -m 600 .env "$bak"
cmp -s .env "$bak" || { echo "backup differs from .env"; exit 7; }
printf '# 2026-11-22: DRAFT_BLOCK_KEPT=2 adopted (runs/overnight-20261122/kept); undo: restore %s\nTF_DS_DRAFT_BLOCK_KEPT=2\n' "$bak" >> .env
diff "$bak" .env || true

date -u +%FT%TZ > "$out/started.txt"
rc=0; ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
echo "$rc" > "$out/restart.rc"
if (( rc != 0 )); then
  grep -q 'the smoke test request failed' "$out/restart.log" || { echo "restart failed rc=$rc: restoring .env"; cp "$bak" .env; tail -5 "$out/restart.log"; exit 10; }
fi
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
insp() { if [[ $1 == 0 ]]; then bash -c "$2" < /dev/null; else ssh -n -o BatchMode=yes "$worker" "$2"; fi; }
for r in 0 1; do
  envs=$(insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C")
  echo "rank$r $(insp $r "docker inspect -f '{{.Config.Image}}' $C") $(echo "$envs" | grep -E '^(TF_DS_|TENSORFOLD_)' | grep -vE '_(KEY|TOKEN)=' | sort | tr '\n' ' ')"
  for kv in TF_DS_DRAFT_BLOCK_KEPT=2 TF_DS_DRAFT_BLOCK=6 TF_DS_PLACE_BOUND_ALL=1 TF_DS_KV_FETCH_REUSE=1 TF_DS_TOPK_MERGE=1; do
    echo "$envs" | grep -qx "$kv" || { echo "rank$r missing $kv"; exit 12; }
  done
done > "$out/armed.txt"
echo "both ranks armed"
python3 -I tools/exact.py record --url http://127.0.0.1:8000 --out "$out/exact.json" --label kept2 --like runs/overnight-20261122/kept/ctl1.exact.json > "$out/exact.log" 2>&1
python3 -I tools/exact.py compare runs/overnight-20261122/kept/ctl1.exact.json "$out/exact.json" | tail -1
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" \
  --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1 && echo "validate rc=0" || { echo "validate FAILED"; tail -5 "$out/validate.log"; exit 13; }
date -u +%FT%TZ > "$out/finished.txt"
