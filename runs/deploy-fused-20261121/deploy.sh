#!/usr/bin/env bash
# Production on the fused image: production's .env plus the bench-prune-six image (production source + fused kernels
# 0193-0199) and its four switches. From the recipe root on the head:  runs/deploy-fused-20261121/deploy.sh
# Evidence: runs/suite-fused-20261121 (four blocks ABBA: exact 39/39 identical on both fused blocks, every probe/decode
# reply equal, probe +0.76%/+0.91%, decode rounds 0.1-1.1% shorter, prefill within +-0.6%).
# Refuses if the pair has clients or non-loopback traffic, or if .env already names an image or a fused switch.
# Undo: cp ~/dsv41-env-backups/.env.bak-20261121-pre-fused .env && ./start.sh restart
# The image change starts a fresh spill namespace: production's kept states go cold (the old namespace stays on disk).
set -euo pipefail
out=runs/deploy-fused-20261121; mkdir -p "$out"
C=dsv41-1017-exl3-tf
IMG=tensorfold-dsv41:bench-prune-six-20261120
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1

live=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null || echo down)
[[ "$live" == "0 0" ]] || { echo "pair is not idle ($live): not restarting"; exit 3; }
if docker logs $C 2>&1 | grep 'POST /v1/' | grep -v '\[tensorfold\] 127\.0\.0\.1 ' | grep -q .; then
  echo "non-loopback requests in the running container's log: not restarting"; exit 4
fi
grep -qE '^(IMAGE|PREPARE|TF_DS_IDX_FUSED|TF_DS_FUSED_[A-Z0-9_]+)=' .env && { echo ".env already sets an image, PREPARE or a fused switch"; exit 5; }
docker image inspect "$IMG" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$IMG' >/dev/null" || { echo "$IMG missing on a rank"; exit 6; }

mkdir -p "$HOME/dsv41-env-backups"
bak=$HOME/dsv41-env-backups/.env.bak-20261121-pre-fused
[[ -e "$bak" ]] || install -m 600 .env "$bak"
cmp -s .env "$bak" || { echo "backup differs from .env"; exit 7; }
cat >> .env <<EOF
# 2026-11-21: fused kernels 0193-0199 (runs/suite-fused-20261121, runs/deploy-fused-20261121); undo: restore $bak
IMAGE=$IMG
PREPARE=0
TF_DS_IDX_FUSED=1
TF_DS_FUSED_COMPRESS2_NORM=1
TF_DS_FUSED_ROPE_STORE=1
TF_DS_FUSED_ROUTE=1
EOF
echo "appended 7 lines (.env now $(wc -l < .env) lines)"

# the image and switches must come from .env alone, and nothing may be built
DRY_RUN=1 ./start.sh restart > "$out/dry-run.log" 2>&1 || true
grep -q "$IMG" "$out/dry-run.log" || { echo "dry run does not name $IMG: restoring .env"; cp "$bak" .env; tail -5 "$out/dry-run.log"; exit 8; }
if grep -qiE 'docker build|prepare\.sh|building' "$out/dry-run.log"; then echo "dry run would build: restoring .env"; cp "$bak" .env; exit 9; fi

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
  for kv in TF_DS_IDX_FUSED=1 TF_DS_FUSED_COMPRESS2_NORM=1 TF_DS_FUSED_ROPE_STORE=1 TF_DS_FUSED_ROUTE=1 TF_DS_DRAFT_BLOCK=6 TF_DS_ENGRAM_LEAD=1 TF_DS_T26_KEYED_DRAFT=1; do
    insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -qx "$kv" || { echo "rank$r missing $kv"; exit 12; }
  done
done
echo "both ranks on $IMG with the fused switches and production's switches"

python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" \
  --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1 && echo "validate rc=0" || { echo "validate FAILED"; tail -5 "$out/validate.log"; exit 13; }
tail -3 "$out/validate.log" | cut -c1-160
date -u +%FT%TZ > "$out/finished.txt"
