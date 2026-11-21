#!/usr/bin/env bash
# Turn the Engram lead reader on in production: one line in .env, a restart, checks. From the recipe root on the head.
#   runs/deploy-lead-20261119/deploy.sh
# Refuses if the pair has clients, if anything but the head has sent requests to the running container, or if .env
# already sets an Engram lead, dedupe or row-cache switch. Undo: delete the TF_DS_ENGRAM_LEAD line from .env and run
# ./start.sh restart (the .env before this change is ~/dsv41-env-backups/.env.bak-20261119-pre-lead).
set -euo pipefail
out=runs/deploy-lead-20261119; mkdir -p "$out"
C=dsv41-1017-exl3-tf
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PREPARE=0

live=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null || echo down)
[[ "$live" == "0 0" ]] || { echo "pair is not idle ($live): not restarting"; exit 3; }
if docker logs $C 2>&1 | grep 'POST /v1/' | grep -v '\[tensorfold\] 127\.0\.0\.1 ' | grep -q .; then
  echo "non-loopback requests in the running container's log: not restarting"; exit 4
fi
if grep -qE '^TF_DS_ENGRAM_(LEAD|DEDUPE|ROWCACHE_MB)=' .env; then echo ".env already sets an Engram lead, dedupe or row-cache switch"; exit 5; fi
bak=$HOME/dsv41-env-backups/.env.bak-20261119-pre-lead
[[ -e "$bak" ]] || install -m 600 .env "$bak"
[[ "$(wc -l < "$bak")" == 41 && "$(stat -c %a "$bak")" == 600 ]] || { echo "backup is not the 41-line .env at mode 600"; exit 6; }
printf 'TF_DS_ENGRAM_LEAD=1\n' >> .env
echo "appended: $(sed -n '42p' .env) (.env now $(wc -l < .env) lines)"

date -u +%FT%TZ > "$out/started.txt"
rc=0; ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
echo "$rc" > "$out/restart.rc"
if (( rc != 0 )); then
  # the known first-request failure after a start: one retry decides
  grep -q 'the smoke test request failed' "$out/restart.log" || { echo "restart failed rc=$rc"; tail -5 "$out/restart.log"; exit 7; }
fi
reply=$(curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":16}')
echo "$reply" > "$out/smoke.json"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert "error" not in r and r["usage"]["completion_tokens"] > 0, r; print("smoke ok:", r["choices"][0]["message"].get("content"))' "$out/smoke.json"

allow='^(TENSORFOLD_TOKEN_CACHE_MB|TF_DS_DRAFT_BLOCK|TF_DS_DRAFT_BLOCK_KEPT|TF_DS_T26_KEYED_DRAFT|TF_DS_T13_EXPERT_PRUNE|TF_DS_T13_EXPERT_CAP|TF_DS_PREFILL_SLICE|TF_DS_KV_SHARD|TF_DS_SPILL_GIB|TF_DS_STREAMS|TF_DS_ENGRAM_LEAD|TF_DS_ENGRAM_AIO|TF_DS_ENGRAM_DEDUPE|TF_DS_ENGRAM_ROWCACHE_MB|TF_DS_ENGRAM_SCALES|TF_DS_SCORE)='
insp() { if [[ $1 == 0 ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$worker" "$2"; fi; }
{
  for r in 0 1; do
    echo "rank$r $(insp $r "docker inspect -f '{{.Config.Image}} started {{.State.StartedAt}}' $C")"
    insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -E "$allow" | sort | tr '\n' ' '; echo
    insp $r "docker inspect -f '{{range .Mounts}}{{println .Destination .Source}}{{end}}' $C" | grep '^/spill'
    insp $r "docker logs $C 2>&1" | grep -E 'engram-lead|Engram row cache' | cut -c1-110
  done
} > "$out/armed.txt"
cat "$out/armed.txt"
[[ "$(grep -c 'engram-lead: on' "$out/armed.txt")" == 2 ]] || { echo "the lead banner is not on at both ranks"; exit 8; }
grep -q 'Engram row cache\|TF_DS_ENGRAM_DEDUPE=1' "$out/armed.txt" && { echo "row cache or dedupe is on"; exit 9; }

python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" \
  --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1 && echo "validate rc=0" || { echo "validate FAILED"; tail -5 "$out/validate.log"; exit 10; }
tail -4 "$out/validate.log" | cut -c1-160
python3 "$out/prefill_check.py" | tee "$out/prefill_check.txt"
date -u +%FT%TZ > "$out/finished.txt"
