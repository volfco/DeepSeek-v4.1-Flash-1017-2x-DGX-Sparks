#!/usr/bin/env bash
# Production: image bench-fused-next-27baa93 -> bench-fused-next-10c9866 (kernel-stack-0272 = 27baa93 + 0272 + 0273 +
# 0274 + 0275). Adopts the two unconditional bug fixes: 0273 (a failed stretch capture joins its forks and clears
# side_out/pf_out, so one failed capture no longer breaks every later capture until restart) and 0274 (the tool-call
# gate commits the drawn token as the first token). 0272 TF_DS_RING_TRACE and 0275 TF_DS_COPY_MIN stay off (unset).
# Evidence: runs/overnight-20261122/k272 (ctl1/ctl2 on 10c9866 exact 39/39 + sampled IDENTICAL vs production's
# sched2 ctl1; forward_ms 38.89/38.96 vs k269 controls 38.85/38.87).
#   IDLE_MIN=1 bash runs/deploy-10c9866-20261122/deploy.sh > runs/deploy-10c9866-20261122/run.log 2>&1
# Undo: cp ~/dsv41-env-backups/.env.bak-20261122-pre-10c9866 .env && ./start.sh restart
set -euo pipefail
out=runs/deploy-10c9866-20261122; mkdir -p "$out"
C=dsv41-1017-exl3-tf
OLD=tensorfold-dsv41:bench-fused-next-27baa93
NEW=tensorfold-dsv41:bench-fused-next-10c9866
IDLE_MIN=${IDLE_MIN:-30}
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
grep -qx "IMAGE=$OLD" .env || { echo ".env does not name IMAGE=$OLD"; exit 5; }
grep -qx 'TF_DS_DRAFT_BLOCK_KEPT=2' .env || { echo ".env lacks KEPT=2"; exit 5; }
grep -qE '^TF_DS_(RING_TRACE|COPY_MIN)=' .env && { echo ".env sets a new switch"; exit 5; }
for r in 0 1; do
  if [[ $r == 0 ]]; then docker image inspect $NEW >/dev/null; else ssh -n -o BatchMode=yes "$worker" "docker image inspect $NEW >/dev/null"; fi || { echo "$NEW missing on rank$r"; exit 6; }
done
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
bak=$HOME/dsv41-env-backups/.env.bak-20261122-pre-10c9866
[[ -e "$bak" ]] || install -m 600 .env "$bak"
cmp -s .env "$bak" || { echo "backup differs from .env"; exit 7; }
sed -i "s#^IMAGE=$OLD\$#IMAGE=$NEW#" .env
printf '# 2026-11-22: image 10c9866 adopted (0273/0274 bug fixes; runs/overnight-20261122/k272); undo: restore %s\n' "$bak" >> .env
diff "$bak" .env || true
date -u +%FT%TZ > "$out/started.txt"
rc=0; PREPARE=0 ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
echo "$rc" > "$out/restart.rc"
if (( rc != 0 )); then
  grep -q 'the smoke test request failed' "$out/restart.log" || { echo "restart failed rc=$rc: restoring .env"; cp "$bak" .env; PREPARE=0 ./start.sh restart > "$out/rollback.log" 2>&1; tail -5 "$out/restart.log"; exit 10; }
fi
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
insp() { if [[ $1 == 0 ]]; then bash -c "$2" < /dev/null; else ssh -n -o BatchMode=yes "$worker" "$2"; fi; }
for r in 0 1; do
  img=$(insp $r "docker inspect -f '{{.Config.Image}}' $C")
  [[ "$img" == "$NEW" ]] || { echo "rank$r runs $img"; exit 12; }
  echo "rank$r $img"
done > "$out/armed.txt"
echo "both ranks on $NEW"
python3 -I tools/exact.py record --url http://127.0.0.1:8000 --out "$out/exact.json" --label prod10c --like runs/overnight-20261122/sched2/ctl1.exact.json > "$out/exact.log" 2>&1
cmpres=$(python3 -I tools/exact.py compare runs/overnight-20261122/sched2/ctl1.exact.json "$out/exact.json" | tail -1); echo "$cmpres"
python3 runs/overnight-20261122/sampled.py prod10c "$out/sampled.json" --compare runs/overnight-20261122/sched2/ctl1.sampled.json 2>&1 | tail -1
# 0274 functional check: tool_choice required with thinking on must return a tool call
curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"required","max_tokens":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}' > "$out/toolcall.json"
python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1]))["choices"][0]["message"]; tc=m.get("tool_calls") or []; print("toolcall:", "OK" if tc and tc[0]["function"]["name"]=="get_weather" else "MISSING", json.dumps(tc)[:200])' "$out/toolcall.json"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" \
  --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1 && echo "validate rc=0" || { echo "validate FAILED"; tail -5 "$out/validate.log"; exit 13; }
[[ "$cmpres" == *IDENTICAL* ]] || { echo "exact panel NOT identical: rolling back"; cp "$bak" .env; PREPARE=0 ./start.sh restart > "$out/rollback.log" 2>&1; exit 14; }
date -u +%FT%TZ > "$out/finished.txt"
