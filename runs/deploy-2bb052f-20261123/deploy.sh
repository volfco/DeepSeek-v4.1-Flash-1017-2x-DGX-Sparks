#!/usr/bin/env bash
# Production: image 10c9866 -> 2bb052f (kernel-stack-0278 = 10c9866 + 0278 + 0279) with TF_DS_TAPS_PACKED=1 and
# TF_DS_IDX_PICK_WIDE=1. Evidence: runs/k278 (all arms exact 39/39 + sampled IDENTICAL, 300K/600K decode shas equal to
# control; 0278 cold prefill +0.4% 32K / +0.6% 96K, all cells above both controls; 0279 neutral at 300K/600K).
#   bash runs/deploy-2bb052f-20261123/deploy.sh > runs/deploy-2bb052f-20261123/run.log 2>&1
# Undo: cp ~/dsv41-env-backups/.env.bak-20261123-pre-2bb052f .env && PREPARE=0 ./start.sh restart
set -euo pipefail
out=runs/deploy-2bb052f-20261123; mkdir -p "$out"
C=dsv41-1017-exl3-tf
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
grep -qx 'IMAGE=tensorfold-dsv41:bench-fused-next-10c9866' .env || { echo "unexpected IMAGE"; exit 5; }
grep -qE '^TF_DS_(TAPS_PACKED|IDX_PICK_WIDE)=' .env && { echo ".env already sets a new switch"; exit 5; }
for r in 0 1; do if [[ $r == 0 ]]; then docker image inspect tensorfold-dsv41:bench-fused-next-2bb052f >/dev/null; else ssh -n -o BatchMode=yes "$worker" "docker image inspect tensorfold-dsv41:bench-fused-next-2bb052f >/dev/null"; fi || { echo "image missing on rank$r"; exit 6; }; done
live=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])')
[[ "$live" == "0 0" ]] || { echo "not idle: $live"; exit 6; }
mkdir -p "$HOME/dsv41-env-backups"; bak=$HOME/dsv41-env-backups/.env.bak-20261123-pre-2bb052f
[[ -e "$bak" ]] || install -m 600 .env "$bak"; cmp -s .env "$bak" || { echo "backup differs"; exit 7; }
sed -i 's#^IMAGE=tensorfold-dsv41:bench-fused-next-10c9866$#IMAGE=tensorfold-dsv41:bench-fused-next-2bb052f#' .env
printf '# 2026-11-23: image 2bb052f + 0278 TAPS_PACKED (prefill +0.5%%) + 0279 IDX_PICK_WIDE (neutral; >262K decode), exact; runs/k278; undo: restore %s\nTF_DS_TAPS_PACKED=1\nTF_DS_IDX_PICK_WIDE=1\n' "$bak" >> .env
diff "$bak" .env || true
rc=0; PREPARE=0 ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
if (( rc != 0 )) && ! grep -q 'the smoke test request failed' "$out/restart.log"; then echo "restart failed: rollback"; cp "$bak" .env; PREPARE=0 ./start.sh restart > "$out/rollback.log" 2>&1; exit 10; fi
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
insp() { if [[ $1 == 0 ]]; then bash -c "$2" < /dev/null; else ssh -n -o BatchMode=yes "$worker" "$2"; fi; }
for r in 0 1; do v=$(insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -E '^(NCCL_MAX_CTAS|TF_DS_TAPS_PACKED|TF_DS_IDX_PICK_WIDE)=' | sort | tr '\n' ' ')
  img=$(insp $r "docker inspect -f '{{.Config.Image}}' $C"); echo "rank$r $img $v"
  [[ "$img" == tensorfold-dsv41:bench-fused-next-2bb052f && "$v" == *TF_DS_IDX_PICK_WIDE=1*TF_DS_TAPS_PACKED=1* ]] || { echo "rank$r not armed"; exit 12; }; done | tee "$out/armed.txt"
python3 -I tools/exact.py record --url http://127.0.0.1:8000 --out "$out/exact.json" --label prod2bb --like runs/deploy-10c9866-20261122/exact.json > "$out/exact.log" 2>&1
cmpres=$(python3 -I tools/exact.py compare runs/deploy-10c9866-20261122/exact.json "$out/exact.json" | tail -1); echo "$cmpres"
python3 runs/overnight-20261122/sampled.py prod2bb "$out/sampled.json" --compare runs/overnight-20261122/sched2/ctl1.sampled.json 2>&1 | tail -1
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" \
  --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1 && echo "validate rc=0" || { echo "validate FAILED"; tail -5 "$out/validate.log"; exit 13; }
[[ "$cmpres" == *IDENTICAL* ]] || { echo "exact NOT identical: rolling back"; cp "$bak" .env; PREPARE=0 ./start.sh restart > "$out/rollback.log" 2>&1; exit 14; }
date -u +%FT%TZ > "$out/finished.txt"; echo deployed
