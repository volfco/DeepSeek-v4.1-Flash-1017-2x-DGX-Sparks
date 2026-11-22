#!/usr/bin/env bash
# TF_DS_KV_FETCH_REUSE A/B (candidate 1): one image (84806bb + the reuse patch, switch off by default), ABBA
#   off1 on1 on2 off2 : TF_DS_KV_FETCH_REUSE unset / =1, everything else production's .env, graphs on.
# Each arm: restart; tools/exact.py record (39-case panel; arms 2-4 --like off1, so prompts are byte-identical);
# the kvreuse-size workload under TF_DS_ROUND_PROFILE=1 (graphs-on forward_ms a round, 4K/32K/128K x 1/4 streams);
# the reuse banner count on both ranks. Then production is restored from .env alone and validated.
# From the recipe root on the head:
#   IMG=tensorfold-dsv41:bench-fused-next-<sha> setsid nohup bash runs/kvreuse-ab-20261122/run.sh > runs/kvreuse-ab-20261122/run.log 2>&1 < /dev/null &
set -uo pipefail
out=runs/kvreuse-ab-20261122; mkdir -p "$out"
: "${IMG:?set IMG}"
C=dsv41-1017-exl3-tf
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
grep -qx 'IMAGE=tensorfold-dsv41:bench-fused-next-84806bb' .env || { echo "production is not on 84806bb"; exit 2; }
docker image inspect "$IMG" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$IMG' >/dev/null" || { echo "$IMG missing on a rank"; exit 3; }

arm() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) arm $name: $*"
  env IMAGE="$IMG" PREPARE=0 TF_DS_ROUND_PROFILE=1 "$@" ./start.sh restart > "$out/$name.restart.log" 2>&1 \
    || grep -q 'the smoke test request failed' "$out/$name.restart.log" || { echo "restart $name failed"; tail -5 "$out/$name.restart.log"; return 1; }
  for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
  for r in 0 1; do
    if [[ $r == 0 ]]; then q=bash; else q="ssh -o BatchMode=yes $worker"; fi
    echo "rank$r $($q -c "docker inspect -f '{{.Config.Image}}' $C") $($q -c "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -E '^TF_DS_(KV_FETCH_REUSE|ROUND_PROFILE|IDX_FUSED|FUSED_ROUTE)=' | sort | tr '\n' ' ')"
  done > "$out/$name.armed.txt"
  local like=(); [[ $name != off1 ]] && like=(--like "$out/off1.exact.json")
  python3 -I tools/exact.py record --out "$out/$name.exact.json" --label "$name" "${like[@]}" > "$out/$name.exact.log" 2>&1; echo "exact rc=$?"
  python3 runs/kvreuse-size-20261122/workload.py "$name" "$out/$name.json" 2>&1 | tee "$out/$name.log"
  for r in 0 1; do
    if [[ $r == 0 ]]; then q=bash; else q="ssh -o BatchMode=yes $worker"; fi
    echo "rank$r reuse banners: $($q -c "docker logs $C 2>&1 | grep -c 'TF_DS_KV_FETCH_REUSE: decode reuses'")"
  done | tee "$out/$name.banner.txt"
}
arm off1 || true
arm on1 TF_DS_KV_FETCH_REUSE=1 || true
arm on2 TF_DS_KV_FETCH_REUSE=1 || true
arm off2 || true
for a in on1 on2 off2; do
  python3 -I tools/exact.py compare "$out/off1.exact.json" "$out/$a.exact.json" > "$out/cmp-off1-$a.txt" 2>&1; echo "exact off1 vs $a rc=$?"
done
echo "$(date -u +%FT%TZ) restore production (.env alone)"
./start.sh restart > "$out/restore.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$out/restore.restart.log" || echo "restore restart failed"
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
docker inspect -f '{{.Config.Image}}' $C
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$out/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
tail -2 "$out/restored.validate.log"
echo "$(date -u +%FT%TZ) finished"
