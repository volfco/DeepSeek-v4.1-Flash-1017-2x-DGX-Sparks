#!/usr/bin/env bash
# Overnight arm runner (2026-11-22). From the recipe root on the head:
#   OUT=runs/overnight-20261122/<suite> REF=<exact reference json> ARMS_FILE=<file> \
#     setsid nohup bash runs/overnight-20261122/arms.sh > <OUT>/run.log 2>&1 < /dev/null &
# ARMS_FILE: one arm a line, `name|image|ENV=V ENV=V ...` (image empty = production's .env image; '#' lines skipped).
# Per arm: restart with production's .env + the arm's IMAGE/ENV (+ PREPARE=0 TF_DS_ROUND_PROFILE=1), env proof on both
# ranks, tools/exact.py record --like REF (REF empty: the first arm records fresh and becomes REF) + compare against
# REF, the kvreuse-size workload (graphs-on forward_ms a round, 4K/32K/128K x 1/4 streams), and banner counts for
# BANNERS (a |-separated grep -E pattern). WORKLOAD=0 skips the workload, EXACT=0 skips the panel.
# Then production is restored from .env alone and validated. Never touches .env.
set -uo pipefail
: "${OUT:?set OUT}" "${ARMS_FILE:?set ARMS_FILE}"
REF=${REF:-}
BANNERS=${BANNERS:-TF_DS_KV_FETCH_REUSE: decode reuses}
WORKLOAD=${WORKLOAD:-1}
EXACT=${EXACT:-1}
SAMPLED=${SAMPLED:-0}   # 1: runs/overnight-20261122/sampled.py each arm, compared against the first arm's (SREF)
SREF=${SREF:-}
mkdir -p "$OUT"
C=dsv41-1017-exl3-tf
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
insp() { if [[ $1 == 0 ]]; then bash -c "$2" < /dev/null; else ssh -n -o BatchMode=yes "$worker" "$2"; fi; }
prod_img=$(grep -m1 '^IMAGE=' .env | cut -d= -f2-)
echo "$(date -u +%FT%TZ) production image $prod_img; ref=${REF:-<first arm>}"

while IFS= read -r line <&3; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  name=${line%%|*}; rest=${line#*|}; img=${rest%%|*}; envs=${rest#*|}
  [[ -z "$img" ]] && img=$prod_img
  echo "$(date -u +%FT%TZ) arm $name image=$img env=[$envs]"
  for r in 0 1; do insp $r "docker image inspect '$img' >/dev/null" || { echo "  $img missing on rank$r: skip"; continue 2; }; done
  # shellcheck disable=SC2086
  env IMAGE="$img" PREPARE=0 TF_DS_ROUND_PROFILE=1 $envs ./start.sh restart > "$OUT/$name.restart.log" 2>&1 \
    || grep -q 'the smoke test request failed' "$OUT/$name.restart.log" || { echo "  restart failed"; tail -5 "$OUT/$name.restart.log"; continue; }
  ok=0; for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && { ok=1; break; }; sleep 5; done
  (( ok )) || { echo "  no /health"; continue; }
  for r in 0 1; do
    echo "rank$r $(insp $r "docker inspect -f '{{.Config.Image}}' $C") $(insp $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C" | grep -E '^(TF_DS_|TENSORFOLD_)' | grep -vE '_(KEY|TOKEN)=' | sort | tr '\n' ' ')"
  done > "$OUT/$name.armed.txt"
  if (( EXACT )); then
    like=(); [[ -n "$REF" ]] && like=(--like "$REF")
    python3 -I tools/exact.py record --url http://127.0.0.1:8000 --out "$OUT/$name.exact.json" --label "$name" "${like[@]}" > "$OUT/$name.exact.log" 2>&1
    echo "  exact record rc=$?"
    if [[ -z "$REF" ]]; then REF="$OUT/$name.exact.json"; echo "  (reference is now $REF)"
    else python3 -I tools/exact.py compare "$REF" "$OUT/$name.exact.json" > "$OUT/$name.cmp.txt" 2>&1; echo "  exact vs ref rc=$? $(tail -1 "$OUT/$name.cmp.txt")"; fi
  fi
  if (( SAMPLED )); then
    if [[ -z "$SREF" ]]; then python3 runs/overnight-20261122/sampled.py "$name" "$OUT/$name.sampled.json" 2>&1 | tail -1 | sed 's/^/  /'; SREF="$OUT/$name.sampled.json"
    else python3 runs/overnight-20261122/sampled.py "$name" "$OUT/$name.sampled.json" --compare "$SREF" 2>&1 | tail -1 | sed 's/^/  /'; fi
  fi
  if (( WORKLOAD )); then python3 runs/kvreuse-size-20261122/workload.py "$name" "$OUT/$name.json" 2>&1 | tee "$OUT/$name.log" | sed 's/^/  /'; fi
  for r in 0 1; do
    echo "  rank$r banners: $(insp $r "docker logs $C 2>&1 | grep -cE '$BANNERS'") errors: $(insp $r "docker logs $C 2>&1 | grep -ciE 'panic|error\\.|CudaFailed|segfault'")"
  done | tee "$OUT/$name.banner.txt"
done 3< "$ARMS_FILE"

echo "$(date -u +%FT%TZ) restore production (.env alone)"
./start.sh restart > "$OUT/restore.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$OUT/restore.restart.log" || echo "restore restart failed"
for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
docker inspect -f '{{.Config.Image}}' $C
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$OUT/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$OUT/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
tail -2 "$OUT/restored.validate.log"
echo "$(date -u +%FT%TZ) finished"
