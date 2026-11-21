#!/usr/bin/env bash
# 0200-0212 short A/B, 2026-11-21 (run on the head from the recipe root). Two images, one switch set per arm:
#   I  = tensorfold-dsv41:bench-followup-0212 = bench-prune-six + zig/patches 0200-0212
#   P  = tensorfold-dsv41:bench-prune-six-20261120 = production source + 0195/0197/0198 + 0199 (the previous bundle)
# Base lines are the live production record (no restart) and a base arm on I. Every fused switch is production's own
# unset one; PDL is off in production, so the last arm turns it on.
set -uo pipefail
bench=runs/fused-followup-20261121
I=tensorfold-dsv41:bench-followup-0212
P=tensorfold-dsv41:bench-prune-six-20261120
production=tensorfold-dsv41:exl3-tp2-ea522d353626
container=dsv41-1017-exl3-tf
worker=$(sed -n 's/^WORKER=//p' .env)
url=http://127.0.0.1:8000
arm=runs/draft-block-20261118/arm.sh
export PATH="$HOME/.local/bin:$PATH"
exec > >(tee -a "$bench/campaign.log") 2>&1
running_image() { docker inspect -f '{{if .State.Running}}{{.Config.Image}}{{end}}' "$container" 2>/dev/null || true; }
restore() {
  rc=$?; trap - EXIT
  echo "== restore $(date -u +%T) (rc=$rc)"
  bash "$arm" followup-restore production > "$bench/restore.start.log" 2>&1 || { echo "RESTORE FAILED"; exit 90; }
  python3 -I tools/validate.py "$url" --model DeepSeek-V4.1-Flash-1017 --save "$bench/restored.json" \
    --compare runs/matrix/base.validate.json > "$bench/restore-validate.log" 2>&1 || { echo "RESTORE VALIDATE FAILED"; exit 91; }
  curl -fsS "$url/health" > "$bench/restored-health.json"
  echo "Production restored ($(running_image)) and validated"; exit "$rc"
}
start_arm() {  # label image switches... ; returns 1 (arm skipped) instead of exiting
  local label=$1 image=$2; shift 2
  local tag=followup-$label
  echo "== arm $label $* $(date -u +%T)"
  bash "$arm" "$tag" "$image" "$@" > "$bench/$label.start.log" 2>&1 || { echo "ARM $label FAILED TO START"; tail -5 "$bench/$label.start.log"; return 1; }
  cp "runs/draft-block-20261118/$tag/armed.txt" "$bench/$label.ranks.txt"
  local inspect="docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $container"
  for kv in "$@"; do
    bash -c "$inspect" | grep -Fx "$kv" >> "$bench/$label.ranks.txt" || { echo "rank0 missing $kv"; return 1; }
    ssh -o BatchMode=yes "$worker" "$inspect" | grep -Fx "$kv" >> "$bench/$label.ranks.txt" || { echo "rank1 missing $kv"; return 1; }
  done
  for r in "" "ssh -o BatchMode=yes $worker"; do $r docker exec $container sha256sum /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes; done >> "$bench/$label.ranks.txt"
}
exact() {  # label compare-against
  local label=$1 against=$2
  echo "-- exact $label $(date -u +%T)"
  python3 -I tools/exact.py record --url "$url" --like "$against" --label "$label" --out "$bench/$label.exact.json" > "$bench/$label.exact.log" 2>&1 || echo "exact record rc=$?"
  python3 -I tools/exact.py compare "$against" "$bench/$label.exact.json" > "$bench/$label.exact-compare.log" 2>&1
  echo "exact $label vs $(basename "$against"): $(tail -1 "$bench/$label.exact-compare.log")"
}
timing() { echo "-- timing $1 $(date -u +%T)"; python3 "$bench/probe-serving.py" "$bench/$1.timing.json" "fab-$1-$(date +%s)" > "$bench/$1.timing.log" 2>&1 || echo "timing $1 FAILED"; tail -4 "$bench/$1.timing.log"; }

[[ "$(running_image)" == "$production" ]] || { echo "production not running"; exit 2; }
docker image inspect "$I" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$I' >/dev/null" || { echo "missing $I"; exit 3; }
docker image inspect "$P" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$P' >/dev/null" || { echo "missing $P"; exit 3; }
trap restore EXIT

# the live production panel is the base line every arm's exact suite is compared against
echo "== production record $(date -u +%T)"
python3 -I tools/exact.py record --url "$url" --label prod --out "$bench/prod.exact.json" > "$bench/prod.exact.log" 2>&1 || echo "prod record rc=$?"
echo "prod: $(tail -1 "$bench/prod.exact.log")"
python3 "$bench/probe-serving.py" "$bench/prod.timing.json" "fab-prod-$(date +%s)" > "$bench/prod.timing.log" 2>&1 || echo "prod timing FAILED"

Y="$bench/prod.exact.json"
IDX=TF_DS_IDX_FUSED=1
C2N=TF_DS_FUSED_COMPRESS2_NORM=1
RST=TF_DS_FUSED_ROPE_STORE=1
RT=TF_DS_FUSED_ROUTE=1
PDL=TF_DS_PDL=1

start_arm base "$I" && { exact base "$Y"; timing base; }
start_arm prev "$P" "$IDX" "$C2N" "$RST" "$RT" && { exact prev "$Y"; timing prev; }
start_arm idx "$I" "$IDX" && { exact idx "$Y"; timing idx; }
start_arm comp "$I" "$C2N" && { exact comp "$Y"; timing comp; }
start_arm rope "$I" "$RST" && { exact rope "$Y"; timing rope; }
start_arm route "$I" "$RT" && { exact route "$Y"; timing route; }
start_arm all "$I" "$IDX" "$C2N" "$RST" "$RT" && { exact all "$Y"; exact all-vs-prev "$bench/prev.exact.json"; timing all; }
start_arm allpdl "$I" "$IDX" "$C2N" "$RST" "$RT" "$PDL" && { exact allpdl "$bench/all.exact.json"; timing allpdl; }
start_arm base2 "$I" && { exact base2 "$Y"; timing base2; }
echo "== arms done $(date -u +%T)"
