#!/usr/bin/env bash
# Short fused-kernel A/B on the pair (run on the head from the recipe root).
#   A = bench-fused-route-0193-0194-fixed (production source + fixed 0193/0194)
#   B = bench-fused-0193-0198 (A + 0195-0198 ported; 0196 rides TF_DS_FUSED_ROUTE)
# Arms: A-off, A-route, A-off2 | B-off, B-195, B-all, B-off2. Exact suite on every arm except the repeat controls.
set -uo pipefail
bench=runs/fused-bench-20261119
A=tensorfold-dsv41:bench-fused-route-0193-0194-fixed
B=tensorfold-dsv41:bench-fused-0193-0198
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
  bash "$arm" fused-bench-restore production > "$bench/restore.start.log" 2>&1 || { echo "RESTORE FAILED"; exit 90; }
  python3 -I tools/validate.py "$url" --model DeepSeek-V4.1-Flash-1017 --save "$bench/restored.json" \
    --compare runs/matrix/base.validate.json > "$bench/restore-validate.log" 2>&1 || { echo "RESTORE VALIDATE FAILED"; exit 91; }
  curl -fsS "$url/health" > "$bench/restored-health.json"
  echo "Production restored ($(running_image)) and validated"; exit "$rc"
}
start_arm() {  # label image switches...
  local label=$1 image=$2; shift 2
  echo "== arm $label $image $* $(date -u +%T)"
  bash "$arm" "fused-bench-$label" "$image" "$@" > "$bench/$label.start.log" 2>&1 || { cat "$bench/$label.start.log"; exit 12; }
  cp "runs/draft-block-20261118/fused-bench-$label/armed.txt" "$bench/$label.ranks.txt"
  local inspect="docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $container"
  for kv in "$@"; do
    bash -c "$inspect" | grep -Fx "$kv" >> "$bench/$label.ranks.txt" || { echo "rank0 missing $kv"; exit 13; }
    ssh -o BatchMode=yes "$worker" "$inspect" | grep -Fx "$kv" >> "$bench/$label.ranks.txt" || { echo "rank1 missing $kv"; exit 13; }
  done
  for r in "" "ssh -o BatchMode=yes $worker"; do $r docker exec $container sha256sum /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes; done >> "$bench/$label.ranks.txt"
}
exact() {  # label template [compare-against]
  local label=$1 template=$2 against=${3:-}
  echo "-- exact $label $(date -u +%T)"
  python3 -I tools/exact.py record --url "$url" --like "$template" --label "$label" --out "$bench/$label.exact.json" > "$bench/$label.exact.log" 2>&1 || echo "exact record rc=$?"
  if [[ -n "$against" ]]; then python3 -I tools/exact.py compare "$against" "$bench/$label.exact.json" > "$bench/$label.exact-compare.log" 2>&1; echo "exact $label vs $(basename "$against"): $(tail -1 "$bench/$label.exact-compare.log")"; fi
}
timing() { echo "-- timing $1 $(date -u +%T)"; python3 "$bench/probe.py" "$bench/$1.timing.json" > "$bench/$1.timing.log" 2>&1 || echo "timing $1 FAILED"; tail -6 "$bench/$1.timing.log"; }

[[ "$(running_image)" == "$production" ]] || { echo "production not running"; exit 2; }
for img in "$A" "$B"; do docker image inspect "$img" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$img' >/dev/null" || { echo "missing $img"; exit 3; }; done
trap restore EXIT
export ARM_ALLOW_PAST_TRAFFIC=1  # colum: the Sparks are unused today; live clients still block (health check)
ref=runs/fused-route-0193-0194/production.exact.json

start_arm A-off "$A" TF_DS_FUSED_ROUTE=0
exact A-off "$ref" "$ref"; timing A-off
start_arm A-route "$A" TF_DS_FUSED_ROUTE=1
exact A-route "$bench/A-off.exact.json" "$bench/A-off.exact.json"; timing A-route
start_arm A-off2 "$A" TF_DS_FUSED_ROUTE=0
timing A-off2
start_arm B-off "$B" TF_DS_FUSED_ROUTE=0
exact B-off "$bench/A-off.exact.json" "$bench/A-off.exact.json"; timing B-off
start_arm B-195 "$B" TF_DS_IDX_FUSED=1 TF_DS_FUSED_COMPRESS2_NORM=1 TF_DS_FUSED_ROPE_STORE=1
exact B-195 "$bench/B-off.exact.json" "$bench/B-off.exact.json"; timing B-195
start_arm B-all "$B" TF_DS_FUSED_ROUTE=1 TF_DS_IDX_FUSED=1 TF_DS_FUSED_COMPRESS2_NORM=1 TF_DS_FUSED_ROPE_STORE=1
exact B-all "$bench/B-off.exact.json" "$bench/B-off.exact.json"; timing B-all
start_arm B-off2 "$B" TF_DS_FUSED_ROUTE=0
timing B-off2
echo "== arms done $(date -u +%T)"
