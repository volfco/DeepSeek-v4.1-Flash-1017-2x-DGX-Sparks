#!/usr/bin/env bash
# Route extras A/B, 2026-11-20 (run on the head from the recipe root): triage, prune stats off and PDL, each on top of TF_DS_FUSED_ROUTE=1.
#   Image tensorfold-dsv41:bench-prune-six-20261120 (fused-audit + 0199), unchanged from the 0199 A/B.
set -uo pipefail
bench=runs/route-extras-bench-20261120
I=tensorfold-dsv41:bench-prune-six-20261120
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
  bash "$arm" route-extras-restore production > "$bench/restore.start.log" 2>&1 || { echo "RESTORE FAILED"; exit 90; }
  python3 -I tools/validate.py "$url" --model DeepSeek-V4.1-Flash-1017 --save "$bench/restored.json" \
    --compare runs/matrix/base.validate.json > "$bench/restore-validate.log" 2>&1 || { echo "RESTORE VALIDATE FAILED"; exit 91; }
  curl -fsS "$url/health" > "$bench/restored-health.json"
  echo "Production restored ($(running_image)) and validated"; exit "$rc"
}
start_arm() {  # label switches... ; returns 1 (arm skipped) instead of exiting
  local label=$1; shift
  echo "== arm $label $* $(date -u +%T)"
  bash "$arm" "route-extras-$label" "$I" "$@" > "$bench/$label.start.log" 2>&1 || { echo "ARM $label FAILED TO START"; tail -5 "$bench/$label.start.log"; return 1; }
  cp "runs/draft-block-20261118/route-extras-$label/armed.txt" "$bench/$label.ranks.txt"
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
timing() { echo "-- timing $1 $(date -u +%T)"; python3 "$bench/probe.py" "$bench/$1.timing.json" "fab-$1-$(date +%s)" > "$bench/$1.timing.log" 2>&1 || echo "timing $1 FAILED"; tail -7 "$bench/$1.timing.log"; }

[[ "$(running_image)" == "$production" ]] || { echo "production not running"; exit 2; }
docker image inspect "$I" >/dev/null && ssh -o BatchMode=yes "$worker" "docker image inspect '$I' >/dev/null" || { echo "missing $I"; exit 3; }
trap restore EXIT

y=runs/fused-bench-20261120/off.exact.json
R=TF_DS_FUSED_ROUTE=1
T=TF_DS_DRAFT_TRIAGE=conf:0.2
P=TF_DS_T13_PRUNE_STATS=0
D=TF_DS_PDL=1
# baseline = production + route; base repeated at start, middle and end; each switch alone on top, then all three.
start_arm base "$R" && { exact base "$y"; timing base; }
start_arm tri "$R" "$T" && { exact tri "$bench/base.exact.json"; timing tri; }
start_arm stats0 "$R" "$P" && { exact stats0 "$bench/base.exact.json"; timing stats0; }
start_arm base2 "$R" && timing base2
start_arm pdl "$R" "$D" && { exact pdl "$bench/base.exact.json"; timing pdl; }
start_arm all "$R" "$T" "$P" "$D" && { exact all "$bench/base.exact.json"; timing all; }
start_arm base3 "$R" && timing base3
echo "== arms done $(date -u +%T)"
