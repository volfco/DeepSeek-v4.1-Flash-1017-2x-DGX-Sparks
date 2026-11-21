#!/usr/bin/env bash
set -euo pipefail
bench=runs/fused-route-0193-0194
out="$bench/retest"
image=tensorfold-dsv41:bench-fused-route-0193-0194-fixed
production=tensorfold-dsv41:exl3-tp2-ea522d353626
container=dsv41-1017-exl3-tf
worker=$(sed -n 's/^WORKER=//p' .env)
url=http://127.0.0.1:8000
arm=runs/draft-block-20261118/arm.sh
exact_template=runs/combo3/blocks/a01.exact.json
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$out"
exec > >(tee "$out/campaign.log") 2>&1

running_image() {
  docker inspect -f '{{if .State.Running}}{{.Config.Image}}{{end}}' "$container" 2>/dev/null || true
}
restore() {
  rc=$?
  trap - EXIT
  if [[ "$(running_image)" != "$production" ]]; then
    bash "$arm" fused-route-retest-restore production || exit 90
    python3 -I tools/validate.py "$url" --model DeepSeek-V4.1-Flash-1017 \
      --save "$out/restored.json" --compare runs/matrix/base.validate.json > "$out/restore-validate.log" 2>&1 || exit 91
    curl -fsS "$url/health" > "$out/restored-health.json" || exit 92
    echo "Production restored and validated"
  fi
  exit "$rc"
}
record() {
  local label=$1 template=$2
  echo "Exact suite: $label"
  python3 -I tools/exact.py record --url "$url" --like "$template" --label "$label" \
    --out "$out/$label.exact.json" > "$out/$label.exact.log" 2>&1
}
timing() {
  local label=$1
  echo "Decode timing: $label"
  python3 "$bench/probe.py" "$out/$label.timing.json"
}
start_arm() {
  local label=$1
  shift
  if ! bash "$arm" "fused-route-retest-$label" "$image" "$@" > "$out/$label.start.log" 2>&1; then
    cat "$out/$label.start.log"
    exit 12
  fi
  cp "runs/draft-block-20261118/fused-route-retest-$label/armed.txt" "$out/$label.ranks.txt"
  local inspect="docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $container"
  for kv in "$@"; do
    bash -c "$inspect" | grep -Fx "$kv" >> "$out/$label.ranks.txt"
    ssh -o BatchMode=yes "$worker" "$inspect" | grep -Fx "$kv" >> "$out/$label.ranks.txt"
  done
}
[[ "$(running_image)" == "$production" ]]
docker image inspect "$image" >/dev/null
ssh -o BatchMode=yes "$worker" "docker image inspect '$image' >/dev/null"
curl -fsS "$url/health" | python3 -c 'import json,sys; live=json.load(sys.stdin)["live"]; sys.exit(bool(live["connections"] or live["waiting"]))'
record production "$exact_template"
trap restore EXIT
start_arm control TF_DS_FUSED_ROUTE=0
record control "$out/production.exact.json"
python3 -I tools/exact.py compare "$out/production.exact.json" "$out/control.exact.json" | tee "$out/control-exact-compare.log"
start_arm on TF_DS_FUSED_ROUTE=1
record on "$out/control.exact.json"
python3 -I tools/exact.py compare "$out/control.exact.json" "$out/on.exact.json" | tee "$out/exact-compare.log"
start_arm pdl TF_DS_FUSED_ROUTE=1 TF_DS_PDL=1
record pdl "$out/control.exact.json"
python3 -I tools/exact.py compare "$out/control.exact.json" "$out/pdl.exact.json" | tee "$out/pdl-exact-compare.log"
start_arm off TF_DS_FUSED_ROUTE=0
timing off
start_arm timed-on TF_DS_FUSED_ROUTE=1
timing on
