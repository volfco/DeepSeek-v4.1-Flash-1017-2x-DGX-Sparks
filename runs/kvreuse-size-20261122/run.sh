#!/usr/bin/env bash
# Candidate 1 sizing on production's image (bench-fused-next-84806bb, graphs on): three restarts, same workload.
#   m4  : TF_DS_ROUND_PROFILE=1 TF_DS_CLOCK_MARKS=4  (the sharded attention's parts + a_q/a_idx/a_core only)
#   m1  : TF_DS_ROUND_PROFILE=1 TF_DS_CLOCK_MARKS=1  (every phase mark)
#   off : production exactly (the restore), wall times only = the marks' perturbation control; then validate.
# From the recipe root on the head:  setsid nohup bash runs/kvreuse-size-20261122/run.sh > runs/kvreuse-size-20261122/run.log 2>&1 < /dev/null &
set -uo pipefail
out=runs/kvreuse-size-20261122; mkdir -p "$out"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
grep -qx 'IMAGE=tensorfold-dsv41:bench-fused-next-84806bb' .env || { echo "production is not on 84806bb"; exit 2; }
arm() {
  local name=$1; shift
  echo "$(date -u +%FT%TZ) arm $name: $*"
  env "$@" ./start.sh restart > "$out/$name.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$out/$name.restart.log" || { echo "restart $name failed"; tail -5 "$out/$name.restart.log"; return 1; }
  for i in $(seq 60); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' dsv41-1017-exl3-tf | grep -E '^(TF_DS_ROUND_PROFILE|TF_DS_CLOCK_MARKS)=' > "$out/$name.armed.txt" || true
  python3 "$out/workload.py" "$name" "$out/$name.json" 2>&1 | tee "$out/$name.log"
}
arm m4 TF_DS_ROUND_PROFILE=1 TF_DS_CLOCK_MARKS=4 || true
arm m1 TF_DS_ROUND_PROFILE=1 TF_DS_CLOCK_MARKS=1 || true
arm off || true
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$out/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
tail -3 "$out/restored.validate.log"
echo "$(date -u +%FT%TZ) finished"
