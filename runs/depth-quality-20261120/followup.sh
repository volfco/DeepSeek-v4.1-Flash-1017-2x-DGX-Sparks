#!/usr/bin/env bash
# Waits for 120 s of continuous idleness, then: lead-off arm + probe, then restore production + validate.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/depth-quality-20261120
IMG=tensorfold-dsv41:exl3-tp2-ea522d353626
export ARM_ALLOW_PAST_TRAFFIC=1
idle() { [[ "$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null)" == "0 0" ]]; }
wait_idle() { local q=0; echo "$(date -u +%T) waiting for 120 s idle"; while (( q < 12 )); do if idle; then q=$((q+1)); else q=0; fi; sleep 10; done; echo "$(date -u +%T) idle"; }
until wait_idle && runs/draft-block-20261118/arm.sh dq-off "$IMG" TF_DS_ENGRAM_LEAD=0; do sleep 30; done
echo "lead banners on rank0 (want 0): $(docker logs dsv41-1017-exl3-tf 2>&1 | grep -c 'engram-lead: on')"
python3 -I "$D/probe.py" off || echo "probe off failed"
until wait_idle && runs/draft-block-20261118/arm.sh dq-restore production; do sleep 30; done
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$D/restored.json" \
  --compare runs/matrix/base.validate.json > "$D/restore-validate.log" 2>&1; echo "restore validate rc=$?"
tail -2 "$D/restore-validate.log"
echo "== $(date -u +%T) followup done"
