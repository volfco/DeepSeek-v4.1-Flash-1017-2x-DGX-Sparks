#!/usr/bin/env bash
# Lead on / lead off / lead on, each a fresh process on production's image with a private spill dir, then restore.
#   nohup runs/depth-quality-20261120/campaign.sh > runs/depth-quality-20261120/campaign.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/depth-quality-20261120
IMG=tensorfold-dsv41:exl3-tp2-ea522d353626
export ARM_ALLOW_PAST_TRAFFIC=1   # colum asked for this run (2026-11-20); live clients still block via /health
banners() { docker logs dsv41-1017-exl3-tf 2>&1 | grep -c 'engram-lead: on'; }
run() {  # label, extra env...
  local label=$1; shift
  echo "== $(date -u +%T) arm $label $*"
  runs/draft-block-20261118/arm.sh "dq-$label" "$IMG" "$@" || { echo "arm $label failed"; return 1; }
  echo "lead banners on rank0: $(banners)"
  python3 -I "$D/probe.py" "$label" || echo "probe $label failed"
}
run on1 TF_DS_ENGRAM_LEAD=1
run off TF_DS_ENGRAM_LEAD=0
run on2 TF_DS_ENGRAM_LEAD=1
echo "== $(date -u +%T) restore production"
runs/draft-block-20261118/arm.sh dq-restore production
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$D/restored.json" \
  --compare runs/matrix/base.validate.json > "$D/restore-validate.log" 2>&1; echo "restore validate rc=$?"
tail -2 "$D/restore-validate.log"
echo "== $(date -u +%T) done"
