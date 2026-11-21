#!/usr/bin/env bash
# The draft-block comparison: control, block 6, block 7, control again on one benchmark image, then production back.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
D=runs/draft-block-20261118; img=$(cat $D/package/tag.txt)
bash $D/run_arm.sh a0-off "$img" 1
bash $D/run_arm.sh b6 "$img" 1 TF_DS_DRAFT_BLOCK=6
bash $D/run_arm.sh b7 "$img" 1 TF_DS_DRAFT_BLOCK=7
bash $D/run_arm.sh a1-off "$img" 0
# production back from .env alone, its own spill tier, then the saved replies
bash $D/arm.sh production production > $D/production.arm.log 2>&1; echo "$(date -u +%FT%TZ) production arm rc=$?" >> $D/status.log
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/production/validate.json --compare runs/matrix/base.validate.json > $D/production/validate.log 2>&1
echo "$(date -u +%FT%TZ) production validate rc=$? $(tail -1 $D/production/validate.log)" >> $D/status.log
curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("production health", h["status"], h["live"]["connections"], h["live"]["waiting"], "spill", [(s["rank"], s["points"]) for s in h["prompt_cache"]["spill"]])' >> $D/status.log 2>&1
echo "$(date -u +%FT%TZ) CAMPAIGN DONE" >> $D/status.log
