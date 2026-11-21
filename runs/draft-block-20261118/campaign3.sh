#!/usr/bin/env bash
# Third pass, on the image with the held-count fix: block 6 first (its validate and probe are the gate), block 7,
# then the control; the first failed gate skips the rest. Production comes back from .env at the end, whatever happened.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
D=runs/draft-block-20261118; img=$(cat $D/package/tag.txt)
bash $D/run_arm3.sh d6 "$img" TF_DS_DRAFT_BLOCK=6 &&
  bash $D/run_arm3.sh d7 "$img" TF_DS_DRAFT_BLOCK=7 &&
  bash $D/run_arm3.sh d0-off "$img"
echo "$(date -u +%FT%TZ) arms ended rc=$?" >> $D/status.log
bash $D/arm.sh production3 production > $D/production3.arm.log 2>&1; echo "$(date -u +%FT%TZ) production arm rc=$?" >> $D/status.log
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/production3/validate.json --compare runs/matrix/base.validate.json > $D/production3/validate.log 2>&1
echo "$(date -u +%FT%TZ) production validate rc=$? $(tail -1 $D/production3/validate.log)" >> $D/status.log
curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("production health", h["status"], h["live"]["connections"], h["live"]["waiting"], "spill", [(s["rank"], s["points"]) for s in h["prompt_cache"]["spill"]])' >> $D/status.log 2>&1
echo "$(date -u +%FT%TZ) CAMPAIGN3 DONE" >> $D/status.log
