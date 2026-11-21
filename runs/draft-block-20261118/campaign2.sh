#!/usr/bin/env bash
# Second pass, on the refined image (a wider pass only after a round that kept TF_DS_DRAFT_BLOCK_KEPT rows and with
# room in the round): control, block 6, block 7, block 7 with a lower threshold, control again, then production back.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
D=runs/draft-block-20261118; img=$(cat $D/package/tag.txt)
bash $D/run_arm.sh c0-off "$img" 1
bash $D/run_arm.sh c6 "$img" 1 TF_DS_DRAFT_BLOCK=6
bash $D/run_arm.sh c7 "$img" 1 TF_DS_DRAFT_BLOCK=7
bash $D/run_arm.sh c7k3 "$img" 0 TF_DS_DRAFT_BLOCK=7 TF_DS_DRAFT_BLOCK_KEPT=3
bash $D/run_arm.sh c1-off "$img" 0
bash $D/arm.sh production2 production > $D/production2.arm.log 2>&1; echo "$(date -u +%FT%TZ) production arm rc=$?" >> $D/status.log
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $D/production2/validate.json --compare runs/matrix/base.validate.json > $D/production2/validate.log 2>&1
echo "$(date -u +%FT%TZ) production validate rc=$? $(tail -1 $D/production2/validate.log)" >> $D/status.log
curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("production health", h["status"], h["live"]["connections"], h["live"]["waiting"], "spill", [(s["rank"], s["points"]) for s in h["prompt_cache"]["spill"]])' >> $D/status.log 2>&1
echo "$(date -u +%FT%TZ) CAMPAIGN2 DONE" >> $D/status.log
