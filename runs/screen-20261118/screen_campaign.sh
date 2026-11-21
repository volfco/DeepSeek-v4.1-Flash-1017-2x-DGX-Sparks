#!/usr/bin/env bash
# The screen of pruning off, two milder expert caps and the Engram lead reader, between two controls, on the
# production image. The first failed arm skips the rest; production comes back from .env whatever happened.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
S=runs/screen-20261118; D=runs/draft-block-20261118
bash $S/screen_arm.sh ctl-a ctl 5 &&
  bash $S/screen_arm.sh prune-off prune 5 TF_DS_T13_EXPERT_PRUNE=0 &&
  bash $S/screen_arm.sh cap32 prune 1 TF_DS_T13_EXPERT_CAP=32,1 &&
  bash $S/screen_arm.sh cap24 prune 1 TF_DS_T13_EXPERT_CAP=24,1 &&
  bash $S/screen_arm.sh lead lead 1 TF_DS_ENGRAM_LEAD=1 &&
  bash $S/screen_arm.sh ctl-b ctl 5
echo "$(date -u +%FT%TZ) arms ended rc=$?" >> $S/status.log
mkdir -p $S/production
bash $D/arm.sh screen-production production > $S/production/arm.log 2>&1; echo "$(date -u +%FT%TZ) production arm rc=$?" >> $S/status.log
cp -r $D/screen-production/. $S/production/ 2>/dev/null; rm -rf $D/screen-production
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $S/production/validate.json --compare runs/matrix/base.validate.json > $S/production/validate.log 2>&1
echo "$(date -u +%FT%TZ) production validate rc=$? $(tail -1 $S/production/validate.log)" >> $S/status.log
curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("production health", h["status"], h["live"]["connections"], h["live"]["waiting"], "spill", [(s["rank"], s["points"]) for s in h["prompt_cache"]["spill"]])' >> $S/status.log 2>&1
echo "$(date -u +%FT%TZ) SCREEN DONE" >> $S/status.log
