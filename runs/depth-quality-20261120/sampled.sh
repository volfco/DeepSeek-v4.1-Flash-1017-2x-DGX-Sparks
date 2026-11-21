#!/usr/bin/env bash
# Waits for 120 s of no clients on production, then runs the sampled probe against it (no restart, no config change).
cd "$(dirname "$0")/../.."
idle=0
while (( idle < 120 )); do
  if [[ "$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null)" == "0 0" ]]; then idle=$((idle+10)); else idle=0; fi
  sleep 10
done
echo "== $(date -u +%T) idle, starting"
python3 -I runs/depth-quality-20261120/sampled.py sampled
echo "== $(date -u +%T) done"
