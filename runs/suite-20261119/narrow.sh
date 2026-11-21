#!/usr/bin/env bash
# Shorten the suite: stop the runner inside block s01 (the pair stays on the base arm), run the narrow probe on that
# same base process, close s01, then start the runner again on the short schedule. Run detached on the head.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
O=runs/suite-20261119
log() { echo "$(date -u +%FT%TZ) $*" >> $O/status.log; }
pid=$(pgrep -f "^python3 runs/suite-20261119/run_suite.py" | head -1)
if [ -n "$pid" ]; then
  kids=$(pgrep -P "$pid" | tr '\n' ' ')
  kill -TERM "$pid"; sleep 1; kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
  [ -n "$kids" ] && kill $kids 2>/dev/null
  log "NARROW: runner $pid stopped inside block s01; the pair stays on the base arm"
fi
for i in $(seq 1 90); do
  live=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; l=json.load(sys.stdin)["live"]; print(l["connections"], l["waiting"])' 2>/dev/null)
  [ "$live" = "0 0" ] && break; sleep 2
done
sleep 8
log "NARROW: pair idle ($live); base probe on the running base process"
python3 $O/suite_probe.py s01 $O/blocks/s01.probe.json --narrow > $O/blocks/s01.probe.log 2>&1; log "NARROW: base probe rc=$?"
python3 $O/finalize_s01.py > $O/blocks/s01.finalize.log 2>&1; log "NARROW: s01 closed rc=$? $(tail -1 $O/blocks/s01.finalize.log | cut -c1-200)"
[ -e $O/STOP ] && { log "NARROW: STOP present, not starting the runner"; exit 0; }
nohup env CAMPAIGN_OUT=runs/suite-20261119 CAMPAIGN_IMAGE=exl3-tp2-ea522d353626 CAMPAIGN_PROD='{}' CAMPAIGN_CLOCK_ABORT=0 python3 runs/suite-20261119/run_suite.py >> $O/runner.log 2>&1 < /dev/null &
