#!/usr/bin/env bash
# Capture a TF_DS_MARKOV_TRACE dump: the markov-trace image, a sampled workload (live runs T=1.0), PARALLEL=8.
#   bash runs/markov-wb-20261125/capture.sh [requests] [tokens]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export WORKLOAD_CORPUS=runs/overnight-20261122/corpus.json
REQS=${1:-80}
TOKS=${2:-64}
IMG=tensorfold-dsv41:bench-markov-trace
C=dsv41-1017-exl3-tf
OUT=runs/markov-wb-20261125/capture
mkdir -p "$OUT"
DUMP=/spill/markov.bin
echo "$(date -u +%FT%TZ) restart on $IMG, TF_DS_MARKOV_TRACE=$DUMP"
env IMAGE="$IMG" PREPARE=0 TF_DS_MARKOV_TRACE="$DUMP" \
  ./start.sh restart > "$OUT/restart.log" 2>&1 || { echo "  restart failed"; tail -8 "$OUT/restart.log"; }
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null || { echo "  no /health"; exit 1; }
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C | grep -E '^(PARALLEL|TF_DS_MARKOV_TRACE)=' | tee "$OUT/armed.txt"
echo "$(date -u +%FT%TZ) capture: $REQS x $TOKS, sampled"
TEMPERATURE=1.0 PARALLEL=8 python3 runs/m3-20261125/capture.py "$REQS" "$TOKS" 2>&1 | tee "$OUT/capture.log" | tail -2
FOUND=$(docker exec $C sh -c 'find / -name "markov.bin" -not -path "/proc/*" 2>/dev/null | head -1')
echo "dump at '$FOUND' $(docker exec $C sh -c "stat -c %s '$FOUND' 2>/dev/null || echo 0") bytes"
docker cp "$C:$FOUND" "$OUT/markov.bin"
docker logs $C 2>&1 | grep -E "MARKOV_TRACE" | head -2 | tee "$OUT/trace.log"
echo "$(date -u +%FT%TZ) restore production"
./start.sh restart > "$OUT/restore.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$OUT/restore.restart.log" || echo "restore restart failed"
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
docker inspect -f '{{.Config.Image}}' $C
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$OUT/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$OUT/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
echo "$(date -u +%FT%TZ) done; $(stat -c %s "$OUT/markov.bin" 2>/dev/null || echo 0) bytes pulled"
