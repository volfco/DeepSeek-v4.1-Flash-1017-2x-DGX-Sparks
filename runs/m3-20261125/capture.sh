#!/usr/bin/env bash
# Capture an M3 probe dump: start the probe image at PARALLEL=1 with the dump switched on, decode a batch of requests,
# copy the dump out, restore production.
#   bash runs/m3-20261125/capture.sh <layer> <requests> [max_tokens]
# PARALLEL=1 is required: train.py's chain-prefix label assumes one window a round.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export WORKLOAD_CORPUS=runs/overnight-20261122/corpus.json
LAYER=${1:?layer}
REQS=${2:-120}
TOKS=${3:-128}
IMG=tensorfold-dsv41:bench-probe-dump
C=dsv41-1017-exl3-tf
OUT=runs/m3-20261125/capture
mkdir -p "$OUT"
DUMP=/spill/m3-probe.bin

echo "$(date -u +%FT%TZ) restart on $IMG, PARALLEL=1, layer $LAYER, dump $DUMP"
env IMAGE="$IMG" PREPARE=0 PARALLEL=1 TF_DS_PROBE_LAYER="$LAYER" TF_DS_PROBE_DUMP="$DUMP" \
  ./start.sh restart > "$OUT/restart.log" 2>&1 || { echo "  restart failed"; tail -8 "$OUT/restart.log"; }
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null || { echo "  no /health"; exit 1; }
# do NOT delete the dump here: start.sh's smoke test already made the engine's first verify round, so the writer has
# the file open; unlinking it now leaves the engine writing to a deleted inode and nothing ever reaches the disk.
# The writer truncates on open, so a stale file from an earlier run is cleared by the engine itself.
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $C | grep -E '^(PARALLEL|TF_DS_PROBE_)' | tee "$OUT/armed.txt"
echo "$(date -u +%FT%TZ) capture: $REQS requests x $TOKS tokens"
python3 runs/m3-20261125/capture.py "$REQS" "$TOKS" 2>&1 | tee "$OUT/capture.log" | tail -3
# the engine opened the dump; find where it landed rather than trusting the absolute path (Dir.cwd() did not anchor
# /spill as expected on the first attempt: the file ended up in the container's overlay, which dies with it)
# (no -xdev: /spill and the container's cwd may be different mounts)
FOUND=$(docker exec $C sh -c 'find / -name "m3-probe.bin" -not -path "/proc/*" 2>/dev/null | head -1')
docker exec $C sh -c 'ls -la /m3-probe.bin /spill/m3-probe.bin 2>/dev/null; echo cwd=$(readlink /proc/1/cwd)' | head -3
SIZE=$(docker exec $C sh -c "stat -c %s '$FOUND' 2>/dev/null || echo 0")
echo "$(date -u +%FT%TZ) dump at $FOUND, bytes: $SIZE"
docker cp "$C:$FOUND" "$OUT/m3-layer$LAYER.bin"
docker logs $C 2>&1 | grep -E "PROBE_LAYER|probe" | head -3 | tee "$OUT/probe.log"

echo "$(date -u +%FT%TZ) restore production"
./start.sh restart > "$OUT/restore.restart.log" 2>&1 || grep -q 'the smoke test request failed' "$OUT/restore.restart.log" || echo "restore restart failed"
for i in $(seq 90); do curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
docker inspect -f '{{.Config.Image}}' $C
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$OUT/restored.validate.json" \
  --compare runs/matrix/base.validate.json > "$OUT/restored.validate.log" 2>&1 && echo "restored: validate rc=0" || echo "restored: validate FAILED"
echo "$(date -u +%FT%TZ) done; $(stat -c %s "$OUT/m3-layer$LAYER.bin" 2>/dev/null || echo 0) bytes pulled"
