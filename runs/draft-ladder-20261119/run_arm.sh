#!/usr/bin/env bash
# One gated arm, from the recipe root on the head: start it, validate it, probe it, optionally the 39-case exact
# record, then save the engine's log.
#   runs/draft-ladder-20261119/run_arm.sh <label> <image> <exact: 0|1> [KEY=VALUE ...]
# Any failed gate ends the arm with a non-zero code (the campaign then restores production).
set -uo pipefail
label=$1; image=$2; exact=$3; shift 3
D=runs/draft-ladder-20261119; out=$D/$label
say() { echo "$(date -u +%FT%TZ) $label $*" | tee -a $D/status.log; }
leave() {
  docker logs -t dsv41-1017-exl3-tf > "$out/engine.log" 2>&1 || true
  grep -iE 'error|mismatch|refus|missing|broke|toowide' "$out/engine.log" | grep -v 'prompt=8 tokens=1' > "$out/engine-errors.log" 2>/dev/null || true
  say "leave rc=$1 engine-error-lines=$(wc -l < "$out/engine-errors.log" 2>/dev/null || echo ?) ladder-log-lines=$(grep -c '\[ladder\]' "$out/engine.log" 2>/dev/null || echo 0) non-loopback=$(grep 'POST /v1/' "$out/engine.log" | grep -vc '\[tensorfold\] 127\.0\.0\.1 ')"
  exit "$1"
}
say "start $*"
mkdir -p "$out"
bash $D/arm.sh "$label" "$image" "$@" > "$out.arm.log" 2>&1 || { say "ARM FAILED rc=$? $(tail -1 "$out.arm.log" | cut -c1-160)"; exit 1; }
say "armed: $(grep -i 'TF_DS_DRAFT_LADDER on' "$out.arm.log" | cut -c1-160 || true)"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1; rc=$?
say "validate rc=$rc $(grep -c '^PASS' "$out/validate.log") pass, $(grep -c '^FAIL' "$out/validate.log") fail: $(tail -1 "$out/validate.log")"
(( rc == 0 )) || leave 10
python3 $D/ladder_probe.py "$label" "$out/probe.json" > "$out/probe.log" 2>&1; rc=$?
say "probe rc=$rc $(grep -c 'FAILED' "$out/probe.log") failed rows"
(( rc == 0 )) || leave 11
if [[ "$exact" == 1 ]]; then
  python3 -I tools/exact.py record --url http://127.0.0.1:8000 --like runs/combo3/blocks/a01.exact.json --label "$label" --out "$out/exact.json" > "$out/exact-record.log" 2>&1; rc=$?
  say "exact record rc=$rc"
  (( rc == 0 )) || leave 12
  python3 -I tools/exact.py compare runs/combo3/blocks/a01.exact.json "$out/exact.json" > "$out/exact-compare.log" 2>&1; rc=$?
  say "exact compare rc=$rc $(tail -1 "$out/exact-compare.log" | cut -c1-160)"
  (( rc == 0 )) || leave 13
fi
leave 0
