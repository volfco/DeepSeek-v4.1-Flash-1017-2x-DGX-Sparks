#!/usr/bin/env bash
# One gated arm, from the recipe root on the head: start it, validate it, probe it, then the 39-case exact record.
#   runs/draft-block-20261118/run_arm3.sh <label> <image> [KEY=VALUE ...]
# Any failed gate ends the arm with a non-zero code (the campaign then restores production). The engine's error
# lines are saved before the arm is left, whatever happened.
set -uo pipefail
label=$1; image=$2; shift 2
D=runs/draft-block-20261118; out=$D/$label
say() { echo "$(date -u +%FT%TZ) $label $*" | tee -a $D/status.log; }
leave() {
  docker logs dsv41-1017-exl3-tf 2>&1 | grep -iE 'error|mismatch|refus|missing|broke|toowide' | grep -v 'prompt=8 tokens=1' > "$out/engine-errors.log" 2>/dev/null || true
  say "leave rc=$1 engine-error-lines=$(wc -l < "$out/engine-errors.log" 2>/dev/null || echo ?) health=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print(h["status"], h["live"]["connections"], h["live"]["waiting"])' 2>/dev/null)"
  exit "$1"
}
say "start $*"
mkdir -p "$out"
bash $D/arm.sh "$label" "$image" "$@" > "$out.arm.log" 2>&1 || { say "ARM FAILED rc=$?"; exit 1; }
say "armed"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1; rc=$?
say "validate rc=$rc $(grep -c '^PASS' "$out/validate.log") pass, $(grep -c '^FAIL' "$out/validate.log") fail: $(tail -1 "$out/validate.log")"
(( rc == 0 )) || leave 10
python3 $D/probe.py "$label" "$out/probe.json" > "$out/probe.log" 2>&1; rc=$?
say "probe rc=$rc $(grep -c 'FAILED' "$out/probe.log") failed rows"
(( rc == 0 )) || leave 11
python3 -I tools/exact.py record --url http://127.0.0.1:8000 --like runs/combo3/blocks/a01.exact.json --label "$label" --out "$out/exact.json" > "$out/exact-record.log" 2>&1; rc=$?
say "exact record rc=$rc"
(( rc == 0 )) || leave 12
python3 -I tools/exact.py compare runs/combo3/blocks/a01.exact.json "$out/exact.json" > "$out/exact-compare.log" 2>&1; rc=$?
say "exact compare rc=$rc $(tail -1 "$out/exact-compare.log" | cut -c1-160)"
(( rc == 0 )) || leave 13
leave 0
