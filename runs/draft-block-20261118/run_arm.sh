#!/usr/bin/env bash
# One arm of the draft-block comparison, from the recipe root on the head: start it, gate it, measure it.
#   runs/draft-block-20261118/run_arm.sh <label> <image> <exact: 0|1> [KEY=VALUE ...]
set -uo pipefail
label=$1; image=$2; exact=$3; shift 3
D=runs/draft-block-20261118; out=$D/$label
say() { echo "$(date -u +%FT%TZ) $label $*" | tee -a $D/status.log; }
say "start $*"
bash $D/arm.sh "$label" "$image" "$@" > "$out.arm.log" 2>&1 || { say "ARM FAILED rc=$?"; exit 1; }
say "armed"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1
say "validate rc=$? $(grep -c '^PASS' "$out/validate.log") pass, $(grep -c '^FAIL' "$out/validate.log") fail: $(tail -1 "$out/validate.log")"
python3 $D/probe.py "$label" "$out/probe.json" > "$out/probe.log" 2>&1
say "probe rc=$?"
python3 -I tools/bench.py --url http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --suite decode,multi --streams 1,2,4,8 --reps 3 --label "$label" --out "$out/bench.json" > "$out/bench.log" 2>&1
say "bench rc=$? $(grep -E 'decode.tok_s|multi\.[1248]\.aggregate' "$out/bench.log" | awk '{printf "%s=%s ", $1, $2}')"
if [[ "$exact" == 1 ]]; then
  python3 -I tools/exact.py record --url http://127.0.0.1:8000 --like runs/combo3/blocks/a01.exact.json --label "$label" --out "$out/exact.json" > "$out/exact-record.log" 2>&1
  say "exact record rc=$?"
  python3 -I tools/exact.py compare runs/combo3/blocks/a01.exact.json "$out/exact.json" > "$out/exact-compare.log" 2>&1
  say "exact compare rc=$? $(tail -1 "$out/exact-compare.log" | cut -c1-160)"
fi
docker logs dsv41-1017-exl3-tf 2>&1 | grep -ciE 'MissingTritonVariant|CudaFailed|pair broke' > "$out/engine-errors.count"
say "done engine-error-lines=$(cat "$out/engine-errors.count") health=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print(h["status"], h["live"]["connections"], h["live"]["waiting"])' 2>/dev/null)"
