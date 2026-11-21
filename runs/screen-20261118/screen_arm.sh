#!/usr/bin/env bash
# One arm of the screen, from the recipe root on the head: start it on the production image with a private spill
# directory, then the measurements its kind asks for.
#   runs/screen-20261118/screen_arm.sh <label> <kind: ctl|prune|lead> <sampled Tetris seeds> [KEY=VALUE ...]
# A start that fails or a request that fails ends the arm non-zero (the campaign then restores production).
set -uo pipefail
label=$1; kind=$2; seeds=$3; shift 3
S=runs/screen-20261118; D=runs/draft-block-20261118; out=$S/$label
img=tensorfold-dsv41:exl3-tp2-ea522d353626
say() { echo "$(date -u +%FT%TZ) $label $*" | tee -a $S/status.log; }
leave() {
  docker logs dsv41-1017-exl3-tf 2>&1 | grep -iE 'error|mismatch|missing|broke|toowide' | grep -v 'prompt=8 tokens=1\|a broken rank pair ends\|CUDA_ERROR_INVALID_VALUE (1)' > "$out/engine-errors.log" 2>/dev/null || true
  say "leave rc=$1 engine-error-lines=$(wc -l < "$out/engine-errors.log" 2>/dev/null || echo ?) health=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print(h["status"], h["live"]["connections"], h["live"]["waiting"])' 2>/dev/null)"
  exit "$1"
}
say "start $kind $*"
mkdir -p "$out"
# arm.sh writes under the draft-block run directory: give it this run's label and move its record here afterwards
bash $D/arm.sh "screen-$label" "$img" TF_DS_SCORE=1 "$@" > "$out/arm.log" 2>&1; rc=$?
cp -r $D/screen-$label/. "$out/" 2>/dev/null; rm -rf $D/screen-$label
(( rc == 0 )) || { say "ARM FAILED rc=$rc: $(tail -2 "$out/arm.log" | tr '\n' ' ' | cut -c1-200)"; exit 1; }
say "armed"
# what each rank says it runs: a rejected cap or lead prints its refusal here and the engine carries on without it
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
{ echo "== rank 0"; docker logs dsv41-1017-exl3-tf 2>&1 | grep -iE 'prune|T13|expert.cap|the cap|engram-lead|lead|TF_DS_SCORE|score|DRAFT_BLOCK' | grep -v 'POST\|GET\|prompt cache' | head -12
  echo "== rank 1"; ssh -o BatchMode=yes "$worker" "docker logs dsv41-1017-exl3-tf 2>&1" | grep -iE 'prune|T13|expert.cap|the cap|engram-lead|lead|DRAFT_BLOCK' | head -12; } > "$out/startup-lines.txt" 2>&1
say "startup: $(grep -ciE 'cap is off|refus|expected' "$out/startup-lines.txt") refusals; $(grep -iE 'prune|cap|lead' "$out/startup-lines.txt" | head -3 | cut -c1-110 | tr '\n' '|')"
python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/validate.json" --compare runs/matrix/base.validate.json > "$out/validate.log" 2>&1; rc=$?
say "validate rc=$rc: $(grep 'concurrent == solo\|equal to the saved' "$out/validate.log" | sed 's/  */ /g' | tr '\n' ';' | cut -c1-160)"
[[ "$kind" == prune ]] || (( rc == 0 )) || leave 10      # a lossy arm may change replies; a lossless one may not
python3 $S/screen.py "$label" "$out/screen.json" "$seeds" > "$out/screen.log" 2>&1; rc=$?
say "screen rc=$rc"
(( rc == 0 )) || leave 11
if true; then   # every arm: a lossless arm's 80 scores must equal the control's
  python3 -I tools/score.py record --url http://127.0.0.1:8000 --in runs/engine-followup/runtime01/items.jsonl --out "$out/score.json" --label "$label" > "$out/score.log" 2>&1; rc=$?
  say "score rc=$rc: $(grep '^overall' "$out/score.log" | cut -c1-120)"
  (( rc == 0 )) || leave 12
  python3 -I tools/score.py record --url http://127.0.0.1:8000 --in runs/overnight-opt/prune/width8-02/confirmation-items.jsonl --out "$out/confirmation.json" --label "$label" > "$out/confirmation.log" 2>&1; rc=$?
  say "confirmation rc=$rc: $(grep '^overall' "$out/confirmation.log" | cut -c1-120)"
  (( rc == 0 )) || leave 13
fi
if [[ "$kind" != prune ]]; then
  python3 -I tools/bench.py --url http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --suite prefill --prefill-sizes 8000,32000 --reps 3 --salt screen1118 --label "$label" --out "$out/prefill.json" > "$out/prefill.log" 2>&1; rc=$?
  say "prefill rc=$rc: $(grep -E 'prefill\.[0-9]+\.tok_s' "$out/prefill.log" | awk '{printf "%s=%s ", $1, $2}')"
  (( rc == 0 )) || leave 14
  python3 -I tools/bench.py --url http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --suite prefill --prefill-sizes 100000 --reps 1 --salt screen1118 --label "$label" --out "$out/prefill100k.json" > "$out/prefill100k.log" 2>&1; rc=$?
  say "prefill100k rc=$rc: $(grep -E 'prefill\.[0-9]+\.tok_s' "$out/prefill100k.log" | awk '{printf "%s=%s ", $1, $2}')"
  (( rc == 0 )) || leave 15
fi
leave 0
