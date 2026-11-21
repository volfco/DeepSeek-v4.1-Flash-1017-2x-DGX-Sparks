#!/usr/bin/env bash
# One signal-grade pass over to-benchmark.md's matrix, run on the head. Per arm: restart the pair with the arm's
# flags, identity gate (tools/validate.py --save/--compare against the base save), one bench block (reps 3), then
# the engine log, the rank-0 done lines (bench window only) and both Sparks' lowest MemAvailable. Signal-grade, per
# the operator's decision: one block per arm (no 3x base/arm alternation), validate.py as the per-switch gate (no
# per-switch exact.py record), quality column not scored. Enough to sign a toggle + or -; to-benchmark.md holds the
# protocol.
#
# Usage: ./scripts/bench_matrix.sh [arm ...]      # default: every arm, in order
# Writes runs/matrix/<arm>.{restart.log,validate.json,bench.json,engine.log,done.log,mem.txt}, runs/matrix/status.log.
# The gate label is honest: SAVED-BASE (the base arm makes the baseline), IDENTICAL / GATE-FAILED-rcN (compared),
# SAVED-NO-BASE (no baseline to compare against). Every arm's artifacts are deleted before it runs, so a failed
# arm cannot present the previous run's numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
export PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"
OUT=runs/matrix
CONTAINER=dsv41-1017-exl3-tf
MODEL="${MODEL:-DeepSeek-V4.1-Flash-1017}"
WORKER="$(grep -m1 '^WORKER=' .env | cut -d= -f2-)"
[[ -n "$WORKER" ]] || { echo "no WORKER= in .env" >&2; exit 1; }
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || { echo "another bench_matrix.sh is live" >&2; exit 1; }

# name | restart env flags | thinking (0/1). One block per arm; base runs bookend the phases for a drift read.
# Engram arms (0094-0098 switches and the existing knobs); depth arms pin one draft depth each (two refuse the start).
ARMS=(
  "base||0"
  "map|TF_DS_ENGRAM_SCALES=map|0"
  "dedupe|TF_DS_ENGRAM_DEDUPE=1|0"
  "fast|TF_DS_ENGRAM_FAST=1|0"
  "strict|TF_DS_ENGRAM_STRICT=1|0"
  "lead|TF_DS_ENGRAM_LEAD=1|0"
  "aio0|ENGRAM_AIO=0|0"
  "metrics0|TF_DS_ENGRAM_METRICS=0|0"
  "map-dedupe|TF_DS_ENGRAM_SCALES=map TF_DS_ENGRAM_DEDUPE=1|0"
  "dedupe-lead|TF_DS_ENGRAM_DEDUPE=1 TF_DS_ENGRAM_LEAD=1|0"
  "fast-dedupe|TF_DS_ENGRAM_FAST=1 TF_DS_ENGRAM_DEDUPE=1|0"
  "aio0-map|ENGRAM_AIO=0 TF_DS_ENGRAM_SCALES=map|0"
  "depth-base||1"
  "t25-005|TF_DS_T25_DRAFT_CAL=0.05|1"
  "t25-02|TF_DS_T25_DRAFT_CAL=0.2|1"
  "depth2|TF_DS_DRAFT_DEPTH2=1|1"
  "depth3|TF_DS_DRAFT_DEPTH3=1|1"
  "depth4|TF_DS_DRAFT_DEPTH4=1|1"
  "depth5|TF_DS_DRAFT_DEPTH5=1|1"
  "depth3-t25|TF_DS_DRAFT_DEPTH3=1 TF_DS_T25_DRAFT_CAL=0.05|1"
  "nocopies|TF_DS_NO_COPIES=1|1"
  "nocopies-depth3|TF_DS_NO_COPIES=1 TF_DS_DRAFT_DEPTH3=1|1"
  "fixk3|TF_DS_FIX_K=3|1"
  "base-final||0"
  # Hit-rate patches (zig/patches/0110-0116): one block each. Every switch is bit-safe, so every gate must read
  # IDENTICAL against runs/matrix/base.validate.json — the PREVIOUS image's save (these names never overwrite it),
  # which makes each block a cross-image identity check as well. hr-base/hr-base-final bookend for drift.
  "hr-base||0"
  "hr-cuts|TF_DS_SHARED_CUTS=1|0"
  "hr-edit|TF_DS_EDIT_MARK=1|0"
  "hr-protect|TF_DS_KEEP_PROTECT=1|0"
  "hr-stride|TF_SPILL_STRIDE=8|0"
  "hr-rowcache|TF_DS_ENGRAM_ROWCACHE_MB=256|0"
  "hr-plan|TF_DS_SHARED_PLAN=1|0"
  "hr-metrics0|TF_DS_CACHE_METRICS=0|0"
  "hr-base-final||0"
)

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/status.log"; }

# Sample both Sparks' MemAvailable every 15s into <prefix>.head/.worker until killed; an orphan from a killed
# sampler can only write into its own (previous) arm's files.
mem_sample() {
  exec 9>&-   # children must not inherit the runner's lock fd (a killed run's orphans would hold it)
  while :; do
    awk '/MemAvailable/{print $2}' /proc/meminfo >> "$1.head"
    timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=10 "$WORKER" \
      "awk '/MemAvailable/{print \$2}' /proc/meminfo" >> "$1.worker" 2>/dev/null
    sleep 15
  done
}

min_of() { sort -n "$1" 2>/dev/null | head -1; }

run_arm() {
  local name=$1 flags=$2 thinking=$3 t0 t_bench gate rc
  log "arm $name start flags=[$flags] thinking=$thinking"
  rm -f "$OUT/$name".{restart.log,validate.json,validate.log,bench.json,bench.log,engine.log,done.log,mem.txt}
  rm -f "$OUT/$name.mem.head" "$OUT/$name.mem.worker" "$OUT/$name.engram-before.txt" "$OUT/$name.engram-after.txt"
  mem_sample "$OUT/$name.mem" & local sampler=$!
  t0=$(date -u +%FT%TZ)

  ( [[ -z "$flags" ]] || eval "export $flags"; exec 9>&-; ./start.sh restart ) > "$OUT/$name.restart.log" 2>&1
  rc=$?
  if (( rc != 0 )); then
    log "$name RESTART-FAILED rc=$rc (see $name.restart.log)"
    kill "$sampler" 2>/dev/null; return 1
  fi

  if [[ "$name" == "base" ]]; then
    python3 -I tools/validate.py "$URL" --model "$MODEL" --save "$OUT/base.validate.json" > "$OUT/$name.validate.log" 2>&1 9>&-
    rc=$?
    gate=SAVED-BASE
    (( rc == 0 )) || gate="BASE-FAILED-rc$rc"
  elif [[ -s "$OUT/base.validate.json" ]]; then
    python3 -I tools/validate.py "$URL" --model "$MODEL" --save "$OUT/$name.validate.json" \
      --compare "$OUT/base.validate.json" > "$OUT/$name.validate.log" 2>&1 9>&-
    rc=$?
    gate=IDENTICAL
    (( rc == 0 )) || gate="GATE-FAILED-rc$rc"
  else
    python3 -I tools/validate.py "$URL" --model "$MODEL" --save "$OUT/$name.validate.json" > "$OUT/$name.validate.log" 2>&1 9>&-
    rc=$?
    gate=SAVED-NO-BASE
    (( rc == 0 )) || gate="RUN-FAILED-rc$rc"
  fi

  local -a bench=(python3 -I tools/bench.py --url "$URL" --model "$MODEL" --suite prefill,decode,multi
    --streams 1,4 --reps 3 --metrics --label "$name" --out "$OUT/$name.bench.json")
  if [[ "$thinking" == 1 ]]; then bench+=(--prefill-sizes 8000 --thinking); else bench+=(--prefill-sizes 2000,8000,32000); fi
  t_bench=$(date -u +%FT%TZ)
  # The read histogram around the block (bench.py truncates raw snapshots at 20k chars, past which the engram
  # series sit; the counters reset every restart, so only a per-arm scrape can attribute them).
  curl -s "$URL/metrics" | grep -E '^tensorfold:(engram_lookup_seconds|prompt_cache|prefix_cache)' > "$OUT/$name.engram-before.txt" 9>&- || true
  "${bench[@]}" > "$OUT/$name.bench.log" 2>&1 9>&-
  rc=$?
  curl -s "$URL/metrics" | grep -E '^tensorfold:(engram_lookup_seconds|prompt_cache|prefix_cache)' > "$OUT/$name.engram-after.txt" 9>&- || true
  local bench_note="bench-ok"
  (( rc == 0 )) || bench_note="BENCH-FAILED-rc$rc"

  # The engine log covers the whole arm (restart refusals included); the done lines only the bench window,
  # so validate.py's and the smoke's requests cannot join the accept/tokens_per_round figures.
  docker logs -t --since "$t0" "$CONTAINER" > "$OUT/$name.engine.log" 2>&1 9>&- || true
  docker logs -t --since "$t_bench" "$CONTAINER" 2>/dev/null 9>&- | grep 'done ' | tail -300 > "$OUT/$name.done.log"
  local accept
  accept=$(sed -n 's/.*accept=\([0-9.]*\)%.*tokens_per_round=\([0-9.]*\).*/\1 \2/p' "$OUT/$name.done.log" |
    awk '{a+=$1; t+=$2; n++} END {if (n) printf "accept %.1f%% tokens/round %.2f (n=%d)", a/n, t/n, n; else print "no-accept-lines"}')

  # hr-* arms: the hit-rate probes after the bench window (their done lines must not join the accept figures).
  # prompt_reuse: an agent's turns must resume >=90% with other same-system conversations interleaved (exit 1 below,
  # 2 = the server reports no counter); prefix_fanout: common-system-prompt fan-out, duplicate prefill per follower.
  local reuse_note="" fan_note=""
  if [[ "$name" == hr-* ]]; then
    python3 -I tools/prompt_reuse.py 12000 --require-counter > "$OUT/$name.reuse.log" 2>&1 9>&-
    rc=$?
    case $rc in
      0) reuse_note="reuse=ok";; 1) reuse_note="reuse=BELOW90";; 2) reuse_note="reuse=no-counter";;
      *) reuse_note="reuse=rc$rc";;
    esac
    python3 -I tools/prefix_fanout.py --out "$OUT/$name.fanout.json" --label "$name" > "$OUT/$name.fanout.log" 2>&1 9>&-
    rc=$?
    fan_note="fanout=ok"; (( rc == 0 )) || fan_note="fanout=rc$rc"
    curl -s "$URL/metrics" | grep -E '^tensorfold:(engram_lookup_seconds|prompt_cache|prefix_cache)' > "$OUT/$name.counters.txt" 9>&- || true
  fi

  kill "$sampler" 2>/dev/null
  local mh mw
  mh=$(min_of "$OUT/$name.mem.head"); mw=$(min_of "$OUT/$name.mem.worker")
  { [[ -z "$mh" ]] || echo "head_kb $mh"; [[ -z "$mw" ]] || echo "worker_kb $mw"; } > "$OUT/$name.mem.txt"

  local sig
  sig=$(python3 - "$OUT/$name.bench.json" <<'EOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    keys = (("prefill.8000.tok_s", "prefill8k"), ("decode.tok_s", "decode"), ("multi.4.aggregate_tok_s", "multi4"))
    num = lambda v: "n/a" if v is None else f"{v:.1f}"
    print(" ".join(f"{lbl}={num(s.get(k))}" for k, lbl in keys))
except Exception as e:
    print("no-summary", e)
EOF
)
  log "$name DONE gate=$gate $bench_note $sig $accept $reuse_note $fan_note mem_head_kb=${mh:-?} mem_worker_kb=${mw:-?}"
}

log "matrix start: ${*:-all arms}"
# `just matrix-start arms="a b"` lands here as the words "arms=a" "b" (just substitutes into the shell line
# unquoted), so strip a leading "arms=" from every argument: both that documented form and a plain positional
# list then work.
if (( $# )); then
  _args=()
  for a in "$@"; do _args+=("${a#arms=}"); done
  set -- "${_args[@]}"
fi
if (( $# )); then
  for a in "$@"; do
    found=0
    for spec in "${ARMS[@]}"; do [[ "${spec%%|*}" == "$a" ]] && found=1; done
    (( found )) || log "no such arm: $a"
  done
fi
for spec in "${ARMS[@]}"; do
  IFS='|' read -r name flags thinking <<<"$spec"
  if (( $# )); then
    want=0; for a in "$@"; do [[ "$a" == "$name" ]] && want=1; done
    (( want )) || continue
  fi
  run_arm "$name" "$flags" "$thinking"
done
log "matrix done: ${*:-all arms}"
