#!/usr/bin/env bash
# One arm of the bench-prep campaign (0130-0147 switches), end to end:
#   (a) refuse if the pair is up with a bench switch armed,
#   (b) restart the pair with the arm's vars exported (just restart-head flags="..."),
#   (c) wait for health, warm up past the known first-request failure, gate with validate.py
#       --compare against base.validate.json, run the bench block, tee everything to
#       runs/bench-prep/<label>/,
#   (d) restore base (.env) and restart the pair, verify base still matches,
#   (e) print a one-line verdict.
#
# Runs on the workstation from the repo root; the pair stays the operator's. This script RESTARTS the pair.
# NEVER run it in an agent session without the operator asking. NEVER point it at a pair carrying traffic.
#
# usage: runs/bench-prep/run_arm.sh <label> [VAR=v ...]
#   runs/bench-prep/run_arm.sh map-warm   TF_DS_ENGRAM_SCALES=map-warm
#   runs/bench-prep/run_arm.sh aio-batch  TF_DS_ENGRAM_AIO_BATCH=8
#   runs/bench-prep/run_arm.sh base                          # no vars: the base arm
#
# VAR=v args: name must match [A-Za-z_][A-Za-z0-9_]*, value must not contain whitespace or quotes/backticks
# (they ride one flags= string into an unquoted ssh command line; see the quoting trap below).
# Env overrides: HEAD_HOST (default colum@100.95.49.21, the Justfile's head), HEAD_DIR (default
# DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks), PORT (default 8000), MODEL, CONTAINER, HEALTH_DEADLINE_S
# (default 1800), RUN_ARM_FORCE=1 to skip the (a) refusal (operator's override, never the default).
#
# Quoting trap, copied from scripts/bench_matrix.sh and the Justfile (both were bitten by it):
# just substitutes {{arms}}/{{flags}} into the ssh line UNQUOTED, so `just matrix-start arms="a b"` lands
# on the remote as the words "arms=a" "b" — hence the ${var#arms=} / ${var#flags=} strips below, which
# forgive that one shape of operator typo. Our own call is always `just restart-head flags="<vars>"` with
# the vars space-separated inside ONE flags= string, which is the only form that reaches the remote's
# `export` intact. A value with whitespace would split into two words on the remote and silently arm the
# wrong thing, so it is refused outright.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1

[[ $# -ge 1 ]] || { echo "usage: $0 <label> [VAR=v ...]" >&2; exit 2; }
LABEL="$1"; shift
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "bad label: $LABEL" >&2; exit 2; }

HEAD_HOST="${HEAD_HOST:-colum@100.95.49.21}"   # keep in sync with the Justfile's head :=
HEAD_DIR="${HEAD_DIR:-DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-DeepSeek-V4.1-Flash-1017}"
CONTAINER="${CONTAINER:-dsv41-1017-exl3-tf}"
HEALTH_DEADLINE_S="${HEALTH_DEADLINE_S:-1800}"
URL="http://127.0.0.1:$PORT"   # as the pair serves it ON THE HEAD; tools get --url explicitly
OUT="runs/bench-prep/$LABEL"
REMOTE_OUT="$HEAD_DIR/runs/bench-prep/$LABEL"
BASE_SAVE="runs/matrix/base.validate.json"
mkdir -p "$OUT"
exec 9>"runs/bench-prep/.lock"
flock -n 9 || { echo "another run_arm.sh is live" >&2; exit 1; }

ARM_VARS=()
for v in "$@"; do
  # bench_matrix.sh's arms= strip and the Justfile's flags= strip, in that order: a whole `flags="A=1 B=2"`
  # argument then word-splits into its vars, and a bare word left over (a value with a space) is refused below.
  v="${v#arms=}"; v="${v#flags=}"
  for one in $v; do
    [[ "$one" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "arm var must be NAME=value, got: $one" >&2; exit 2; }
    [[ "${one#*=}" != *[[:space:]]* ]] || { echo "arm var value contains whitespace (would split on the remote): $one" >&2; exit 2; }
    [[ "${one#*=}" != *[\"\'\`]* ]] || { echo "arm var value contains a quote or backtick: $one" >&2; exit 2; }
    ARM_VARS+=("$one")
  done
done
FLAGS_STR="${ARM_VARS[*]-}"
# The bench switches: names only, never values (docker env's values may carry tokens; only `cut -d= -f1`
# is ever read). .env's own production TF_DS_* must not match, so this is the bench set and nothing else.
ARMED_RE='^(TF_DS_ENGRAM_(SCALES|DEDUPE|FAST|STRICT|LEAD|ROWCACHE_MB|DECODE_THREADS|POOL|CPUS|AIO_BATCH|SPLIT|HOST_MS|METRICS|TRIM|AIO_SLOTS)|TF_DS_DRAFT_DEPTH[0-9]+|TF_DS_FIX_K|TF_DS_NO_COPIES|TF_DS_T25_DRAFT_CAL|TF_DS_SHARED_CUTS|TF_DS_SHARED_PLAN|TF_DS_EDIT_MARK|TF_DS_KEEP_PROTECT|TF_DS_CACHE_METRICS|TF_DS_ROUND_PROFILE|TF_DS_EXACT_RESUME|TF_DS_SPEC_SAMPLE|TF_DS_PREFILL_PROFILE|TF_DS_EXPERT_BYTES|TF_DS_ALLOCATE_LEARNED|TF_DS_RDMA_PROBES|TF_DS_SPILL_SHARED_MIN_TOKENS|TF_SPILL_[A-Z_]+|SPILL_GIB|ENGRAM_AIO|TENSORFOLD_TOKEN_CACHE_MB|TF_DS_T26_KEYED_DRAFT|TF_DS_T27_DEVICE_DRAW)$'
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/status.log"; }
# A typo'd name would arm nothing and read as a neutral arm, the one failure mode this driver must not hide.
for one in "${ARM_VARS[@]-}"; do
  [[ -z "$one" || "$one" =~ $ARMED_RE ]] || log "WARN unknown switch name: ${one%%=*} (typo? an unknown var arms nothing)"
done
ssh_head() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=60 "$HEAD_HOST" "$@"; }

log "arm $LABEL vars=[$FLAGS_STR] (values are exported on the head, never written to any file)"
[[ -s "$BASE_SAVE" ]] || { log "missing $BASE_SAVE (the base validate save); refusing"; exit 2; }

# (a) refuse if a server is up with a bench switch armed: the names in ARMED_RE, read off the container's
# env with `cut -d= -f1` (values never leave the head). Anything armed by hand outside that set is not
# detectable this way — that is what the operator's eyes and RUN_ARM_FORCE=1 are for.
set +e
server_names=$(ssh_head "docker exec $CONTAINER env 2>/dev/null | cut -d= -f1" 2>/dev/null)
rc=$?
set -e
if (( rc == 0 )) && [[ -n "$server_names" ]]; then
  armed=$(printf '%s\n' "$server_names" | grep -E "$ARMED_RE" | sort | tr '\n' ' ' || true)
  if [[ -n "$armed" && "${RUN_ARM_FORCE:-0}" != 1 ]]; then
    log "REFUSE: a server is up with bench switches armed: $armed"
    log "restore base first ('just restart-head') or re-run with RUN_ARM_FORCE=1"
    exit 2
  fi
  log "pre-flight: server names matching bench switches: ${armed:-none} (rc=0, force=${RUN_ARM_FORCE:-0})"
else
  log "pre-flight: no server env readable via docker exec (pair down?); proceeding to restart"
fi

# remote helpers; every ssh call is bounded, every poll has a deadline
wait_health() {
  local deadline=$(( $(date +%s) + HEALTH_DEADLINE_S )) body
  while :; do
    body=$(ssh_head "curl -sf --max-time 5 $URL/health" 2>/dev/null || true)
    # curl -sf already fails on /health's 503 (stalled/broken); the body check is the belt to that brace.
    if [[ "$body" == *'"status":"ok"'* || "$body" == *'"status": "ok"'* ]] && [[ "$body" != *'"broken":true'* && "$body" != *'"broken": true'* ]]; then
      log "health: ok"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      log "health: deadline (${HEALTH_DEADLINE_S}s) exceeded; last body: ${body:-<empty>}"
      return 1
    fi
    sleep 5
  done
}

# The known environmental failure: the first request after ANY restart deterministically fails
# (cuLaunchKernel: CUDA_ERROR_INVALID_VALUE, prompt=8 tokens=1) on this image and its predecessor.
# One identical retry passes. Returns 0 on the first or the second attempt.
warm_up() {
  local out attempt
  for attempt in 1 2; do
    out=$(ssh_head "curl -s --max-time 120 $URL/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}'" 2>/dev/null || true)
    if [[ -n "$out" && "$out" != *'"error"'* ]]; then
      log "warm-up: attempt $attempt ok"
      return 0
    fi
    if (( attempt < 2 )); then log "warm-up: attempt 1 failed (${out:-no reply}); retrying once"; else log "warm-up: attempt 2 failed (${out:-no reply}); giving up"; fi
  done
  return 1
}

# run a remote step, tee its output locally; capture rc without aborting the arm (restore must still run)
REMOTE_RC=0
remote_step() {
  local logfile="$1"; shift
  set +e
  ssh_head "$@" 2>&1 | tee "$OUT/$logfile"
  REMOTE_RC=$?
  set -e
  return 0
}

# stage the base save on the head (the head's clone has no runs/matrix/; sync-head excludes it)
ssh_head "mkdir -p $REMOTE_OUT"
scp -q "$BASE_SAVE" "$HEAD_HOST:$REMOTE_OUT/base.validate.json"

# (b) restart with the arm vars exported, via the Justfile target. one flags= string, space separated.
log "restart: just restart-head flags=\"$FLAGS_STR\""
set +e
just restart-head flags="$FLAGS_STR" 2>&1 | tee "$OUT/restart.log"
REMOTE_RC=$?
set -e
just_rc=$REMOTE_RC
if (( just_rc != 0 )); then
  log "restart FAILED rc=$just_rc (see $OUT/restart.log); restoring base"
  GATE_RC=255; BENCH_RC=255; WARMED=no
else
  # (c) health, warm-up past the known first-request failure, gate, bench
  if wait_health; then
    if warm_up; then
      WARMED=yes
      remote_step validate.log "cd $HEAD_DIR && python3 -I tools/validate.py $URL --model $MODEL --save runs/bench-prep/$LABEL/validate.json --compare runs/bench-prep/$LABEL/base.validate.json"
      GATE_RC=$REMOTE_RC
      log "gate: rc=$GATE_RC (0 = the replies equal $BASE_SAVE, labelled IDENTICAL in the status line)"
      if (( GATE_RC == 0 )); then
        remote_step bench.log "cd $HEAD_DIR && export PORT=$PORT && python3 -I tools/bench.py --url $URL --model $MODEL --suite prefill,decode,multi --streams 1,4 --reps 3 --metrics --prefill-sizes 2000,8000,32000 --label $LABEL --out runs/bench-prep/$LABEL/bench.json"
        BENCH_RC=$REMOTE_RC
        log "bench: rc=$BENCH_RC"
      else
        BENCH_RC=3
        log "bench: skipped (gate failed)"
      fi
    else
      WARMED=no; GATE_RC=254; BENCH_RC=254
      log "warm-up failed twice; skipping gate and bench"
    fi
  else
    WARMED=no; GATE_RC=253; BENCH_RC=253
    log "health never came up; skipping gate and bench"
  fi
fi

# pull the arm's artifacts back (small JSON only)
scp -q "$HEAD_HOST:$REMOTE_OUT/validate.json" "$OUT/validate.json" 2>/dev/null || true
scp -q "$HEAD_HOST:$REMOTE_OUT/bench.json"    "$OUT/bench.json"    2>/dev/null || true

# (d) restore base (.env) and restart, verify base still matches
log "restore: just restart-head (flags=\"\" -> .env only, i.e. base production)"
set +e
just restart-head 2>&1 | tee "$OUT/restore.restart.log"
REMOTE_RC=$?
set -e
restore_rc=$REMOTE_RC
RESTORED=no
if (( restore_rc == 0 )) && wait_health && warm_up; then
  remote_step restore.validate.log "cd $HEAD_DIR && python3 -I tools/validate.py $URL --model $MODEL --save runs/bench-prep/$LABEL/restore.validate.json --compare runs/bench-prep/$LABEL/base.validate.json"
  if (( REMOTE_RC == 0 )); then RESTORED=yes; fi
  log "restore: validate rc=$REMOTE_RC restored=$RESTORED"
  scp -q "$HEAD_HOST:$REMOTE_OUT/restore.validate.json" "$OUT/restore.validate.json" 2>/dev/null || true
else
  log "restore FAILED (rc=$restore_rc) or base no longer matches: stop benchmarking, the pair is not in a known state"
fi

# (e) verdict line
python3 - "$OUT/bench.json" "$LABEL" "$GATE_RC" "$BENCH_RC" "$RESTORED" <<'EOF'
import json, sys
path, label, gate_rc, bench_rc, restored = sys.argv[1:6]
try:
    s = json.load(open(path))["summary"]
    num = lambda k: (f"{s[k]:.1f}" if s.get(k) is not None else "n/a")
    prefill = "/".join(num(f"prefill.{n}.tok_s") for n in ("2000", "8000", "32000"))
    sig = f"prefill 2K/8K/32K={prefill} decode={num('decode.tok_s')} multi1={num('multi.1.aggregate_tok_s')} multi4={num('multi.4.aggregate_tok_s')}"
except Exception as e:
    sig = f"no-summary ({e})"
print(f"arm {label} VERDICT gate_rc={gate_rc} bench_rc={bench_rc} restored={restored} {sig}")
EOF
log "arm $LABEL done: gate_rc=$GATE_RC bench_rc=$BENCH_RC restored=$RESTORED (artifacts: $OUT/ and $REMOTE_OUT/ on the head)"
