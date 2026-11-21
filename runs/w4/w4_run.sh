#!/usr/bin/env bash
# W4 (docs/benchmark-gaps-and-moe-review.md section 6): is the decode round's MoE bus-bound or trellis-decode-bound?
# One detached run on the head, from the repo root, on the image the pair serves (no build):
#   1. record both served ranks (image, command, TF_*/TENSORFOLD_*/HF_HUB env), /health, clocks; wait for idle
#   2. ./stop.sh, then the pure-read probe (0142, --synthetic; --spec has no source of device addresses)
#   3. tf-dsv41-lanes as both ranks (runs/ultrafast/uf_prof.sh's recipe, spill stripped), eager --profile 2:
#      core cells (TF_DS_EXPERT_BYTES=1 at 1 and 4 streams, a CUDA_LAUNCH_BLOCKING=1 twin of each for a read-back that
#      cannot race the stream, a no-flag control), then optional cells while the downtime is under OPT_CAP_S
#   4. EXIT trap: ./start.sh restart (production, head .env, nothing exported), health, a chat, env diff, validate
# Attribution only: the profile syncs every round.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PORT=8000
O=runs/w4; P=$HOME/.cache/w4-private; C=dsv41-1017-exl3-tf
IMG=tensorfold-dsv41:exl3-tp2-234f9080b326
W="$(grep -m1 '^WORKER=' .env | cut -d= -f2-)"
OPT_CAP_S=${OPT_CAP_S:-1260}
mkdir -p $O $P
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $O/status.log; }
wssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 "$W" "$@"; }
smi='nvidia-smi --query-gpu=clocks.gr,clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader'
filt() { python3 -I - "$1" <<'PY'
import json, re, sys
c = json.load(open(sys.argv[1]))[0]
print(json.dumps({"image": c["Config"]["Image"], "cmd": c["Config"]["Cmd"],
                  "env": sorted(e for e in c["Config"]["Env"] if re.match(r"^(TF_|TENSORFOLD_|HF_HUB)", e))}, indent=1))
PY
}
metric() { curl -s -m 5 127.0.0.1:8000/metrics | awk -v k="$1" '$1==k {print $2}'; }

STOPPED=0; SAMPLER=
restore() {
  trap - EXIT
  [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null
  [[ $STOPPED == 1 ]] || { log "not stopped: nothing to restore"; return; }
  docker rm -f tf-w4-0 tf-w4-probe >/dev/null 2>&1; wssh "docker rm -f tf-w4-1 tf-w4-probe >/dev/null 2>&1"
  log "RESTORE start: ./start.sh restart (head .env, no switch exported)"
  local rc ok=0
  for attempt in 1 2; do
    ./start.sh restart > $O/restore.restart.log.$attempt 2>&1; rc=$?
    log "restore start.sh attempt $attempt rc=$rc"
    if [[ $rc == 0 ]]; then ok=1; break; fi
    if grep -q "smoke test request failed" $O/restore.restart.log.$attempt; then ok=2; break; fi
    if grep -q "already running" $O/restore.restart.log.$attempt; then sleep 300; else sleep 60; fi
  done
  for i in $(seq 1 120); do
    h=$(curl -s -m 5 127.0.0.1:8000/health)
    if python3 -I -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get("status")=="ok" and not d["pair"]["broken"] else 1)' "$h" 2>/dev/null; then break; fi
    sleep 5
  done
  log "API UP: health=$(python3 -I -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("status"), "broken=%s" % d["pair"]["broken"])' "$h" 2>/dev/null || echo down)"
  for attempt in 1 2; do
    python3 -I - > $O/restore.chat.$attempt.json 2>&1 <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import checklib as C
S = C.Server("http://127.0.0.1:8000", timeout=300)
r = S.post_json("/v1/chat/completions", S.body([{"role": "user", "content": "Reply with OK."}], 16))
print(json.dumps({"error": r.get("error"), "usage": r.get("usage"),
                  "text": ((r.get("choices") or [{}])[0].get("message") or {}).get("content")}))
PY
    log "restore chat attempt $attempt: $(cat $O/restore.chat.$attempt.json | cut -c1-300)"
    grep -q '"error": null' $O/restore.chat.$attempt.json && break
  done
  docker inspect $C > $P/post.rank0.inspect.json; wssh "docker inspect $C" > $P/post.rank1.inspect.json
  for r in 0 1; do filt $P/post.rank$r.inspect.json > $O/post.rank$r.json; done
  for r in 0 1; do
    if diff $O/pre.rank$r.json $O/post.rank$r.json > $O/restore.envdiff.rank$r.txt; then log "rank$r image/cmd/env MATCH"; else log "rank$r image/cmd/env DIFFER: $(tr '\n' ' ' < $O/restore.envdiff.rank$r.txt | cut -c1-600)"; fi
  done
  timeout 1800 python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save $O/restore.validate.json \
    --compare runs/matrix/base.validate.json > $O/restore.validate.log 2>&1; log "restore validate rc=$? $(tail -1 $O/restore.validate.log)"
  curl -s -m 5 127.0.0.1:8000/health > $O/post.health.json
  { echo "head $($smi)"; echo "worker $(wssh "$smi")"; } > $O/post.clocks.txt
  log "RESTORE done (start rc=$rc ok=$ok)"
}
trap restore EXIT
trap 'log "signal: restoring"; exit 1' INT TERM HUP

log "W4 start"
docker inspect $C > $P/rank0.inspect.json || { log "no rank 0 container"; exit 1; }
wssh "docker inspect $C" > $P/rank1.inspect.json || { log "no rank 1 container"; exit 1; }
for r in 0 1; do filt $P/rank$r.inspect.json > $O/pre.rank$r.json; done
for r in 0 1; do grep -q "\"$IMG\"" $O/pre.rank$r.json || { log "rank$r is not on $IMG: refusing"; exit 1; }; done
curl -s -m 5 127.0.0.1:8000/health > $O/pre.health.json
{ echo "head $($smi)"; echo "worker $(wssh "$smi")"; } > $O/pre.clocks.txt
log "pre: $(tr '\n' ' ' < $O/pre.clocks.txt)"

for i in $(seq 1 60); do
  run=$(metric tensorfold:requests_running); wait_=$(metric tensorfold:requests_waiting)
  [[ "$run" == "0" || "$run" == "0.0" ]] && [[ "$wait_" == "0" || "$wait_" == "0.0" ]] && break
  log "busy: running=$run waiting=$wait_; waiting 10 s"; sleep 10; run=busy
done
[[ "$run" == "0" || "$run" == "0.0" ]] || { log "never idle in 10 min: not stopping"; exit 0; }
log "idle (requests_running=$run waiting=$wait_): stopping the pair"
STOPPED=1
./stop.sh > $O/stop.log 2>&1; log "API DOWN: stop.sh rc=$?"
T_DOWN=$(date +%s)
docker ps --format '{{.Names}}' | grep -qx $C && log "WARN rank 0 container still present"
wssh "docker ps --format '{{.Names}}'" | grep -qx $C && log "WARN rank 1 container still present"
log "mem after stop: head $(free -g | awk '/Mem:/{print $7}') GiB avail, worker $(wssh "free -g" | awk '/Mem:/{print $7}') GiB avail"

( while :; do echo "$(date -u +%T) head $($smi) | worker $(wssh "$smi")"; sleep 5; done ) > $O/clocks.log 2>&1 &
SAMPLER=$!

# ---- the pure-read probe (0142): synthetic ranges (one cudaMalloc'd buffer), decode-shaped grid unless swept
probe() {
  local lab=$1; shift
  timeout 300 docker run --rm --gpus all --name tf-w4-probe $IMG tf-dsv41-probe "$@" > $O/probe.$lab.log 2>&1
  local rc=$?; docker rm -f tf-w4-probe >/dev/null 2>&1
  log "probe $lab rc=$rc results=$(grep -c '^RESULT' $O/probe.$lab.log) $(grep -m1 -i 'error\|panic' $O/probe.$lab.log | cut -c1-200)"
}
EB=6469984   # 6.47 MB (the file-average expert place a rank), rounded to 16 bytes
timeout 400 ssh -o BatchMode=yes -o ConnectTimeout=10 "$W" "docker run --rm --gpus all --name tf-w4-probe $IMG tf-dsv41-probe --synthetic --experts 13 --expert-bytes $EB --tiles 9 --reps 50 > /tmp/w4.probe.w13t9.log 2>&1; docker run --rm --gpus all --name tf-w4-probe $IMG tf-dsv41-probe --synthetic --experts 16 --expert-bytes 67108864 --tiles 64 --reps 20 > /tmp/w4.probe.w16big.log 2>&1" &
WPROBE=$!
probe h13t9 --synthetic --experts 13 --expert-bytes $EB --tiles 9 --reps 50
probe h13t18 --synthetic --experts 13 --expert-bytes $EB --tiles 18 --reps 50
probe h13t36 --synthetic --experts 13 --expert-bytes $EB --tiles 36 --reps 50
probe h16t9 --synthetic --experts 16 --expert-bytes $EB --tiles 9 --reps 50
probe h16t36 --synthetic --experts 16 --expert-bytes $EB --tiles 36 --reps 50
probe h13t9gap --synthetic --experts 13 --expert-bytes $EB --stride-bytes $((EB * 4)) --tiles 9 --reps 50
probe h16big --synthetic --experts 16 --expert-bytes 67108864 --tiles 64 --reps 20
probe h13t9b --synthetic --experts 13 --expert-bytes $EB --tiles 9 --reps 50
wait $WPROBE
wssh "cat /tmp/w4.probe.w13t9.log" > $O/probe.w13t9.log; wssh "cat /tmp/w4.probe.w16big.log" > $O/probe.w16big.log
log "probe worker: $(grep -c '^RESULT' $O/probe.w13t9.log)+$(grep -c '^RESULT' $O/probe.w16big.log) results"

# ---- batch-driver cells. cell LABEL REQFILE PARALLEL TIMEOUT_S [both:K=V | r0:K=V | unset:K]...
cell() {
  local lab=$1 req=$2 par=$3 tmo=$4; shift 4
  local a0=() a1=() x
  for x in "$@"; do
    case $x in
      both:*) a0+=(--env "${x#both:}"); a1+=(--env "${x#both:}") ;;
      r0:*) a0+=(--env "${x#r0:}") ;;
      unset:*) a0+=(--unset "${x#unset:}"); a1+=(--unset "${x#unset:}") ;;
    esac
  done
  local line1 line0
  line1=$(python3 -I $O/w4_cmd.py 1 $P/rank0.inspect.json $P/rank1.inspect.json tf-w4-1 --graphs 0 --profile 1 "${a1[@]}") || { log "$lab cmd1 failed"; return 1; }
  line0=$(python3 -I $O/w4_cmd.py 0 $P/rank0.inspect.json $P/rank1.inspect.json tf-w4-0 --graphs 0 --profile 2 \
    --requests $req --parallel $par --reqdir "$PWD/$O/req" "${a0[@]}") || { log "$lab cmd0 failed"; return 1; }
  echo "$line1" > $O/$lab.rank1.cmd; echo "$line0" > $O/$lab.rank0.cmd
  log "$lab launch (req=$req parallel=$par env=[$*])"
  wssh "docker rm -f tf-w4-1 >/dev/null 2>&1; $line1" > $O/$lab.rank1.id 2>&1 || { log "$lab rank1 launch failed: $(tail -2 $O/$lab.rank1.id)"; return 1; }
  docker rm -f tf-w4-0 >/dev/null 2>&1
  eval "$line0" > $O/$lab.rank0.id 2>&1 || { log "$lab rank0 launch failed: $(tail -2 $O/$lab.rank0.id)"; wssh "docker rm -f tf-w4-1" >/dev/null 2>&1; return 1; }
  local rc
  rc=$(timeout $tmo docker wait tf-w4-0); log "$lab rank0 exit=${rc:-timeout}"
  docker logs tf-w4-0 > $O/$lab.rank0.log 2>&1
  wssh "timeout 180 docker wait tf-w4-1 >/dev/null; docker logs tf-w4-1" > $O/$lab.rank1.log 2>&1
  docker rm -f tf-w4-0 >/dev/null 2>&1; wssh "docker rm -f tf-w4-1" >/dev/null 2>&1
  log "$lab rank0: $(grep -h '"profile"\|"expert_prune"\|"expert_bytes"\|"all_equal"' $O/$lab.rank0.log | tr '\n' ' ' | cut -c1-1400)"
  log "$lab rank0 moe: $(grep -h '"phases_ms"' $O/$lab.rank0.log | grep -o '"moe": [0-9.]*')"
  log "$lab rank1: $(grep -h 'follow_profile\|"expert_bytes"' $O/$lab.rank1.log | tr '\n' ' ' | cut -c1-600)"
  [[ "$rc" == "0" ]]
}
# the known first-request failure: one identical retry
cellr() {
  cell "$@" && return 0
  if grep -q 'CUDA_ERROR_INVALID_VALUE' $O/$1.rank0.log $O/$1.rank1.log 2>/dev/null; then
    for f in $O/$1.rank0.log $O/$1.rank1.log; do [[ -f $f ]] && mv $f $f.try1; done
    log "$1: known cuLaunchKernel failure, retrying once unchanged"; cell "$@"
  fi
}
elapsed() { echo $(( $(date +%s) - T_DOWN )); }
opt() { if (( $(elapsed) < OPT_CAP_S )); then cellr "$@"; else log "SKIP $1 (downtime $(elapsed) s >= cap $OPT_CAP_S s)"; fi; }

XB=both:TF_DS_EXPERT_BYTES=1
LB=r0:CUDA_LAUNCH_BLOCKING=1
cellr e-r1-xb r1.json 1 900 $XB
cellr e-r4-xb r4.json 4 900 $XB
cellr e-r1-lb r1.json 1 900 $XB $LB
cellr e-r1-ctl r1.json 1 900
cellr e-r4-lb r4.json 4 900 $XB $LB
log "core done at downtime $(elapsed) s"
opt k3-r1 r4.json 1 900 $XB both:TF_DS_FIX_K=3
opt k3-r1-np r4.json 1 900 $XB both:TF_DS_FIX_K=3 unset:TF_DS_T13_EXPERT_PRUNE
opt k0-r4 r4.json 4 900 $XB both:TF_DS_FIX_K=0
opt e-r8 r8.json 8 900 $XB
opt pf pf.json 1 900 both:TF_DS_PREFILL_PROFILE=1
log "cells done at downtime $(elapsed) s"
exit 0
