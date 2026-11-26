#!/usr/bin/env bash
# Production: KV_SHARD=1 -> KV_SHARD=0 (same image bench-fused-next-1421187), applied once the engine has had zero
# traffic for IDLE_S (90) seconds. Requested by the operator 2026-11-26 after runs/observe-20261126/REPORT.md.
# Effect: rank 1's pool 2,101,248 -> 1,048,576 positions (CONTEXT), the TF_DS_KV_SHARD* variables go, and
# PREFILL_SLICE=5 takes effect (slicing is disabled under KV sharding). Cost: the pool is about 545K rows short of
# the 1.59M-row peak seen 2026-11-26, so long kept conversations get evicted sooner.
# Gates (references were recorded with KV_SHARD=1; shard-off identity has no prior record): exact panel IDENTICAL
# 39/39 vs runs/deploy-10c9866-20261122/exact.json and validate.py rc=0 vs runs/matrix/base.validate.json.
# Any failure -> restore .env, restart, re-validate. Results stay in $out for review either way.
#   Arm (head, repo root):  setsid nohup bash runs/deploy-kvshard0-20261126/deploy.sh > runs/deploy-kvshard0-20261126/run.log 2>&1 < /dev/null &
#   Status:                 cat runs/deploy-kvshard0-20261126/status; tail runs/deploy-kvshard0-20261126/run.log
#   Cancel before it fires: kill $(cat runs/deploy-kvshard0-20261126/pid)
#   Undo after it fired:    cp ~/dsv41-env-backups/.env.bak-20261126-pre-kvshard0 .env && PREPARE=0 ./start.sh restart
set -euo pipefail
out=runs/deploy-kvshard0-20261126; mkdir -p "$out"
exec 9> "$out/lock"; flock -n 9 || { echo "another deploy.sh holds $out/lock"; exit 3; }
echo $$ > "$out/pid"
C=dsv41-1017-exl3-tf
IMG=tensorfold-dsv41:bench-fused-next-1421187
IDLE_S=${IDLE_S:-90}
bak=$HOME/dsv41-env-backups/.env.bak-20261126-pre-kvshard0
worker=$(grep -m1 '^WORKER=' .env | cut -d= -f2-)
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
st() { echo "$(date -u +%FT%TZ) $*" > "$out/status"; echo "$(date -u +%FT%TZ) $*"; }
pre() {
  grep -qx "IMAGE=$IMG" .env || { st "ABORTED: .env does not name IMAGE=$IMG"; exit 5; }
  grep -qx 'KV_SHARD=1' .env || { st "ABORTED: .env lacks KV_SHARD=1"; exit 5; }
  [[ $(grep -cE '^(TF_DS_)?KV_SHARD(_[A-Z]+)?=' .env) == 1 ]] || { st "ABORTED: .env sets other KV_SHARD switches"; exit 5; }
}
# idle = /health answers with 0 connections and 0 waiting continuously for IDLE_S s (polled every 5 s; a failed or
# non-zero poll resets the clock), and no external POST /v1/ in the container log over that window
busy() {
  curl -s -m 5 http://127.0.0.1:8000/health | python3 -I -c 'import json,sys; l=json.load(sys.stdin)["live"]; sys.exit(0 if (l["connections"], l["waiting"]) == (0, 0) else 1)' 2>/dev/null && return 1 || return 0
}
wait_idle() {
  local since=
  while :; do
    if busy; then since=; else
      [[ -n $since ]] || since=$(date +%s)
      if (( $(date +%s) - since >= IDLE_S )); then
        local n; n=$(docker logs --since "${IDLE_S}s" $C 2>&1 | grep 'POST /v1/' | grep -vc '\[tensorfold\] 127\.0\.0\.1 ' || true)
        [[ $n == 0 ]] && return 0
        since=
      fi
    fi
    sleep 5
  done
}
pre
for r in 0 1; do
  if [[ $r == 0 ]]; then docker image inspect $IMG >/dev/null; else ssh -n -o BatchMode=yes "$worker" "docker image inspect $IMG >/dev/null"; fi || { st "ABORTED: $IMG missing on rank$r"; exit 6; }
done
st "armed: waiting for ${IDLE_S}s of zero traffic"
wait_idle
pre   # re-check at fire time: the wait may have lasted hours
st "idle for ${IDLE_S}s: switching to KV_SHARD=0"
mkdir -p "$HOME/dsv41-env-backups"
[[ -e "$bak" ]] || install -m 600 .env "$bak"
cmp -s .env "$bak" || { st "ABORTED: $bak differs from .env"; exit 7; }
sed -i 's/^KV_SHARD=1$/KV_SHARD=0/' .env
printf '# 2026-11-26: KV_SHARD=0 (operator request; runs/observe-20261126, runs/deploy-kvshard0-20261126); undo: restore %s\n' "$bak" >> .env
diff "$bak" .env || true
date -u +%FT%TZ > "$out/started.txt"

insp() { if [[ $1 == 0 ]]; then bash -c "$2" < /dev/null; else ssh -n -o BatchMode=yes "$worker" "$2"; fi; }
healthy() { for i in $(seq 120); do curl -sf -m 5 http://127.0.0.1:8000/health > "$out/$1.health.json" && return 0; sleep 5; done; return 1; }
gate_validate() {  # $1: label
  python3 -I tools/validate.py http://127.0.0.1:8000 --model DeepSeek-V4.1-Flash-1017 --save "$out/$1.validate.json" \
    --compare runs/matrix/base.validate.json > "$out/$1.validate.log" 2>&1
}
rollback() {  # $1: reason
  st "ROLLING BACK: $1"
  cp "$bak" .env
  if PREPARE=0 ./start.sh restart > "$out/rollback.log" 2>&1 && healthy rollback && gate_validate rollback; then
    st "ROLLED BACK (KV_SHARD=1 restored, validate rc=0): $1"; exit 20
  fi
  st "ROLLBACK FAILED - PRODUCTION NEEDS ATTENTION ($1); see $out/rollback.log"; exit 99
}
trap 'rc=$?; (( rc == 0 || rc == 20 || rc == 99 )) || { trap - EXIT; rollback "unexpected exit rc=$rc at line $LINENO"; }' EXIT

rc=0; PREPARE=0 ./start.sh restart > "$out/restart.log" 2>&1 || rc=$?
echo "$rc" > "$out/restart.rc"
if (( rc != 0 )) && ! grep -q 'the smoke test request failed' "$out/restart.log"; then rollback "restart rc=$rc"; fi
healthy new || rollback "health never answered"
for r in 0 1; do
  lab=$(insp $r "docker inspect -f '{{index .Config.Labels \"tf.recipe.port\"}} {{.Config.Image}}' $C") || rollback "inspect rank$r"
  [[ $lab == *shard=0* && $lab == *pool=1048576* && $lab == *" $IMG" ]] || rollback "rank$r runs: $lab"
  echo "rank$r $lab"
done > "$out/armed.txt"
python3 -I -c 'import json,sys; h=json.load(open(sys.argv[1])); sys.exit(1 if h.get("kv_shard") else 0)' "$out/new.health.json" || rollback "health still reports kv_shard"
st "both ranks on KV_SHARD=0: gating"
python3 -I tools/exact.py record --url http://127.0.0.1:8000 --out "$out/exact.json" --label kvshard0 --like runs/deploy-10c9866-20261122/exact.json > "$out/exact.log" 2>&1 || rollback "exact record failed"
cmpres=$(python3 -I tools/exact.py compare runs/deploy-10c9866-20261122/exact.json "$out/exact.json" | tail -1); echo "$cmpres" | tee "$out/exact.compare.txt"
gate_validate new && echo "validate rc=0" || rollback "validate failed: $(tail -1 "$out/new.validate.log")"
[[ $cmpres == *IDENTICAL* ]] || rollback "exact panel not identical: $cmpres"
date -u +%FT%TZ > "$out/finished.txt"
trap - EXIT
st "DONE: production on KV_SHARD=0, exact IDENTICAL, validate rc=0"
