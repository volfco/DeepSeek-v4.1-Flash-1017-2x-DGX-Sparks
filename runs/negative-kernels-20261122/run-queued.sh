#!/usr/bin/env bash
# Run on the head from the staged directory, with an independent log.
set -euo pipefail
cd "$(dirname "$0")"
stage=$PWD
repo=$HOME/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
suite=$repo/runs/overnight-20261122/kall
container=dsv41-1017-exl3-tf
deadline=$((SECONDS + 21600))
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

restored() {
    grep -q ' finished$' "$suite/run.log" &&
        grep -q '^restored: validate rc=0$' "$suite/run.log" &&
        ! pgrep -f '^bash runs/overnight-20261122/(kall/run|arms)\.sh$' >/dev/null &&
        [[ $(docker inspect -f '{{.Config.Image}}' "$container") == "$(sed -n 's/^IMAGE=//p' "$repo/.env" | head -1)" ]]
}

suite_ready() {
    local arm
    restored || return 1
    for arm in ctl1 all3a ctl2 all3b all4a all4b; do
        [[ -s "$suite/$arm.json" && -s "$suite/$arm.probe.json" &&
           -s "$suite/$arm.exact.json" && -s "$suite/$arm.sampled.json" &&
           -s "$suite/$arm.banner.txt" ]] || return 1
    done
}

idle_counters() {
    curl -fsS --max-time 3 http://127.0.0.1:8000/health | python3 -c '
import json, sys
h = json.load(sys.stdin)
live = h["live"]
if live["connections"] != 0 or live["waiting"] != 0:
    raise SystemExit(1)
counters = [h.get(name) for name in ("prompt_tokens_total", "generation_tokens_total")]
if not all(type(value) is int and value >= 0 for value in counters):
    raise SystemExit(1)
print(*counters)
'
}

log 'Queued behind the six kall arms and successful production restore.'
until suite_ready && [[ -e tabulation.done ]]; do
    (( SECONDS < deadline )) || { log 'Timed out waiting for suite and tabulation completion.'; exit 1; }
    sleep 20
done
log 'Suite, restore validation, and workstation tabulation finished.'
image=$(docker image inspect --format '{{.Id}}' tensorfold-dsv41:bench-fused-next-0f89e5a)
exec 9>"$HOME/tf-bench/gpu.lock"
until suite_ready && before=$(idle_counters) && flock -n 9; do
    (( SECONDS < deadline )) || { log 'Timed out waiting for the GPU lock and idle production.'; exit 1; }
    sleep 20
done
while true; do
    sleep 30
    if suite_ready && after=$(idle_counters) && [[ "$after" == "$before" ]]; then
        break
    fi
    flock -u 9
    until suite_ready && before=$(idle_counters) && flock -n 9; do
        (( SECONDS < deadline )) || { log 'Timed out waiting for a quiet interval.'; exit 1; }
        sleep 20
    done
    (( SECONDS < deadline )) || { log 'Timed out waiting for a quiet interval.'; exit 1; }
done
log "Launching on $(hostname), benchmark image $image"
status=0
for bench in markov compress; do
    log "Starting $bench"
    if timeout 120s docker run --rm --gpus all --network host \
        --mount "type=bind,src=$stage,dst=/bench,readonly" \
        --mount type=bind,src=/nfs1/tensorfold-dsv41,dst=/artifacts,readonly \
        --entrypoint /usr/bin/python3 "$image" \
        -B "/bench/runs/negative-kernels-20261122/$bench.py" \
        >"$bench.jsonl" 2>"$bench.stderr"; then
        log "Completed $bench"
    else
        code=$?
        log "$bench exited $code"
        status=$code
    fi
    suite_ready && after=$(idle_counters) && [[ "$after" == "$before" ]] || {
        log 'Production state changed; ending benchmark job.'
        exit 1
    }
done
log "Finished with status $status"
exit "$status"
