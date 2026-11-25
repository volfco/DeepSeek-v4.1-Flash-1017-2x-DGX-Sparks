#!/usr/bin/env bash
# k280 hardware window, exactly as staged in runs/k280/README.md: from the workstation's recipe root, rsync the
# README's exact file list to the head, then run the four scripts on the head in the README's order —
# passprof.sh -> run.sh -> l2-run.sh -> topk-run.sh — one at a time, the next only after the previous run.log shows
# the "restored: validate rc=0" verdict and its "finished" line. A step that finishes without that verdict stops
# the window. Nothing here adds or changes an arm: the arms lists are the README's, untouched.
#
# Hazards (per standing rules, echoed at start):
# - Never just sync-head: this checkout's .env is three deploys stale and sync-head would push it over the head's
#   live one. Only the exact file lists below move; .env is never synced.
# - Production must be idle before step 1: every script restarts the server per arm and restores production from
#   .env alone at the end.
# - One step at a time: a missing restore verdict means production may be down or stale; the next step must not run.
# Shortcut: no per-step stall timeout, so a hung step hangs this foreground window visibly; add one if it is ever
# run unattended.
set -euo pipefail
head=${head:-colum@100.95.49.21}
dir=${dir:-DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks}

cat >&2 <<'EOF'
k280 window hazards:
  - never just sync-head: the checkout's .env is three deploys stale; only the README's file lists move
  - production must be idle before this starts
  - one script at a time, each gated on its run.log's "restored: validate rc=0" verdict
EOF

# The README's two rsync commands, verbatim (its exact file list; window.sh itself is not in it).
rsync -a runs/k280/{README.md,arms.txt,run.sh,l2-arms.txt,l2-run.sh,passprof.sh,gate_drafts.py,work.py,topk-arms.txt,topk-run.sh,topk-README.md} \
  "$head:$dir/runs/k280/"
rsync -a runs/overnight-20261122/{arms.sh,table_gpu.py} "$head:$dir/runs/overnight-20261122/"

# name | out dir | script | launch line (the README's own; topk's tee form, detached)
steps=(
  "passprof|runs/k280/passprof|passprof.sh|mkdir -p runs/k280/passprof && setsid nohup bash runs/k280/passprof.sh > runs/k280/passprof/run.log 2>&1 < /dev/null &"
  "run|runs/k280/main|run.sh|mkdir -p runs/k280/main && setsid nohup bash runs/k280/run.sh > runs/k280/main/run.log 2>&1 < /dev/null &"
  "l2-run|runs/k280/l2|l2-run.sh|mkdir -p runs/k280/l2 && setsid nohup bash runs/k280/l2-run.sh > runs/k280/l2/run.log 2>&1 < /dev/null &"
  "topk-run|runs/k280/topk|topk-run.sh|mkdir -p runs/k280/topk && setsid nohup bash -c 'bash runs/k280/topk-run.sh 2>&1 | tee runs/k280/topk/run.log' > /dev/null 2>&1 < /dev/null &"
)

for step in "${steps[@]}"; do
  IFS='|' read -r name out script launch <<<"$step"
  echo "$(date -u +%FT%TZ) $name: launching (log $out/run.log)"
  ssh -n "$head" "cd $dir && bash -c $(printf '%q' "$launch")"
  while :; do
    sleep 60
    log=$(ssh -n "$head" "tail -20 $dir/$out/run.log 2>/dev/null" || true)
    if grep -q ' finished$' <<<"$log"; then
      if grep -q '^restored: validate rc=0$' <<<"$log" && ! grep -q 'restored: validate FAILED\|restore restart failed' <<<"$log"; then
        echo "$(date -u +%FT%TZ) $name: restored: validate rc=0 — next step"
        break
      fi
      echo "$(date -u +%FT%TZ) $name: finished WITHOUT the restore verdict — stopping the window" >&2
      exit 1
    fi
    if ! ssh -n "$head" "pgrep -f '[b]ash runs/k280/$script' > /dev/null"; then
      # the process may have exited between the tail and the pgrep: look once more before failing
      log=$(ssh -n "$head" "tail -20 $dir/$out/run.log 2>/dev/null" || true)
      if grep -q ' finished$' <<<"$log"; then
        if grep -q '^restored: validate rc=0$' <<<"$log" && ! grep -q 'restored: validate FAILED\|restore restart failed' <<<"$log"; then
          echo "$(date -u +%FT%TZ) $name: restored: validate rc=0 — next step"
          break
        fi
        echo "$(date -u +%FT%TZ) $name: finished WITHOUT the restore verdict — stopping the window" >&2
        exit 1
      fi
      echo "$(date -u +%FT%TZ) $name: process gone without a finished line — stopping the window" >&2
      exit 1
    fi
  done
done
echo "$(date -u +%FT%TZ) k280 window finished: all four steps ended with restored: validate rc=0"
