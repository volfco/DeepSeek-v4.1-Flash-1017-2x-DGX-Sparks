#!/usr/bin/env bash
# After the timed campaign has restored and validated production: the same probe on production itself, in a process
# as young as the arms were. This is the timing control. Detached on the head, recipe root.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
D=runs/draft-ladder-20261119
until grep -q " t CAMPAIGN DONE" $D/status.log; do sleep 3; done
mkdir -p $D/production-fresh
docker inspect -f "{{.Config.Image}} started {{.State.StartedAt}}" dsv41-1017-exl3-tf > $D/production-fresh/armed.txt
python3 $D/ladder_probe.py production-fresh $D/production-fresh/probe.json > $D/production-fresh/probe.log 2>&1; rc=$?
docker logs -t dsv41-1017-exl3-tf > $D/production-fresh/engine.log 2>&1
echo "$(date -u +%FT%TZ) production-fresh probe rc=$rc non-loopback=$(grep "POST /v1/" $D/production-fresh/engine.log | grep -vc "\[tensorfold\] 127\.0\.0\.1 ")" >> $D/status.log
