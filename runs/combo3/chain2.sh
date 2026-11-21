#!/usr/bin/env bash
# Head: build 0001-0171 + the bench fixup, then launch the combo3 runner.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
log(){ echo "$(date -u +%FT%TZ) chain2: $*" >> runs/combo3/chain.log; }
cp runs/combo3/0171z-bench-build-fixup.patch zig/patches/
export PATH=$HOME/.local/bin:$PATH
TAG=$(bash -c 'source scripts/config.sh >/dev/null 2>&1; echo "${IMAGE##*:}"')
log "building $TAG (with 0171z fixup)"
./scripts/prepare.sh > runs/combo3/build2.log 2>&1 || { log "prepare.sh failed rc=$?"; exit 1; }
log "built; launching runner"
cp -r runs/combo2/corpus runs/combo2/production.rank* runs/combo3/
CAMPAIGN_OUT=runs/combo3 CAMPAIGN_IMAGE=$TAG CAMPAIGN_PROD="{}" CAMPAIGN_CLOCK_ABORT=0 \
  nohup python3 -u scripts/bench_overnight.py > runs/combo3/runner.out 2>&1 < /dev/null &
log "runner pid=$!"
