#!/usr/bin/env bash
# Head: after combo2's runner exits, install 0160-0171, build, set up runs/combo3, launch its runner.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
log(){ echo "$(date -u +%FT%TZ) chain: $*" >> runs/combo3/chain.log; }
while kill -0 507660 2>/dev/null; do sleep 60; done
grep -q "CAMPAIGN DONE" runs/combo2/status.log || { log "combo2 runner exited without CAMPAIGN DONE; stopping"; exit 1; }
log "combo2 done; installing 0160-0171"
cp runs/combo3/newpatches/*.patch zig/patches/ && cp runs/combo3/newpatches/README.md zig/patches/README.md
(cd zig/patches && sha256sum 0*.patch) | diff -q - runs/combo3/patches.sha256 >/dev/null || { log "patch set differs from patches.sha256; stopping"; exit 1; }
export PATH=$HOME/.local/bin:$PATH
TAG=$(bash -c 'source scripts/config.sh >/dev/null 2>&1; echo "${IMAGE##*:}"')
log "building $TAG"
./scripts/prepare.sh > runs/combo3/build.log 2>&1 || { log "prepare.sh failed rc=$?"; exit 1; }
log "built; launching runner"
cp -r runs/combo2/corpus runs/combo2/production.rank* runs/combo3/
CAMPAIGN_OUT=runs/combo3 CAMPAIGN_IMAGE=$TAG CAMPAIGN_PROD="{}" CAMPAIGN_CLOCK_ABORT=0 \
  nohup python3 -u scripts/bench_overnight.py > runs/combo3/runner.out 2>&1 < /dev/null &
log "runner pid=$!"
