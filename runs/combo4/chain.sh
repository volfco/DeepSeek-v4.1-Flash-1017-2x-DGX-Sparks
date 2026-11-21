#!/usr/bin/env bash
# Head: wait for the combo4 image build (0182-0190), then launch the screen runner.
set -u
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks
log(){ echo "$(date -u +%FT%TZ) chain: $*" >> runs/combo4/chain.log; }
while pgrep -f scripts/prepare.sh >/dev/null; do sleep 30; done
export PATH=$HOME/.local/bin:$PATH
TAG=$(bash -c 'source scripts/config.sh >/dev/null 2>&1; echo "${IMAGE##*:}"')
docker image inspect "tensorfold-dsv41:$TAG" >/dev/null 2>&1 || { log "image $TAG missing after build; see build.log"; exit 1; }
log "built $TAG; launching runner"
cp -r runs/combo2/corpus runs/combo2/production.rank* runs/combo4/
CAMPAIGN_OUT=runs/combo4 CAMPAIGN_IMAGE=$TAG CAMPAIGN_PROD="{}" CAMPAIGN_CLOCK_ABORT=0 \
  nohup python3 -u scripts/bench_overnight.py > runs/combo4/runner.out 2>&1 < /dev/null &
log "runner pid=$!"
