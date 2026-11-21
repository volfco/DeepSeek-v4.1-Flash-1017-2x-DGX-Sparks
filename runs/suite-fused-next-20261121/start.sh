#!/usr/bin/env bash
# suite-fused-next-20261121: production (bench-prune-six-20261120 with the four fused switches, .env as deployed) against
# the same .env on tensorfold-dsv41:bench-fused-next-d5ecba0 (this round's 0213-0253). ABBA s01-s04, then s05 = the new
# image with TF_DS_PDL=1 (0250's split only acts under PDL). Run detached on the head from the recipe root. The runner
# restores production from .env when the schedule is done, on STOP, on any exception, or when a non-loopback client
# shows up.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
O=runs/suite-fused-next-20261121
nohup env CAMPAIGN_OUT=$O CAMPAIGN_IMAGE=bench-prune-six-20261120 CAMPAIGN_BASE='{}' CAMPAIGN_PROD='{}' CAMPAIGN_CLOCK_ABORT=0 \
  python3 $O/run_suite.py >> $O/runner.log 2>&1 < /dev/null &
echo "runner pid $!"
