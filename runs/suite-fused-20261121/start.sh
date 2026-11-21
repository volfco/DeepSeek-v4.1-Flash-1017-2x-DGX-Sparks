#!/usr/bin/env bash
# suite-fused-20261121: production's best config (image exl3-tp2-ea522d353626, .env as deployed) against the same .env
# on tensorfold-dsv41:bench-prune-six-20261120 (production source + the fused kernels 0193-0199) with the four fused
# switches on. Four blocks, ABBA. Run detached on the head from the recipe root. The runner restores production from
# .env when the schedule is done, on STOP, on any exception, or when a non-loopback client shows up.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
O=runs/suite-fused-20261121
nohup env CAMPAIGN_OUT=$O CAMPAIGN_IMAGE=exl3-tp2-ea522d353626 CAMPAIGN_BASE='{}' CAMPAIGN_PROD='{}' CAMPAIGN_CLOCK_ABORT=0 \
  python3 $O/run_suite.py >> $O/runner.log 2>&1 < /dev/null &
echo "runner pid $!"
