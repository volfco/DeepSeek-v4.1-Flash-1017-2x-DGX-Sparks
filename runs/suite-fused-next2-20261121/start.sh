#!/usr/bin/env bash
# suite-fused-next2-20261121: bench-fused-next-d5ecba0 (base) against bench-fused-next-84806bb (0254 topk_indices_cut,
# 0255 cand_fast8, 0256/0258 thread-safe fused_trace with 0254/0255 arms, 0257 planKvSource2). ABBA s01-s04 with the
# fixed-input replay and staggered dec8 cells, then s05 = the new image with TF_DS_FUSED_TRACE=1 on both ranks (traced
# replays of the 8,192/16,384-block buckets and the 200K fallback). Run detached on the head from the recipe root. The
# runner restores production from .env when the schedule is done, on STOP, on any exception, or when a non-loopback
# client shows up.
cd ~/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks || exit 1
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
O=runs/suite-fused-next2-20261121
nohup env CAMPAIGN_OUT=$O CAMPAIGN_IMAGE=bench-fused-next-d5ecba0 CAMPAIGN_BASE='{}' CAMPAIGN_PROD='{}' CAMPAIGN_CLOCK_ABORT=0 \
  python3 $O/run_suite.py >> $O/runner.log 2>&1 < /dev/null &
echo "runner pid $!"
