#!/usr/bin/env bash
# k269 suite: exact + sampled identity vs production's sched2 ctl1, the sampled probe with streamed bursts (chunk-gap
# p50/p95), the round-time workload (its profile carries prefill_ms for 0270).
R=runs/overnight-20261122
OUT=$R/k269 REF=$R/sched2/ctl1.exact.json SREF=$R/sched2/ctl1.sampled.json SAMPLED=1 PROBE=1 PROBE_ARGS='--narrow --sampled --stream-bursts' \
  BANNERS='TF_DS_DRAFT_TAPS_DIRECT|TF_DS_PROMPT_PICKS_ONCE|TF_DS_RESUME_ASSEMBLE' ARMS_FILE=$R/k269/arms.txt bash $R/arms.sh
