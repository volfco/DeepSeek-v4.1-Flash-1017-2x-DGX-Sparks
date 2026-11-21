# combo3 screen: patches 0160-0171 on top of the best config

Image exl3-tp2-234f9080b326 (0129-0171 plus the head-only build fixup 0171z for 0163/0164/0165 compile errors).
Screen run 2026-11-17 00:27-03:18 UTC. Each arm = best config + one switch, one block each, cells warm, validate,
prefill_quick (8K/32K x2), dec1, dec8, think (t1/t4) plus resend/longreply/reuse where relevant. Best blocks
s01 s05 s09 s13 s20 bracket the arms; the noise floor is their spread.

## Gates

a01 (base, all new switches off): exact IDENTICAL 39/39, validate pass, sampled shas equal to the reference.
Every armed block ran validate. Only the all-off block ran the 39-case exact comparison and sampled identity gate.

## Speed vs the mean of the 5 best blocks (%; ttft sign flipped so + = better)

| arm | switch | pd.8000 | pd.32000 | ttft.8000 | dec1 | dec8 | t1 | t4 |
|---|---|---|---|---|---|---|---|---|
| noise (best spread) | | 1.7 | 0.7 | 2.0 | 0.5 | 0.3 | 0.3 | 0.7 |
| a01 base | none | -4.8 | -4.3 | -5.3 | -0.0 | -0.5 | -0.1 | -0.8 |
| s02 absorb | ABSORB_AFTER_PICK=1 | **-26.9** | -0.0 | **-36.2** | +0.2 | +0.0 | +0.1 | +0.0 |
| s03 submit-early | ENGRAM_SUBMIT_EARLY=1 | -0.2 | +0.5 | -0.2 | -0.3 | +0.0 | -0.7 | -1.1 |
| s04 hot2 | EXPERT_HOT_CACHE=hot:2 | -0.5 | +0.3 | -0.9 | **-2.2** | -0.3 | **-2.2** | **-5.5** |
| s06 hot4 | EXPERT_HOT_CACHE=hot:4 | **-11.0** | **-5.1** | **-12.1** | **-2.7** | -0.5 | **-1.9** | **-2.1** |
| s07 scan-suffix | REPLY_SCAN_SUFFIX=1 | -0.8 | +0.2 | -0.5 | +0.2 | +0.3 | -0.0 | +0.0 |
| s08 endstate | PROMPT_ENDSTATE=4 | **-21.5** | -0.1 | **-27.2** | +0.0 | -0.2 | -0.6 | -1.0 |
| s10 endstate-tail | + TAIL_BATCH=4 | **-28.3** | -0.1 | **-55.5** | -0.7 | -0.6 | -0.4 | -0.5 |
| s11 token-align | TOKEN_ALIGN_MB=16 | -0.8 | -0.1 | -1.2 | +0.2 | +0.2 | -0.2 | +0.7 |
| s12 cache-auto | PROMPT_CACHE_AUTO=1 | -0.7 | -0.1 | -1.0 | +0.0 | +0.0 | -0.1 | -0.1 |
| s14/s17 best-noDA | DEVICE_ACCEPT off | +0.3/-1.2 | 0.0/-0.2 | +1.5/-1.3 | +0.2/0.0 | 0.0/-0.1 | -0.3/-0.1 | +0.5/+0.5 |
| s15 triage-0.05 | DRAFT_TRIAGE=conf:0.05 (noDA) | -0.6 | -0.1 | -0.0 | -0.3 | -0.0 | -0.6 | -0.1 |
| s16 triage-0.2 | DRAFT_TRIAGE=conf:0.2 (noDA) | -0.8 | -0.3 | -2.0 | +0.1 | +0.4 | -0.1 | +0.4 |
| s18 rc64 | ENGRAM_ROWCACHE=64 | -1.8 | **-2.7** | -2.1 | -0.3 | -0.6 | **-4.2** | **-2.7** |
| s19 rc64-filter | + ROWCACHE_FILTER=1 | **-13.4** | **-11.2** | **-14.6** | -0.3 | -0.3 | -0.2 | -0.8 |

Bold = outside the noise floor. **No arm is above the noise floor in the positive direction.** The best config
is unchanged; no full-suite (schedule.full.json) runs are warranted for these switches.

(Single-block screen: losses of 2-5% could be partly noise, but nothing here suggests a gain worth confirming.
The 8K-only drops of absorb, endstate and endstate-tail (32K flat) are one block each and 8K is the noisiest point:
s10's two reps were 964 and 1881 tok/s, and best's own resend-cell cold 8K TTFT spans 4.08-4.98 s. Treat those
three as unconfirmed losses, cause unknown; they do not change the verdict since none shows a gain.)

Every arm's flag was confirmed in both ranks' container env (runner arm_proof: no problems in any block).

## Functional checks

- **resend (0170 PROMPT_ENDSTATE, 0171 TAIL_BATCH):** does not work. An identical resent prompt still resumes at
  the grid point (8K: 6144 of ~8150 cached, 32K: 30720 of ~32100) exactly like best; TTFT unchanged (8K ~1.2-1.4 s,
  32K ~1.0 s). Aggregate state counts rose, but they cannot prove a usable full end-state. In 0121 the exact-length lookup rejects `e.at >= prompt.len` ahead of 0170's new eligibility rule. Sharded keeps separately reject off-grid positions in 0015. The s08/s10 rank-0 logs contain 61/69 planner `BadOp` messages; s01/s09/s13 contain zero. Even the aligned 8192-token case resumed at 6144. Fix and test both paths, including 0170's test fixture where 50+128 exceeds 100. Before timing, prove a full store hit on grid and off grid with sharding enabled, a clean planner and actual resume positions. M7 also needs a sequential/concurrent output check and off-grid quality pricing.
- **cache-auto (0164):** no effect on this machine, by design: the budget is still 512 MiB. The auto rule is 70% of
  RAM minus what the loaded server already uses, never below 0.5 GiB; with ~95 GiB of weights resident that is the
  0.5 GiB floor. Not a bug; nothing to gain on a Spark.
- **longreply (0163 scan-suffix):** 6000-token thinking reply l1 104.9 vs best 105.0-105.1 tok/s, l4 141.7 vs
  141.5-141.7. Neutral; identical token sha.
- **load readers (TF_DS_LOAD_READERS=4/8):** slow rank 86.0 s -> 82.7 s (both 4 and 8), fast rank 20.6 -> 20.0 s;
  restart wall 121 s -> 120 s. Negligible.
- **triage:** speed-neutral at both thresholds; no gain to justify it.
- **DEVICE_ACCEPT:** speed-neutral on this image (keep it for the first-request CudaFailed fix).
- **prompt_reuse (s10, s11, s13 - including best):** FAILED the 90% resume threshold: turns of 11.0-11.7K tokens
  resume 10240 (88-91%). Same on best, and the same on the old production image (runs/matrix/hr-base-final.reuse.log:
  10240 cached, 88%, FAILED), so it predates 0160-0171: the resume grid against a growing prompt vs the tool's 90%
  threshold. Not a regression. (s10's turn 0 shows 0 cached where s11/s13 show 10240: the earlier cells' state had
  been evicted/replaced in that arm; the later turns are identical.)

## Verdict

Keep the best config as is (ENGRAM_LEAD, ENGRAM_SCALES=map, ENGRAM_STRICT, TOKEN_CACHE_MB=32, DEVICE_ACCEPT,
COMPRESS_SIDE, PDL, FUSED_COMPRESS2). Leave 0160-0171 switches off. Reject: absorb, hot cache 2/4, endstate(+tail),
rowcache 64 (+filter). Neutral (no reason to turn on): submit-early, scan-suffix, token-align, cache-auto, triage,
load readers. For the patch agent: 0170/0171 lookup and sharded keep defects above;
0163/0164/0165 compile errors (head-only fixup 0171z); 0133 AIO_BATCH 64/256 TooManyRows and 0132 parseCpus
(from combo2).

Engine after the run: serving image 234f9080b326 with the production env (runner RESTORE reported ok=False only
because the image id differs from the old production b40ea899e367). Reverting to the old image means moving
0129-0171 + 0171z out of the head's zig/patches (STATE.md) - not done without the user's say.
