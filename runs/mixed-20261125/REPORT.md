# mixed cell: one cold 128K prefill costs decoders 62% of throughput; the interleave knob only chooses who suffers

Window 2026-11-25 09:16:42Z → 10:14:37Z. Image `tensorfold-dsv41:bench-fused-next-2bb052f`
(`TF_DS_ENGINE_ID=tensorfold@fbbc97132cd0+ea522d353626`), kit `kit-abe07c7d`, `DRAFT_BLOCK=6`,
`DRAFT_BLOCK_KEPT=2`, `PARALLEL=8`, 2 ranks. Production restored and validated after every arm
(`restored: validate rc=0`); production left at 250 ms / 5 layers.

This is the first cell that measures the live interference directly. Live showed a 166–192K cold pass
stretching other streams' rounds from 92 to 155–312 ms and prefill dropping to 1,670 tok/s from
2,040–2,100. No earlier suite had a cold prefill beside decoding streams.

## Cell design

`runs/mixed-20261125/mixed.py`: 7 decoders at 8,192 depth × 1,024 tokens, prompts warmed with a 1-token
request so they hit the prompt cache and decode immediately; one 131,072-token cold prefill launched
1.5 s later, into the decode period. Control = the same 7 decoders with no prefill. Metrics from the
engine's `done` line per request.

## Results

| arm | dec tok/s | ms/round | prefill tok/s | prefill s | wall s |
|---|---|---|---|---|---|
| **ctl** (7 decoders, no prefill) | **24.40** | 95.5 | – | – | 53.88 |
| ctl250s5 (**production**) | 9.37 | 251.2 | 1662 | 78.9 | 120.41 |
| i250s10 | 9.29 | 255.4 | 1654 | 79.2 | 119.94 |
| i250s20 | 9.27 | 246.1 | 1658 | 79.1 | 119.84 |
| i500s5 | 9.24 | 249.9 | 1367 | 95.9 | 120.69 |
| i500s10 | 9.23 | 247.6 | 1370 | 95.7 | 120.10 |
| i500s20 | 9.11 | 245.5 | 1364 | 96.1 | 119.63 |
| i1000s5 | 13.27 | 184.7 | 1231 | 106.5 | 120.49 |
| i1000s10 | 12.31 | 181.7 | 1211 | 108.2 | 122.31 |
| i1000s20 | 12.87 | 184.3 | 1226 | 106.9 | 120.84 |

All 10 arms exact `IDENTICAL` against `runs/deploy-10c9866-20261122/exact.json` (39 cases), banners 0
errors on both ranks. Knobs verified in each `.armed.txt` (250/500/1000 ms and slice 5/10/20).

## Verdicts

1. **A single cold 128K prefill costs the decoders 62% of their throughput** at production's 250/5:
   9.37 vs 24.40 tok/s per decoder, per-round 251 vs 95 ms (2.6×). That is the first direct
   measurement of the live interference, and it is at the top of the range the live logs implied
   (1.7–3.4×).

2. **`TF_DS_PREFILL_SLICE` is inert.** Slice 5 / 10 / 20 is identical at every interval
   (250: 9.37/9.29/9.27; 500: 9.24/9.23/9.11; 1000: 13.27/12.31/12.87). Dead knob in this cell.

3. **`TF_DS_PREFILL_INTERLEAVE_MS` is a pure priority dial, not a free win.** 250 ms keeps the prefill
   near its solo rate (1662 tok/s, 81% of the 2,040–2,100 solo) and starves the decoders (49% of solo).
   1000 ms does the reverse: decoders reach 13.27 tok/s (70% of solo) and the prefill falls to 1231
   tok/s (60% of solo, 106 s). The knob moves work between the two; it does not create it.

4. **500/5 is dominated.** Prefill 1367 tok/s (18% worse than 250's 1662) with decoders at 9.24 tok/s
   (no better than 250's 9.37). Strictly worse than production's 250/5 on both axes. The 250→500→1000
   decoder curve is flat-then-jump (9.37 / 9.24 / 13.27); the non-monotonicity is unexplained by these
   arms and needs the traced attribution (`runs/drafter-degradation-20261124/REPORT.md` §3, the
   ring-trace item) to resolve.

5. **Even the best arm leaves the decoders 46% below the no-prefill control.** The knob cannot recover
   the loss. The interference is structural — bandwidth/queue contention between a cold prefill and
   decode rows — not a scheduling-granularity problem. Fixing it needs a structural change (eviction
   reservation / lookahead 0191, or not co-scheduling cold prefills with decodes), not a knob.

6. **Not cache eviction.** The decoders' `cached` stayed 6144 in every arm, so the loss is contention,
   not prompt-cache loss. The cell's own warming holds.

## Operator decision

Production's 250/5 is the prefill-optimal setting and is the right choice for the live workload, whose
binding constraint was TTFT (422 s mean, 113/153 clients gone at 600 s): slowing the prefill 26% to
gain 40% decoder throughput would worsen the queue. 1000 ms is only a candidate if the traffic mix
turns decode-dominated with TTFT headroom. **Adopt nothing.**

## Files

`runs/mixed-20261125/{control,sweep}/<arm>.{json,log,profile.txt,frames.txt,armed.txt,banner.txt,
cmp.txt,exact.json,exact.log,restart.log}`, `run.log`, `control/ctl.json`, `control/restored.validate.*`.
Replay: `bash runs/mixed-20261125/run.sh` from the recipe root with production idle.
