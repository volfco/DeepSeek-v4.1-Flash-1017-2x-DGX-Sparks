# Supervisor response to the pi review round (2026-11-21)

pi's review ([REVIEW.md](REVIEW.md), six read-only reviewer subagents, coordinator-verified) found one blocker in code,
one open evidence blocker, and a set of should-fix items. Disposition:

## B1, 0241's pair query rounds the odd rotary element in the torch contraction: confirmed and fixed (0252)

- Confirmed against the base: the one-element query kernel (`3c54a9d` `dsv41_torch.cu:794`) computes the odd element
  as `fma(other, sn, v*cs)` (the kit's `_rope_heads` form); 0241's pair chunk computed `fma(v1, cs, v0*sn)`.
- The harness never saw it (0 differing bytes on random inputs). New harness cases `rope query R={32,48,64,1024} D=128 adv`
  (`tools/fused_bench/bench.py`, `adv_query_hits`) place host-searched inputs where the two contractions round to
  different bf16 values at an E2M1-boundary mantissa, with a per-head group maximum that sweeps the scale. The case
  also reads back the kit's rotated value: on every row it equals the kit form (`{'kit': R, 'torch': 0}`), so the
  kit's contraction is measured, not assumed.
- Red before the fix: WP4 tip vs the kit, R=64 36 bytes, R=1024 576 bytes ([b1-red](b1-red/),
  [wp4tip-self-parity.log](wp4tip-self-parity.log)). Green after: [b1-green](b1-green/).
- Fix: engine commit `5110596` (0252), `r1 = __fmaf_rn(v0, sn, __fmul_rn(v1, cs))`. The diff `3c54a9d..fused-next`
  adds no other `__fmaf_rn`; the pair's `r0` already matches the kit's `re = fma(xe, cs, -(xo*sn))`.
- Production was never exposed: the deployed image (prune-six) predates 0241.

## 0216's prompt bypass (review S1-S3, coordinator S2): gated (0253)

Engine commit `d5ecba0` (0253): the bypass now requires `TF_DS_IDX_FUSED` (production sets it, so the benchmark still
exercises it), skips the keyed layers (`li < candidate_source` with a power-of-two key count, where the replaced
producer is the kit's `_topk_sel`, whose k == n order is not byte-checked against `iotaVis`), and requires
`nbk <= candidate_blocks` so the "no pool" argument is a checked condition. The non-keyed layers, whose producer is the
engine's `topk_indices` that the argument cites, keep it.

## B2, the unexplained new-only differences in WP2's run: still open, a lead

pi's failing run (`wp2-full-parity.log`) was two `bench.py` processes on the same `wp2/bin`, redirecting into the **same**
log and JSON a minute apart, and a third `--no-guard` process (no GPU lock) started two minutes later while production
served. The harness loads cubins lazily (`module()` on first use), but no rebuild overlapped the run (the last
`build.sh` finished at 11:01:25, the run started 11:01:37), so a stale cubin is ruled out for that run. No mechanism is
established. Since then: 40/40 isolated repeats, sanitizers clean, six concurrent full runs, and two full parity runs
in this review (WP4 tip self, WP4 tip vs fix, 162 cases each) without it. It stays open and blocks any promotion
recommendation until the pair's exact gate and reply hashes settle it.

## Re-verification of the fixed tip (`d5ecba0`)

- Full harness parity, OLD = WP4 tip `59e71a8`, NEW = `d5ecba0`: 162 cases; every new arm 0 bytes against the
  reference; the only failures are the old arm on the two adversarial query cases (the bug) ([fix-parity.log](fix-parity.log)).
- Coverage added (review S11): prompt compressor/key/latent at R=2048 and 4096: all pass.
- Host tests, fresh cache: 797/799, the two pre-existing failures only (0178 prompt end-state, hot-cache hotPick).

## Not changed in this round (recorded)

- S4 trace statics unsynchronised, S5 the trace test models the dispatch rather than calling it, S12 `warp_end` for
  non-multiple-of-32 blocks: off by default / unreachable at the served block size; left for the next round.
- S6 the PDL pre-wait read relies on load-time weights: true today (`weights.zig` plain uploads); documented here.
- S7/S8 claim wording: 0217's 16384-block timings exist only as prose; 0213 is a 0.4-0.5% regression at the DRAM-bound
  R=1024/2048 prompt rows, accepted (not "unresolvable"). wp4's `timing_clean` holds for the parity/i2 JSONs only.
- S9 0215's 6144 threshold sits between measured points (4096 loses, 8192 wins); not re-measured.
