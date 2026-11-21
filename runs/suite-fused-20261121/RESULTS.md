# Fused kernels 0193-0199 on production's config, November 21: deployed

**Result.** Production's best configuration on the `bench-prune-six-20261120` image with `TF_DS_IDX_FUSED`,
`TF_DS_FUSED_COMPRESS2_NORM`, `TF_DS_FUSED_ROPE_STORE` and `TF_DS_FUSED_ROUTE` set produced the same bytes as production
everywhere the suite can compare, and was slightly faster on decode in both block pairs. It became production at 07:36 UTC
([deploy record](../deploy-fused-20261121/)).

## Setup

Four blocks, ABBA, 05:52-07:17 UTC, each a fresh restart with a private spill directory and the full suite cells
(validation, warm-up, natural-text prefill 2K-100K, code and p125 prefill, decode at 1/4/8 streams, 100K long context,
thinking, the narrow probe, sampled identity, the 39-case exact panel, and a cold 300K prefill with its long-context
turns). The runner is `runs/suite-20261119/run_suite.py` with one change: the image check reads both ranks' running
image instead of grepping the restart log for `exl3-tp2-*` tags ([run_suite.py](run_suite.py)).

| Block | Arm | Image | Switches beyond production's `.env` |
| --- | --- | --- | --- |
| s01, s04 | base | `exl3-tp2-ea522d353626` (production) | none |
| s02, s03 | fused | `bench-prune-six-20261120` (production source + reworked 0193/0194, 0195/0197/0198, 0199) | the four fused switches |

Production's `.env` in every block: six-draft block, keyed drafts, token cache 32 MiB, Engram lead reader, T13 prune,
KV shard, spill, graphs cap 128. All four blocks: status ok, 39/39 cells clean, validation 8/8, lead banner on both ranks,
clocks 2392/2385 MHz, lowest free memory 4.9 GiB on the head.

## Correctness

- Exact panel: s02, s03 and s04 are each **IDENTICAL** to s01 on all 39 cases, including `len_20000`-`len_64000` (which
  0206's key fold had changed; this image has no 0206).
- Probe replies 11/11 and decode/thinking replies 41/41 equal s01's in every later block.
- Prefill and long-context cells use a different corpus passage per block (salted), so their replies are not comparable.

## Speed, per pair (A = s01 vs s02, B = s04 vs s03)

| Metric | s01 base | s02 fused | A | s04 base | s03 fused | B | base-base | fused-fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| probe solo tok/s | 113.15 | 114.02 | +0.76% | 113.18 | 114.21 | +0.91% | +0.03% | +0.17% |
| dec1 tok/s | 82.26 | 83.24 | +1.19% | 82.56 | 82.99 | +0.52% | +0.36% | -0.30% |
| dec4 tok/s | 118.39 | 119.30 | +0.77% | 119.04 | 119.26 | +0.18% | +0.55% | -0.03% |
| dec8 tok/s | 135.98 | 136.53 | +0.40% | 137.33 | 136.35 | -0.71% | +0.99% | -0.13% |
| dec1 ms/round | 34.05 | 33.69 | -1.06% | 34.05 | 33.69 | -1.06% | 0.00% | 0.00% |
| dec4 ms/round | 63.87 | 63.34 | -0.82% | 63.46 | 63.46 | 0.00% | -0.64% | +0.18% |
| dec8 ms/round | 86.70 | 86.33 | -0.42% | 86.51 | 86.40 | -0.13% | -0.22% | +0.07% |
| think t1 tok/s | 98.14 | 98.85 | +0.72% | 98.07 | 98.74 | +0.69% | -0.08% | -0.11% |
| think t4 tok/s | 138.87 | 139.54 | +0.48% | 139.43 | 138.94 | -0.35% | +0.40% | -0.43% |
| prefill 8K | 1982 | 1988 | +0.30% | 1986 | 1996 | +0.53% | +0.18% | +0.41% |
| prefill 32K | 2085 | 2091 | +0.28% | 2100 | 2092 | -0.35% | +0.69% | +0.05% |
| prefill 100K | 2046 | 2043 | -0.14% | 2056 | 2044 | -0.57% | +0.47% | +0.04% |
| prefill 300K | 1796 | 1797 | +0.06% | 1813 | 1804 | -0.50% | +0.93% | +0.36% |

- The probe and single-stream decode gain in both pairs, outside the base-to-base spread on the probe (0.03%).
- Decode rounds are shorter or equal in every decode cell in both pairs, with identical tokens a round: the kernels'
  time, not drafting.
- 8-stream aggregate tok/s is +0.40% / -0.71%, inside the 0.99% base-to-base spread on that cell. Prefill is within
  +-0.6% in both pairs.
- The gains are the size the kernel timings predicted (a few µs a layer against a ~34-87 ms round). The suite resolves
  about 0.5%; it is a correctness gate and regression check here, not a measure of the individual kernels.

## Against the last known benchmarks

- Probe, Nov 19 production after its restore: today's production is 0.0-1.0% slower per solo row and +0.4% on the
  four-stream row, every reply hash equal. The fused build's gain is about the size of that day-to-day drift.
- Nov 19 lead block (production's switches): prefill 8K/32K/100K 1,986/2,093/2,049 tok/s against today's base
  1,982-1,986 / 2,085-2,100 / 2,046-2,056; dec1/dec4 82.2/118.1 against 82.3-82.6/118.4-119.0. Same machine
  performance.

## Notes

- The runner crashed after its restore restart (its post-restore check reads `production.rank0.env`, which the Nov 19
  suite had saved by hand). Production had restarted; the env check and validation were then run by hand: 8/8 saved
  replies equal ([restored.validate.log](restored.validate.log)).
- No banner shows that each fused kernel ran; the switches are proven set on both ranks (arm proof), not engaged.
- Tabulation: [compare.py](compare.py) (pooled) and the per-pair table above.
