# Fused-next 0213-0253 on production's config, November 21: correct, speed-neutral

**Result.** The fused-next image (`bench-fused-next-d5ecba0`: production's source + this round's 0213-0253, including
the review fixes 0252/0253) produced the **same bytes as production everywhere the suite can compare**, and its speed
is inside the suite's resolution in both block pairs. PDL (`TF_DS_PDL=1`, s05) is also exact and also neutral.
Production was not changed; it was restored at 16:39 UTC (env match on both ranks, chat ok).

## Setup

Five blocks, 14:52-16:40 UTC, each a fresh restart with a private spill directory and the full suite cells (same
cells as [suite-fused-20261121](../suite-fused-20261121/)). Launched by [launch.sh](launch.sh) after 30 idle minutes (a
real client had been active until 14:21); the client guard counted only requests after the start and never fired.

| Block | Arm | Image | Switches beyond production's `.env` |
| --- | --- | --- | --- |
| s01, s04 | base | `bench-prune-six-20261120` (production) | none (production's `.env`: the four fused switches on) |
| s02, s03 | next | `bench-fused-next-d5ecba0` | none |
| s05 | next+PDL | `bench-fused-next-d5ecba0` | `TF_DS_PDL=1` |

All five blocks: status ok, 14/14 cells, validation passed, arm proof clean, clocks 2392/2385 MHz, temperature max
84-87 C, lowest free memory 4.9 GiB on the head.

## Correctness

- **Exact panel: s02, s03, s04 and s05 are each identical to s01 on all 39 cases** (text, token hash, token count,
  finish reason). This is the gate that caught 0206 last round; the review's B1 (0241) is fixed in this image.
- Decode/thinking replies equal base in every pairing (dec1 18/18, dec4 48/48, dec8 144/144, t1 12/12, t4 24/24); probe
  replies equal s01 in every block; sampled identity repeats within each block and matches across blocks.
- Long-context cells use a different passage per block (salted), so their replies and rates are not comparable
  (base-to-base spread +9.9%/+15.3% on lc1/lc4).

## Speed, per pair (A = s01 vs s02, B = s04 vs s03)

| Metric | s01 base | s02 next | A | s04 base | s03 next | B | base-base | s05 next+PDL | s05 vs base mean | prev fused s02/s03 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| probe solo tok/s | 114.18 | 114.17 | -0.01% | 114.18 | 114.41 | +0.20% | +0.01% | 114.14 | -0.04% | 114.02 / 114.21 |
| dec1 tok/s | 87.79 | 88.24 | +0.50% | 88.11 | 88.19 | +0.09% | +0.36% | 88.13 | +0.21% | 83.24 / 82.99 |
| dec4 tok/s | 116.19 | 116.25 | +0.05% | 115.92 | 116.66 | +0.64% | -0.23% | 116.53 | +0.40% | 119.30 / 119.26 |
| dec8 tok/s | 138.97 | 138.14 | -0.59% | 138.73 | 137.77 | -0.69% | -0.17% | 138.59 | -0.19% | 136.53 / 136.35 |
| dec1 ms/round | 33.62 | 33.45 | -0.52% | 33.45 | 33.45 | +0.00% | -0.52% | 33.62 | +0.26% | 33.69 / 33.69 |
| dec4 ms/round | 60.93 | 60.81 | -0.19% | 61.13 | 60.66 | -0.78% | +0.34% | 60.66 | -0.61% | 63.34 / 63.46 |
| dec8 ms/round | 86.62 | 86.66 | +0.06% | 86.92 | 86.54 | -0.43% | +0.35% | 86.48 | -0.33% | 86.33 / 86.40 |
| think t1 tok/s | 98.80 | 98.82 | +0.02% | 98.92 | 98.87 | -0.05% | +0.12% | 99.04 | +0.18% | 98.85 / 98.74 |
| think t4 tok/s | 139.09 | 140.54 | +1.04% | 139.79 | 140.64 | +0.61% | +0.50% | 139.53 | +0.06% | 139.54 / 138.94 |
| longctx lc1 tok/s | 74.68 | 72.30 | -3.19% | 82.04 | 75.10 | -8.45% | +9.85% | 80.06 | +2.18% | 76.54 / 69.12 |
| longctx lc4 tok/s | 117.06 | 114.23 | -2.42% | 134.91 | 112.98 | -16.25% | +15.25% | 120.79 | -4.12% | 118.55 / 121.62 |
| prefill 8K | 1990.33 | 1975.82 | -0.73% | 1978.19 | 1997.54 | +0.98% | -0.61% | 1982.56 | -0.09% | 1988.21 / 1996.32 |
| prefill 32K | 2091.39 | 2081.68 | -0.46% | 2092.68 | 2085.78 | -0.33% | +0.06% | 2090.90 | -0.05% | 2091.01 / 2092.14 |
| prefill 100K | 2043.77 | 2039.54 | -0.21% | 2040.46 | 2041.92 | +0.07% | -0.16% | 2047.77 | +0.28% | 2043.25 / 2044.00 |
| prefill 300K | 1798.19 | 1789.37 | -0.49% | 1802.77 | 1800.76 | -0.11% | +0.25% | 1806.21 | +0.32% | 1797.29 / 1803.85 |

- Every non-long-context difference between base and next is within about +-1% and inside or near the base-to-base
  spread of the same cell. The consistent signs are small: dec4 ms/round -0.19%/-0.78%, think t4 +1.04%/+0.61% (base
  spread +0.50%); dec8 aggregate -0.59%/-0.69% against a -0.17% spread, but dec8 ms/round is +0.06%/-0.43%, so that is
  a draft-acceptance/wall effect, not kernel time. Prefill is within +-1%.
- This is the expected size: the round's kernel gains are a few microseconds per launch (wp1-wp4) against 33-87 ms
  rounds, i.e. well under the ~0.5% the suite resolves. The suite is the correctness gate and a regression check here.
- PDL (s05) is within -0.6%/+0.4% of the base mean everywhere outside long context: no measurable gain from 0250 at
  this resolution.

## Against the last known benchmarks

- The previous suite's fused blocks ran exactly today's production config (last column). ms/round agrees: dec1
  33.69 then vs 33.45-33.62 now, dec8 86.33-86.40 vs 86.48-86.92, think and prefill within +-0.5% (8K 1988-1996 vs
  1976-1998; 100K 2043-2044 vs 2040-2048; 300K 1797-1804 vs 1789-1806).
- dec1/dec4 aggregate tok/s moved (+5.7% / -2.8%) between suites with tokens a round changing in step: each suite
  builds its own corpus (`corpus/`), so those prompts differ; compare ms/round across suites, tok/s within one.

## Caveats

- Engagement is inferred, not traced: the image and switches are proven on both ranks, but `TF_DS_FUSED_TRACE` was off,
  so no log shows each new arm (0215's small-row form, 0216's bypass, 0241's pair geometry) ran on served shapes.
- Review item B2 (WP2's unreproduced new-only byte difference) has no mechanism; four exact blocks and every reply
  hash here are clean, which is the evidence the review asked for before promotion.

Tabulation: [table.py](table.py) (per block), [compare_next.py](compare_next.py) (pooled, reply equality).
