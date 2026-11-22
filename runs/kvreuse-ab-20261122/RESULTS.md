# 0259 TF_DS_KV_FETCH_REUSE on the pair, November 22: bit-identical, decode rounds 1.7% shorter. Adopted.

**Result.** On one image (`bench-fused-next-69335b8` = production's `84806bb` + 0259, its switch off by default), ABBA with `TF_DS_KV_FETCH_REUSE` off/on/on/off:
- **Exact:** the 39-case panel is **IDENTICAL** in on1, on2 and off2 against off1 (token hashes, counts, finish reasons, 8 concurrent == solo).
- **Engagement:** the reuse banner printed once on each rank in both on arms and never in the off arms.
- **Speed:** graphs-on decode rounds are **0.5–2.5% shorter (mean −1.7%)** against the mean of both off arms. On beats both controls in 11 of 12 pairings.

Production moved to it at 07:07–07:10 UTC ([deploy record](../deploy-kvreuse-20261122/)).

**Realized vs projected.** The [sizing](../kvreuse-size-20261122/RESULTS.md) projected 3.0–3.5% from removing 30 of 38 exchanges. The A/B realized about half. The likely reason (not measured): part of `sh_rows` is one rank waiting for the other, and without the exchange that wait moves to the layer's next collective (`gather_a`/`gather_m`) instead of disappearing. The removed packing (`sh_fetch`) and the exchange's fixed latency are the realized part.

## Setup

- Head clock 06:42–07:07 UTC. Every arm used production's `.env` plus `IMAGE=bench-fused-next-69335b8 PREPARE=0 TF_DS_ROUND_PROFILE=1`; the on arms also set `TF_DS_KV_FETCH_REUSE=1`. Graphs were on.
- Per arm: a restart; `tools/exact.py record` (off1 fresh, the rest `--like off1`); the [kvreuse-size workload](../kvreuse-size-20261122/workload.py) (greedy 320-token summaries at 4K/32K/128K × 1/4 streams, per-round `forward_ms` from differenced `/metrics` profile lines); the banner count on both ranks. [run.sh](run.sh)
- Restore: `.env` alone (`84806bb`), `validate.py --compare`: all passed, 8/8 saved replies.
- The first attempt (`attempt1/`) used `exact.py`'s default port 8888 and failed before any case. It is kept, not counted.

## Decode round time (ms, graphs-on `forward_ms`)

| Depth × streams | off1 | on1 | on2 | off2 | off mean | on mean | on vs off | off2 vs off1 (control drift) | wall s off/on mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 30.67 | 29.68 | 29.61 | 30.04 | 30.36 | 29.64 | **-2.35%** (-0.71 ms) | -2.07% | 5.85 / 5.78 |
| 4K × 4 | 58.56 | 57.74 | 56.59 | 57.93 | 58.25 | 57.17 | **-1.85%** (-1.08 ms) | -1.08% | 14.91 / 14.73 |
| 32K × 1 | 33.32 | 33.13 | 32.97 | 33.11 | 33.22 | 33.05 | **-0.49%** (-0.16 ms) | -0.62% | 5.51 / 5.46 |
| 32K × 4 | 63.32 | 62.10 | 61.79 | 63.70 | 63.51 | 61.94 | **-2.47%** (-1.57 ms) | +0.60% | 14.59 / 14.47 |
| 128K × 1 | 33.37 | 33.01 | 32.94 | 33.34 | 33.35 | 32.97 | **-1.14%** (-0.38 ms) | -0.07% | 5.92 / 5.90 |
| 128K × 4 | 60.34 | 57.73 | 58.68 | 58.25 | 59.29 | 58.20 | **-1.84%** (-1.09 ms) | -3.47% | 15.97 / 15.73 |

Control drift (off2 vs off1) reaches 3.5% at 128K × 4. That cell's on2 (58.68) is slower than off2 (58.25), the one pairing out of 12 where on did not beat both controls. Every other cell has both on arms below both off arms.

## Caveats

- The profile (`TF_DS_ROUND_PROFILE=1`) synchronizes every round in all four arms alike. Unprofiled served throughput was not measured separately. Wall times move the same direction (−0.3 to −1.6%).
- One image, one window, two repeats per arm. Sampled (T=1.0) decode was not run; the change is in KV gathering, not sampling.
- The recipe (`zig/patches/`) does not yet carry 0259. Export it onto the rebased tree (`tfref-fnext-rebase`) before any recipe image build.
