# Scheduler pricing and block width on production (recipe-0264 + KEPT=2), 2026-11-22 14:22–15:04 UTC

ABBA on one image: ctl1, rm1, b7a, ctl2, b7b, rm2. Each arm: exact panel vs ctl1, sampled identity (sampled.py), the
narrow decode probe. **All arms exact 39/39 and sampled-IDENTICAL** (both switches only choose how many rows a round
verifies).

- `rm` = `TF_DS_ROUND_MS` fitted from `../rowcost` (17.9 + 3.6 ms a row; rows 1–7 measured, past 7 extrapolated).
- `b7` = `TF_DS_DRAFT_BLOCK=7` (seven-draft block; the checkpoint's is 5, production runs 6).

## Measured round costs (rm)

| Probe row | ctl1 | ctl2 | rm1 | rm2 | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris greedy | 121.2 | 121.9 | 121.2 | 120.5 | 121.5 | 120.8 | **-0.59%** |
| tetris sampled | 127.2 | 128.0 | 128.0 | 128.0 | 127.6 | 128.0 | **+0.31%** |
| code greedy | 120.0 | 120.9 | 120.9 | 120.9 | 120.5 | 120.9 | **+0.39%** |
| json greedy | 107.4 | 108.2 | 108.2 | 107.4 | 107.8 | 107.8 | **+0.00%** |
| prose greedy | 60.0 | 60.5 | 60.5 | 60.0 | 60.2 | 60.2 | **+0.00%** |
| tetris sampled x2 | 168.4 | 169.2 | 167.2 | 166.8 | 168.8 | 167.0 | **-1.04%** |
| tetris sampled x4 | 219.1 | 219.9 | 214.6 | 212.8 | 219.5 | 213.7 | **-2.65%** |
| go greedy | 134.7 | 136.2 | 136.2 | 134.7 | 135.5 | 135.5 | **+0.00%** |
| ts greedy | 130.6 | 132.0 | 132.0 | 132.0 | 131.3 | 132.0 | **+0.51%** |
| sql greedy | 117.4 | 118.5 | 118.5 | 118.5 | 118.0 | 118.5 | **+0.46%** |
| yaml greedy | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | 133.3 | **+0.00%** |

mean tok/s change: -0.24% over 11 rows (positive = faster)

## Block 7 (b7)

| Probe row | ctl1 | ctl2 | b7a | b7b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris greedy | 121.2 | 121.9 | 122.6 | 122.6 | 121.5 | 122.6 | **+0.90%** |
| tetris sampled | 127.2 | 128.0 | 124.9 | 124.9 | 127.6 | 124.9 | **-2.13%** |
| code greedy | 120.0 | 120.9 | 129.1 | 129.1 | 120.5 | 129.1 | **+7.15%** |
| json greedy | 107.4 | 108.2 | 107.4 | 107.4 | 107.8 | 107.4 | **-0.35%** |
| prose greedy | 60.0 | 60.5 | 60.0 | 59.5 | 60.2 | 59.8 | **-0.78%** |
| tetris sampled x2 | 168.4 | 169.2 | 165.1 | 165.4 | 168.8 | 165.3 | **-2.08%** |
| tetris sampled x4 | 219.1 | 219.9 | 217.4 | 221.1 | 219.5 | 219.2 | **-0.14%** |
| go greedy | 134.7 | 136.2 | 137.6 | 137.6 | 135.5 | 137.6 | **+1.61%** |
| ts greedy | 130.6 | 132.0 | 134.7 | 134.7 | 131.3 | 134.7 | **+2.63%** |
| sql greedy | 117.4 | 118.5 | 118.5 | 118.5 | 118.0 | 118.5 | **+0.46%** |
| yaml greedy | 133.3 | 133.3 | 133.3 | 132.0 | 133.3 | 132.6 | **-0.51%** |

mean tok/s change: +0.61% over 11 rows (positive = faster)

**Decisions.**
- **rm: reject as fitted.** Neutral solo, −1.0% at x2 and −2.65% at x4, the cells whose rounds exceed 7 rows, where the
  table is the line's extrapolation (3.6 ms a row). A wide round priced too dear makes the allocator draft less at four
  streams. If revisited: measure wide rows directly (unpinned concurrent probes, profile rows histogram), or keep the
  served table past 7 rows.
- **b7: held, not adopted.** Mean +0.61%, from greedy code (+7.2%), ts (+2.6%) and go (+1.6%). Sampled Tetris is −2.1%
  solo and −2.1% at x2, and clients run at T=1.0. Follow-up: every class sampled, two seeds (`../sched3`,
  `suite_probe.py --sampled`).
