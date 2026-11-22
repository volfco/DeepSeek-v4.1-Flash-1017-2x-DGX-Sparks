# Forward ms a round by rows on production (recipe-0264 + KEPT=2), 2026-11-22 13:49–14:21 UTC

Purpose: M1 in `docs/benchmark-gaps-and-moe-review.md` §4.4. The draft window allocator prices a round with the static
`draft.round_ms` table (`draft.zig:840`) unless `TF_DS_ROUND_MS` is set. That table was never refit for today's build.

Method: seven arms `TF_DS_FIX_K=0..6` (every drafted stream verifies K drafts a round), `rowcost.py`: 4K × 1/2/3/4 and
32K × 1/4 streams, 200 greedy tokens each, the frozen corpus; forward ms and rows a round from rank 0's differenced
profile (graphs on).

**Finding on FIX_K:** under the pin, concurrent streams do not share rounds. Every cell ran ~1 stream a round (rows a
round = K + 1 at any stream count, rounds ≈ streams × 200 / (K + 1), wall scales with streams). So this measured rows
1–7 only. Rows past 7 in the table are the line's extrapolation. If the ABBA shows a gain, measure 8–28 rows another
way: profile rows histogram, or concurrent probes without the pin.

Depth costs little: 32K is +0.6–1.2 ms a round over 4K at every K.

| arm | depth | streams | rounds | rows a round | forward ms | served table at rows |
|---|---:|---:|---:|---:|---:|---:|
| fk0 | 4K | 3 | 589 | 1.03 | 21.03 | 30.0 |
| fk0 | 4K | 4 | 787 | 1.03 | 21.04 | 30.0 |
| fk0 | 4K | 1 | 197 | 1.04 | 21.03 | 30.0 |
| fk0 | 4K | 2 | 390 | 1.04 | 21.09 | 30.0 |
| fk1 | 4K | 3 | 346 | 2.04 | 25.17 | 35.2 |
| fk1 | 4K | 4 | 461 | 2.04 | 25.18 | 35.2 |
| fk1 | 4K | 1 | 117 | 2.04 | 25.18 | 35.2 |
| fk1 | 4K | 2 | 226 | 2.06 | 25.33 | 35.2 |
| fk2 | 4K | 4 | 360 | 3.04 | 29.14 | 39.9 |
| fk2 | 4K | 3 | 268 | 3.04 | 29.25 | 39.9 |
| fk2 | 4K | 1 | 91 | 3.05 | 29.18 | 39.9 |
| fk2 | 4K | 2 | 173 | 3.06 | 29.47 | 39.9 |
| fk3 | 4K | 4 | 330 | 4.00 | 32.43 | 44.9 |
| fk3 | 4K | 1 | 85 | 4.00 | 32.63 | 44.9 |
| fk3 | 4K | 3 | 246 | 4.00 | 33.02 | 44.9 |
| fk3 | 4K | 2 | 155 | 4.01 | 32.74 | 44.9 |
| fk4 | 4K | 4 | 305 | 4.99 | 35.95 | 48.6 |
| fk4 | 4K | 3 | 228 | 4.99 | 35.95 | 48.6 |
| fk4 | 4K | 1 | 80 | 5.00 | 36.29 | 48.6 |
| fk4 | 4K | 2 | 143 | 5.00 | 36.43 | 48.6 |
| fk5 | 4K | 3 | 223 | 5.97 | 39.19 | 52.1 |
| fk5 | 4K | 4 | 298 | 5.98 | 39.06 | 52.1 |
| fk5 | 4K | 1 | 78 | 6.00 | 39.50 | 52.1 |
| fk5 | 4K | 2 | 139 | 6.01 | 39.72 | 52.1 |
| fk6 | 4K | 3 | 216 | 6.93 | 42.26 | 55.6 |
| fk6 | 4K | 4 | 291 | 6.94 | 42.15 | 55.6 |
| fk6 | 4K | 2 | 132 | 6.96 | 42.76 | 55.6 |
| fk6 | 4K | 1 | 77 | 6.98 | 42.60 | 55.6 |

line: 17.87 + 3.603 x rows

| rows | served | fit | fit / served |
|---:|---:|---:|---:|
| 1 | 30.0 | 20.9 | 0.70 |
| 2 | 35.2 | 25.0 | 0.71 |
| 3 | 39.9 | 29.1 | 0.73 |
| 4 | 44.9 | 32.7 | 0.73 |
| 5 | 48.6 | 36.2 | 0.74 |
| 6 | 52.1 | 39.4 | 0.76 |
| 7 | 55.6 | 42.6 | 0.77 |
| 8 | 59.1 | 46.2 | 0.78 |
| 9 | 62.7 | 49.8 | 0.79 |
| 10 | 66.3 | 53.4 | 0.81 |
| 11 | 69.6 | 57.0 | 0.82 |
| 12 | 72.9 | 60.6 | 0.83 |
| 13 | 75.5 | 64.2 | 0.85 |
| 14 | 78.2 | 67.8 | 0.87 |
| 15 | 81.3 | 71.4 | 0.88 |
| 16 | 84.4 | 75.0 | 0.89 |
| 17 | 87.5 | 78.6 | 0.90 |
| 18 | 90.6 | 82.2 | 0.91 |
| 19 | 93.7 | 85.8 | 0.92 |
| 20 | 96.8 | 89.4 | 0.92 |
| 21 | 99.9 | 93.0 | 0.93 |
| 22 | 103.0 | 96.6 | 0.94 |
| 23 | 106.1 | 100.3 | 0.95 |
| 24 | 109.2 | 103.9 | 0.95 |

TF_DS_ROUND_MS=20.9,25.0,29.1,32.7,36.2,39.4,42.6,46.2,49.8,53.4,57.0,60.6,64.2,67.8,71.4,75.0,78.6,82.2,85.8,89.4,93.0,96.6,100.3,103.9

**Reading.** Today's forward is 17.9 + 3.6 ms a row. The served table is 23–30% high at 1–7 rows, and its marginal row
costs 5.2/4.7/5.0 ms early against 4.1/4.0/3.5 measured. What moves the allocator is the ratio of a row's marginal cost
to the round's cost. Measured, that is 19.6% for the second row against 17.3% in the table, so the fit makes shallow extra rows
look slightly dearer and deep ones (past 4) about the same. The allocator also learns `overhead` (round time minus table forward,
floored at 0); with the served table reading high, that overhead is stuck at 0, and the true per-round fixed cost
(enqueue 0.5–2 ms + drafter pass ~3.7 ms) is invisible. ABBA: `runs/overnight-20261122/sched2`.
