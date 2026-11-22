# Sharded-KV decode exchange on production, November 22: 3.6–4.2% of a round, about 3% recoverable

**Result.** On production's image (`bench-fused-next-84806bb`), fetch decode spends **1.18–1.29 ms at one stream and 2.48–2.62 ms at four streams per round** packing (`sh_fetch`) and exchanging (`sh_rows`) the picked KV rows. That is 3.6–4.2% of the round, and it is flat from 4K to 128K. `sh_rows` scales with rows, not depth: it is the exchange's wait, not bytes. The checkpoint's 38 compressed layers read only eight (KV source, selection) runs, so 30 of the 38 exchanges repeat one already done. **Skipping them projects to 0.93–2.06 ms a round, 3.0–3.5% of the graphs-on round.** That is candidate 1 of [continue.md's ranked next set](../../continue.md). The patch is `0259` `TF_DS_KV_FETCH_REUSE` (`~/Workspace/tfref-kvreuse`, branch `kv-fetch-reuse` `69335b8`). Its A/B is [runs/kvreuse-ab-20261122](../kvreuse-ab-20261122/).

## Setup

- Three restarts plus a restore on the pair, 2026-11-22 06:05–06:31 UTC (head clock). Every arm used production's `.env` and image.
  - `p2g0`: `TF_DS_ROUND_PROFILE=2 GRAPHS=0`, the eager phase timer (`phases_ms`).
  - `p1g0`: `TF_DS_ROUND_PROFILE=1 GRAPHS=0`, eager `forward_ms` without the timer.
  - `p1`: `TF_DS_ROUND_PROFILE=1`, graphs-on `forward_ms`, the served round.
  - `off`: production exactly, wall times only, then `validate.py --compare` (all passed, 8/8 saved replies).
- Workload ([workload.py](workload.py)): greedy 320-token summaries of repo text at 4,096 / 32,768 / 131,072 prompt tokens, one and four streams. Histories were prefilled first. Per-round figures difference the cumulative `/metrics` lines around each cell. Every summary decoded the whole budget.
- The profile synchronizes the GPU each round. The `p1` vs `off` wall times (within 0.1–2%) bound that perturbation.

## Per-round cost

| Depth × streams | sh_fetch | sh_rows | fetch+exchange | eager round (timer) | share | eager round (no timer) | graphs-on round | projected saving (30/38) | wall s p1 / off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K × 1 | 0.338 | 0.845 | 1.18 ms | 31.5 ms | 3.8% | 30.7 ms | 30.0 ms | 0.93 ms (3.1% of graphs-on) | 5.8 / 5.79 |
| 4K × 4 | 0.449 | 2.030 | 2.48 ms | 59.3 ms | 4.2% | 59.4 ms | 58.4 ms | 1.96 ms (3.3% of graphs-on) | 15.03 / 14.88 |
| 32K × 1 | 0.365 | 0.898 | 1.26 ms | 34.8 ms | 3.6% | 33.8 ms | 33.1 ms | 1.00 ms (3.0% of graphs-on) | 5.5 / 5.48 |
| 32K × 4 | 0.462 | 2.137 | 2.60 ms | 62.4 ms | 4.2% | 65.8 ms | 63.5 ms | 2.05 ms (3.2% of graphs-on) | 14.64 / 14.62 |
| 128K × 1 | 0.384 | 0.905 | 1.29 ms | 35.0 ms | 3.7% | 34.2 ms | 34.3 ms | 1.02 ms (3.0% of graphs-on) | 6.05 / 5.92 |
| 128K × 4 | 0.492 | 2.123 | 2.62 ms | 61.6 ms | 4.2% | 60.0 ms | 59.4 ms | 2.06 ms (3.5% of graphs-on) | 16.0 / 15.86 |

The same eager runs put `moe` at 15.5–17.8 ms (1 stream) and 38.2–40.0 ms (4 streams), `attn` at 1.8–2.4 ms and `a_idx` at 0.8–2.8 ms. `a_idx` grows with depth. Raw data: `p2g0.json`, `p1g0.json`, `p1.json`, `off.json`.

## Findings

- **Graphs hide only 1–2.5% of a decode round** (eager vs graphs-on `forward_ms`). Eager stage shares therefore carry over to served rounds almost unchanged on this build.
- **The served clock-mark profiler is broken.** With `TF_DS_CLOCK_MARKS` (any value) and graphs on, `round_clock_us.rounds` printed `12297829382473034913` (`0xAAAA…`, Zig's undefined fill). Every phase showed `[0.0, n]` and `round_launches` 0. The served `Model`'s `clk_rounds`/`clk_ph`/`clk_cnt` field defaults never apply (the same "starts undefined" pattern `Round.init` documents). The device clock buffer is not read on the graphs path either. The first attempt is kept in `m4-broken/`. Use `TF_DS_ROUND_PROFILE=2 GRAPHS=0` for stage attribution.
- The eager phase timer adds about 0–3% to a round (`p2g0` vs `p1g0`).
