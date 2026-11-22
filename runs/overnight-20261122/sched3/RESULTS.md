# DRAFT_BLOCK=7 on sampled traffic, production (recipe-0264 + KEPT=2), 2026-11-22 15:05–15:33 UTC

ABBA ctl1, b7a, ctl2, b7b. Probe `suite_probe.py --narrow --sampled`: every class (tetris/code/json/prose + go/ts/sql/yaml)
at the clients' sampling (T=1.0, top_p 0.95, top_k 20), seeds 700 and 701, then the x2/x4 sampled bursts. Controls
repeat within one row (±1 tok/s); test arms likewise. (Exactness for b7 was established in ../sched2: 39/39 and sampled-identical.)

| Probe row | ctl1 | ctl2 | b7a | b7b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris sampled s700 | 128.0 | 128.0 | 124.9 | 124.9 | 128.0 | 124.9 | **-2.44%** |
| code sampled s700 | 123.9 | 123.9 | 126.9 | 126.9 | 123.9 | 126.9 | **+2.48%** |
| json sampled s700 | 106.7 | 106.7 | 105.9 | 105.9 | 106.7 | 105.9 | **-0.69%** |
| prose sampled s700 | 61.0 | 60.5 | 60.5 | 61.0 | 60.7 | 60.7 | **+0.00%** |
| tetris sampled s701 | 121.9 | 121.9 | 121.2 | 121.2 | 121.9 | 121.2 | **-0.59%** |
| code sampled s701 | 121.9 | 121.9 | 126.9 | 126.9 | 121.9 | 126.9 | **+4.13%** |
| json sampled s701 | 105.9 | 105.9 | 108.2 | 107.4 | 105.9 | 107.8 | **+1.76%** |
| prose sampled s701 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | 58.6 | **+0.00%** |
| go sampled s700 | 128.0 | 128.0 | 130.6 | 130.6 | 128.0 | 130.6 | **+2.04%** |
| ts sampled s700 | 129.3 | 129.3 | 130.6 | 130.6 | 129.3 | 130.6 | **+1.02%** |
| sql sampled s700 | 111.3 | 111.3 | 108.5 | 107.6 | 111.3 | 108.0 | **-2.95%** |
| yaml sampled s700 | 128.0 | 126.7 | 126.7 | 125.5 | 127.4 | 126.1 | **-0.99%** |
| go sampled s701 | 134.7 | 134.7 | 134.7 | 133.3 | 134.7 | 134.0 | **-0.52%** |
| ts sampled s701 | 133.3 | 133.3 | 132.0 | 132.0 | 133.3 | 132.0 | **-1.03%** |
| sql sampled s701 | 114.3 | 114.3 | 112.3 | 112.3 | 114.3 | 112.3 | **-1.76%** |
| yaml sampled s701 | 129.3 | 128.0 | 129.3 | 129.3 | 128.6 | 129.3 | **+0.50%** |
| tetris sampled x2 | 167.9 | 167.7 | 164.8 | 164.5 | 167.8 | 164.7 | **-1.87%** |
| tetris sampled x4 | 219.3 | 219.8 | 217.3 | 217.6 | 219.6 | 217.5 | **-0.96%** |

mean tok/s change: -0.10% over 18 rows (positive = faster)

**Decision: reject `DRAFT_BLOCK=7`.** On sampled traffic it is −0.10% mean (code +2.5/+4.1%, go/ts mixed, sql −3.0/−1.8%,
tetris −2.4%) and **slower at concurrency** (x2 −1.9%, x4 −1.0%). The greedy-code win from sched2 (+7.2%) does not carry
over to T=1.0, where drafts survive less often and the seventh row is mostly wasted. Production stays at block 6.
