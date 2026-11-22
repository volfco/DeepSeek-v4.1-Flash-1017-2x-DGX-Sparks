# DRAFT_BLOCK_KEPT on production (recipe-0264, 27baa93), 2026-11-22 ~11:50–12:24 UTC

What it is: with `TF_DS_DRAFT_BLOCK=6` the drafter widens to six drafts only after a round that kept
`TF_DS_DRAFT_BLOCK_KEPT` rows (default 4). Lower threshold = widen sooner.

Order (ABBA): ctl1, k3a, k2a, ctl2, k2b, k3b. Each arm: exact panel (39 cases) vs ctl1, plus the decode probe
(`suite_probe.py --narrow`: tetris/code/json/prose/go/ts/sql/yaml greedy, tetris sampled, sampled x2/x4 bursts).

**Exact:** all six arms IDENTICAL 39/39.

**Mechanism is deterministic:** round counts repeat exactly within each pair (json greedy: ctl 372 rounds,
k3 367, k2 356). Lower threshold → fewer rounds, each slightly longer (json 39.25 → 40.0 ms). Controls drift ~0.

## k3

| Probe row | ctl1 | ctl2 | k3a | k3b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris greedy | 123.4 | 122.6 | 123.4 | 123.4 | 123.0 | 123.4 | **+0.30%** |
| tetris sampled | 126.4 | 126.4 | 127.2 | 128.0 | 126.4 | 127.6 | **+0.93%** |
| code greedy | 120.0 | 120.0 | 120.9 | 120.9 | 120.0 | 120.9 | **+0.78%** |
| json greedy | 105.2 | 105.2 | 105.9 | 105.9 | 105.2 | 105.9 | **+0.68%** |
| prose greedy | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | 61.0 | **+0.00%** |
| tetris sampled x2 | 167.5 | 167.7 | 168.6 | 168.4 | 167.6 | 168.5 | **+0.53%** |
| tetris sampled x4 | 219.1 | 219.6 | 219.2 | 219.8 | 219.3 | 219.5 | **+0.09%** |
| go greedy | 136.2 | 136.2 | 134.7 | 136.2 | 136.2 | 135.5 | **-0.53%** |
| ts greedy | 132.0 | 132.0 | 132.0 | 132.0 | 132.0 | 132.0 | **+0.00%** |
| sql greedy | 118.5 | 118.5 | 118.5 | 118.5 | 118.5 | 118.5 | **+0.00%** |
| yaml greedy | 132.0 | 132.0 | 132.0 | 132.0 | 132.0 | 132.0 | **+0.00%** |

mean tok/s change: +0.25% over 11 rows (positive = faster)

## k2

| Probe row | ctl1 | ctl2 | k2a | k2b | control | test | test vs control |
|---|---:|---:|---:|---:|---:|---:|---:|
| tetris greedy | 123.4 | 122.6 | 121.9 | 121.9 | 123.0 | 121.9 | **-0.89%** |
| tetris sampled | 126.4 | 126.4 | 128.0 | 128.0 | 126.4 | 128.0 | **+1.25%** |
| code greedy | 120.0 | 120.0 | 120.9 | 120.9 | 120.0 | 120.9 | **+0.78%** |
| json greedy | 105.2 | 105.2 | 107.4 | 108.2 | 105.2 | 107.8 | **+2.45%** |
| prose greedy | 61.0 | 61.0 | 60.5 | 60.5 | 61.0 | 60.5 | **-0.79%** |
| tetris sampled x2 | 167.5 | 167.7 | 167.3 | 167.6 | 167.6 | 167.4 | **-0.12%** |
| tetris sampled x4 | 219.1 | 219.6 | 219.6 | 220.3 | 219.3 | 219.9 | **+0.29%** |
| go greedy | 136.2 | 136.2 | 136.2 | 136.2 | 136.2 | 136.2 | **+0.00%** |
| ts greedy | 132.0 | 132.0 | 130.6 | 132.0 | 132.0 | 131.3 | **-0.51%** |
| sql greedy | 118.5 | 118.5 | 118.5 | 118.5 | 118.5 | 118.5 | **+0.00%** |
| yaml greedy | 132.0 | 132.0 | 133.3 | 133.3 | 132.0 | 133.3 | **+1.04%** |

mean tok/s change: +0.32% over 11 rows (positive = faster)

Earlier sweep1 (prose-heavy workload, round-time): k3 within noise (−0.16% mean rounds).

**Decision:** k2 adopted (exact; +0.32% mean decode tok/s, best on structured output: json +2.5%, yaml +1.0%;
prose/tetris-greedy −0.8/−0.9% are single-request rows). Per the adopt rule (exact + faster/neutral → production).
Deploy: `runs/deploy-kept2-20261122`. k2 becomes the control. k1 not tried (would widen almost always).
