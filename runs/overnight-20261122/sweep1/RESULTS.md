# Single-switch screen on production, November 22: two rejects, nothing else clears noise

Production (`bench-fused-next-27baa93` = recipe-0264, `.env` with 0259/0260/0261/0264 on), 10:45–11:47 UTC. One arm per switch, bracketed by two controls. Each arm ran the exact panel `--like` the reference, the frozen-corpus round-time workload (4K/32K/128K × 1/4 streams) and salted cold prefill at 32K and 96K. Production was restored and validated. **Every arm is exact 39/39.**

Δ vs the mean of ctl1/ctl2 (rounds: graphs-on `forward_ms` a round, + = slower; prefill: tok/s, + = faster):

| Arm | 4K×1 | 4K×4 | 32K×1 | 32K×4 | 128K×1 | 128K×4 | rounds mean | wall mean | prefill 32K | prefill 96K | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctl1 | −0.04 | −0.12 | −0.19 | +1.02 | +0.31 | +1.02 | +0.33 | −0.16 | +0.24 | +0.20 | control |
| `TF_DS_PDL=1` | −0.13 | −0.51 | −0.47 | +0.82 | +0.03 | −0.06 | −0.05 | −0.21 | +0.00 | +0.24 | noise |
| `TF_DS_DEVICE_ACCEPT=1` | +0.03 | −0.14 | −0.10 | +0.74 | +0.04 | +0.54 | +0.18 | −0.06 | +0.25 | −0.08 | noise |
| `TF_DS_COMPRESS_SIDE=1` | +0.12 | −0.45 | +0.04 | −0.66 | +0.17 | +1.61 | +0.14 | −0.05 | −0.54 | +1.84 | noise |
| `TF_DS_ENGRAM_SCALES=map` | +0.88 | +1.35 | +7.15 | +6.06 | +8.54 | +5.46 | **+4.91** | +1.43 | **−3.83** | +1.41 | **reject** |
| `TF_DS_ENGRAM_STRICT=1` | −0.15 | −0.77 | +0.28 | −0.36 | +0.16 | −1.56 | −0.40 | +0.11 | −0.22 | −0.30 | noise |
| `TF_DS_PREFILL_GATHER_OVERLAP=1` | +0.01 | −0.10 | +0.41 | +1.54 | +0.41 | −0.14 | +0.35 | +0.66 | +0.01 | +0.57 | noise |
| `TF_DS_GREEDY_ARGMAX=1` | −0.04 | +0.51 | +0.56 | +1.44 | +3.05 | +1.43 | **+1.16** | +0.14 | +0.18 | +0.09 | **reject** (exact, slower) |
| `TF_DS_DRAFT_BLOCK_KEPT=3` | +0.28 | −0.70 | −0.00 | −0.42 | −0.14 | +0.00 | −0.16 | +0.10 | −0.08 | +1.93 | prose-neutral; retested on code/JSON in [kept](../kept/) |
| ctl2 | +0.04 | +0.12 | +0.19 | −1.02 | −0.31 | −1.02 | −0.33 | +0.16 | −0.24 | −0.20 | control |

- **Control-to-control spread is ±0.33% mean and ±1% per cell**, so single-arm effects under about 0.5% are not resolved by this screen.
- **`TF_DS_ENGRAM_SCALES=map` slows decode rounds by 5–8.5% from 32K up.** It was part of the November 17 eight-switch "best" bundle. On today's build it is a clear loss; keep it off.
- **`TF_DS_GREEDY_ARGMAX=1` (0186, the repaired greedy exchange) is exact on today's panel**, including the 8 concurrent == solo cases the original failed, **but 1.2% slower**, up to +3% at 128K × 1. Keep it off.
- PDL, device accept, compressor side stream, Engram strict and prefill gather overlap are all within the screen's noise in decode and prefill. Nothing to adopt; they could be retested only in ABBA if wanted.
- Cold prefill on production: 32K ≈ 2,106 tok/s, 96K ≈ 2,015 tok/s.
