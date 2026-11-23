# NCCL knob screen (03:26–04:17 UTC 2026-11-23): no winner; production's NCCL_MAX_CTAS=4 stays

One knob per arm on top of production (image 10c9866, `.env` incl. `NCCL_MAX_CTAS=4`); 3 controls spread through the
run. Workload `short.py`: cold 32K ×2 and 96K ×1 prefill (salted), then greedy 256-token decode at x1/x4 on 4K prompts.
No exact panel: NCCL here only all-gathers and send/recvs (`comm.zig`), so no knob can move bits. Env proof per arm in
`<arm>.armed.txt` (now records `NCCL_*`).

| Arm | Knob | 32K tok/s (mean of 2) | 96K tok/s | vs ctl 32K | vs ctl 96K | x1 cell tok/s | x4 cell tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| ctlA | production | 2130.9 | 2082.6 | -0.01% | +0.06% | 58.13 | 54.06 |
| ctlB | production | 2129.9 | 2080.1 | -0.06% | -0.06% | 58.23 | 54.04 |
| ctlC | production | 2132.5 | 2081.3 | +0.06% | -0.00% | 58.09 | 54.11 |
| cta2 | MAX_CTAS=2 | 2083.2 | 2034.0 | -2.25% | -2.27% | 57.26 | 54.03 |
| cta8 | MAX_CTAS=8 | 2134.2 | 2082.8 | +0.15% | +0.07% | 58.02 | 53.98 |
| min4 | MIN_CTAS=4 | 2126.6 | 2077.8 | -0.21% | -0.17% | 58.17 | 54.06 |
| simple | PROTO=Simple | 2131.8 | 2082.3 | +0.03% | +0.05% | 58.15 | 54.12 |
| ll | PROTO=LL | 1729.5 | 1696.4 | -18.84% | -18.49% | 55.32 | 51.44 |
| nch1 | NCHANNELS_PER_NET_PEER=1 | 2130.9 | 2080.8 | -0.01% | -0.03% | 58.10 | 54.13 |
| nch4 | NCHANNELS_PER_NET_PEER=4 | 2133.4 | 2082.7 | +0.11% | +0.07% | 58.17 | 54.12 |
| buf8 | BUFFSIZE=8M | 2122.9 | 2072.4 | -0.39% | -0.43% | 57.04 | 54.02 |
| chunk512k | P2P_NET_CHUNKSIZE=512K | 2134.6 | 2082.1 | +0.16% | +0.04% | 58.12 | 54.05 |
| qps2 | IB_QPS_PER_CONNECTION=2 | 2107.2 | 2056.7 | -1.12% | -1.18% | 57.95 | 53.98 |

- Control spread: 32K 2129.9–2132.5, 96K 2080.1–2082.6 tok/s (≈±0.1%).
- **Worse:** `PROTO=LL` −19% prefill and −5% decode cells; `MAX_CTAS=2` −2.3%; `IB_QPS_PER_CONNECTION=2` −1.2%;
  `BUFFSIZE=8M` −0.4%.
- **Neutral (within ±0.2%):** `MAX_CTAS=8`, `MIN_CTAS=4`, `PROTO=Simple` (so Simple is already what NCCL picks),
  `NCHANNELS_PER_NET_PEER` 1/4, `P2P_NET_CHUNKSIZE=512K`.
- So `MAX_CTAS` 4 and 8 tie, and both beat the default (the nccl suite's +0.8%) and 2. Nothing left to adopt from NCCL.
- The decode cells include each 4K prompt's grid-remainder recompute (the x4 cell queues four ~2K-row passes), so they are
  a sanity check, not a decode benchmark; they move only with LL.
