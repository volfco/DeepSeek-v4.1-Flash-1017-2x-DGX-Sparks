# 0261 on sampled rounds, and how to score this directory's suites

Written by the audit-review session (2026-11-23). See `continue.md`, "Audit review and re-score of rounds 0259–0279".

- `work.py`: the round-time workload with `WORKLOAD_SAMPLED=1` (seeded T=1.0 cells). Unset it is the original.
- `topk-arms.txt`, `topk-run.sh`: shipped 0261 against the full-logit fallback on sampled rounds. No build. Not run yet.
- Score every suite here with `runs/overnight-20261122/table_gpu.py` (GPU clock for one-stream cells, total forward and
  wall for multi-stream cells, `pass_ms` from the profile lines). `table.py`'s host metric is quantized.

What the 0261 pair can show: the fallback moves R × 258,560 bytes a rank over the RDMA ring and draws from 129,280
columns on rank 0; shipped 0261 runs a radix select on each rank and an NCCL all-gather of R × kmax × 8 bytes. Fewer
bytes, a different transport. If the on arms are not faster on the GPU clock, 0261's selector is not worth optimizing
and the switch could go back off; if they are, the audit's early-exit selector has a measured base to improve on.
