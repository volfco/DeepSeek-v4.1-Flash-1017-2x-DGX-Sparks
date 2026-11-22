# Scheduling round A results (2026-11-22 18:38–18:42 UTC)

## A1: standalone ring bench at the hidden payload

`tf-dsv41-rdma-bench` from the production image (`27baa93`) with the ring fatbin compiled in-image. Width 5120, production
HCAs, GID 3, PDL on, `TF_RDMA_IDLE_S=2 TF_RDMA_NAP_US=500`, cpuset 15-19, proxy thread alone pinned to CPU 15 (the
`*.pin` files show each rank's proxy TID moved 15-19 → 15). Engine stopped. 1000 gathers per size (50 per graph,
replayed), 3 repeats. `bad`=0 and `bad_into`=0 on both ranks everywhere. Restore: validate rc=0.

| rows | bytes a rank | us_gather (3 repeats) | mean |
|---:|---:|---|---:|
| 1 | 20,480 | 9.84 / 9.92 / 9.91 | 9.89 |
| 7 | 143,360 | 17.35 / 17.31 / 17.22 | 17.29 |
| 14 | 286,720 | 27.11 / 27.27 / 27.25 | 27.21 |
| 21 | 430,080 | 34.25 / 33.87 / 33.53 | 33.88 |
| 28 | 573,440 | 41.19 / 40.97 / 40.77 | 40.98 |

- Linear fit: **9.4 µs + 1.16 µs a row** (≈17.7 GB/s of one rank's payload). Repeats agree within ±0.4 µs. Ranks
  agree to 0.03 µs, as they should: the gathers are paired. `us_into` (strided) is within ±0.7 µs of `us_gather`.
- Width 2560 (the bench default, older numbers): 1 row 8.9–9.4 µs, 6 rows 12.8 µs, 16 rows 19.1 µs.
- **Protocol price per decode round** at the ringprobe's 7.22 rows: 17.8 µs a gather × 80 hidden gathers ≈ **1.4 ms**,
  about 3.7% of a 38 ms round, if gathers ran back to back with nothing between them.
- Against the served aggregate probe (ringprobe, rank 0, 7.22 rows): stage 4 + doorbell→flag 22 + copy 3 ≈ 29 µs a
  probed gather, vs 17.8 µs standalone. That is ~11 µs a gather (~0.9 ms a round across 80 gathers) above the
  protocol price. **Treat it as an upper bound with caveats:**
  - the probe inflates the round;
  - the bench's back-to-back replays can overlap consecutive gathers in a way served gathers, separated by layer compute, cannot;
  - the probe pools all ring ops, not only hidden gathers.

  The per-sequence trace (0272) is what can split the ~11 µs into producer skew vs proxy vs protocol.

## A2: proxy thread scheduling while serving (read-only, production)

Snapshots of the `tf-rdma-proxy` thread before/after a 1-stream (6.2 s) and a 4-stream (14.3 s) sampled Tetris burst.

| burst | rank | CPUs allowed | voluntary sw | involuntary sw | migrations | runq wait | run time |
|---|---|---|---:|---:|---:|---:|---:|
| x1 | 0 | 15 | +1,520 | 0 | 0 | 0.30 ms | 5,344 ms |
| x1 | 1 | 15 | +519 | 0 | 0 | 0.00 ms | 6,614 ms |
| x4 | 0 | 15 | 0 | 0 | 0 | 0.00 ms | 15,360 ms |
| x4 | 1 | 15 | 0 | 0 | 0 | 0.00 ms | 15,360 ms |

- The proxy is pinned to CPU 15 on both ranks. Across both bursts it had **zero involuntary switches, zero migrations
  and ≤0.3 ms total run-queue wait**. Under 4 streams it spins the whole time (run time = wall, no switches at all).
  86 (rank 0) / 81 (rank 1) other engine threads are *allowed* on CPU 15, but none of them displaced the proxy.
- **Verdict: no proxy contention.** SCHEDULING.md row 3 (reserve CPU 15) is gated on observed contention, so it is
  dropped. Proxy service delay, if any, is not due to CPU scheduling.

## What this leaves for the trace (0272)

The protocol floor is now priced: 9.4 µs + 1.16 µs a row. The proxy is not descheduled. So any excess wait seen in
served gathers comes from producer skew between ranks (one rank's layer compute finishing later) or from proxy/NIC
service time per sequence. The trace's producer-interval vs doorbell→flag pairing decides between them, and
SCHEDULING.md row 1 (the MoE prefetch-fork move) is the candidate if it is skew after MoE.
