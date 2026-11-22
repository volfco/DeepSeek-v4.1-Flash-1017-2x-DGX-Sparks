# Decode ring-gather attribution (TF_DS_RDMA_PROBES=1), production recipe-0264, 2026-11-22 13:40–13:44 UTC

One arm on production's image and switches plus `TF_DS_RDMA_PROBES=1` (`rdma.zig:97`: the ring kernels time each
gather's stage, its doorbell wait for the peer's flags, and its copy-out). The kvreuse-size workload ran
(4K/32K/128K × 1/4 streams). Rank 0's cumulative profile at the end (904 rounds, 7.22 rows a round):

| Per gather (mean over every probed gather) | µs | share |
|---|---:|---:|
| stage (local rows into the send slot) | 4 | 14% |
| flight (doorbell: waiting for the peer's flags) | 22 | 76% |
| copy (peer's rows out of the slot) | 3 | 10% |
| **total in-kernel** | **29** | |

**Reading.**
- The ~40 µs a gather estimated from the phase timer is ~29 µs in kernel; the rest is launch/dependency gaps around it.
- **Flight dominates.** The bytes are small: a decode gather moves a few KiB to tens of KiB. At CX7 bandwidth that is
  well under 5 µs on the wire, so most of the 22 µs is the peer arriving later, i.e. per-layer skew between the ranks,
  not transport. The ranks are balanced over a round (item 1, closed), but a gather still waits for whichever rank is
  behind at that layer.
- So a faster transport would not move this. A cheaper decode gather comes from fewer gathers (fusing two per layer
  into one, or the 0259-style reuse) or from overlapping independent work with the flight (start the next layer's
  rank-local work before the gather lands), not from a faster ring.
- **Caveat:** probes are not free. Enqueue went 3.15 → 8.43 ms and forward 37.5 → 44.7 ms a round, against the kept
  control's profile on a different workload, so the absolute µs are upper bounds. The split is the finding.

Workload cells (with the probe overhead, not comparable to the controls): 4K 33.4/55.4, 32K 32.4/62.0,
128K 32.5/57.4 ms a round at 1/4 streams.
