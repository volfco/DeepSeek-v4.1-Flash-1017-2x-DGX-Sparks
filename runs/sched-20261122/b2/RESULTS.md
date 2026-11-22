# Round B2: root-causing rank 1's wait (2026-11-22 21:30–22:40 UTC)

**Answer.** The large "rank 1 waits for rank 0" signal in k272 was mostly **our own measurement**: `arms.sh` starts
every arm with `TF_DS_ROUND_PROFILE=1`, which only rank 0 reads (rank 1 needs `RANK1_PROFILE`). Its clock marks made
rank 0 the late producer. Production sets no profile. With the profile off on both ranks, what remains in
production-like 1-stream decode is small and the other way round. **Rank 1 is late by ≈2–5 µs a gather, ≈0.15 ms a
round (~0.4%), and the paced L2 weight prefetch causes it**: prefetch saves rank 0 ~4 µs a gather more than rank 1.
With prefetch off the ranks are identical. At 4 streams the ranks are balanced. Production was restored and validated
after every engine-stopped step.

## Steps

1. **Paired-trace decomposition of k272** (`skew.py`): rank 0's excess GPU time between gathers was row-dependent
   (+4–16 µs at 1–7 rows, ~0 at 8+), uniform across layers, inside captured graphs. Corrected cost model: each rank
   proceeds when its peer's data lands, so imbalance costs ≈ half the producer gap per gather. That gives ≈0.25 ms a
   round at 1–7 rows, not the "≈1 ms" first written (k272 RESULTS and continue.md corrected).
2. **Not the box's config:** GPU clocks equal (2,385 vs 2,379 MHz under load, no throttle); identical driver, VBIOS,
   kernel, boot flags (`isolcpus`, `nohz_full`), governor, THP, ATS; no other GPU process on either node; rank 1
   runs the same `--graphs/--side/--prefetch/--aio` flags.
3. **Box microbench** (`hwbench.py`, engine stopped, same image and cpuset, 3 runs alone + 2 concurrent): rank 0's
   GB10 is 1.2–1.4% slower on 1- and 7-row bf16 GEMV chains (219.3–220.3 vs 216.6–217.3 µs; 266.6–268.2 vs
   263.8–264.5 µs), equal at 28 rows, on copy bandwidth (222.4–223.3 GB/s) and on launch latency (≈3.0 µs). The
   penalty survives `drop_caches` + `compact_memory` on the head, so it isn't fragmentation. Cause unknown (box-level).
   **It does not show in the engine's own kernels:** with prefetch off, the ranks' producer times are equal to 0.1%
   (step 6).
4. **nsys:** the engine wrapped in `nsys launch` (image `tensorfold-dsv41:nsys-10c9866`) captured only OS-runtime
   calls; nsys's CUDA injection does not take effect in this engine. Abandoned (not worth more downtime at this size).
5. **Row / concurrency split** (`rowsplit/`, profile **off on both**, trace N=16): the skew mostly vanished.

   | arm | rows | gap r0−r1 p50 attn / moe | late |
   |---|---:|---|---|
   | k0x1 (FIX_K=0, 1 stream) | 1 | +8.9 / +10.7 µs | rank 0 (the box's 1-row penalty) |
   | k0x4 | 4 | +1.6 / +2.0 µs | ≈ balanced |
   | k6x1 (FIX_K=6) | 7 | −3.7 / −5.4 µs | rank 1 |
   | k6x4 | 16 | +1.1 / −0.2 µs | balanced |
   | fx1 (unpinned) | 6 | −3.5 / −10.3 µs | rank 1 |
   | fx4 (unpinned) | 14 | +0.7 / −1.6 µs | balanced |

6. **Mechanism** (`mech/`, profile off, trace N=4, 8 × 384 greedy tokens per arm, deterministic: 39,294 records per
   rank in every arm):

   | arm | gap p50 attn / moe | late | cost | producer r0 / r1, attn mean | moe mean |
   |---|---|---|---:|---|---|
   | fx1L production | −1.5 / −5.2 µs | rank 1 | ≈0.14 ms/round | 198.2 / 200.7 µs | 537.4 / 542.0 µs |
   | fx1Lb production (repeat) | −2.6 / −5.2 µs | rank 1 | ≈0.16 ms/round | 197.0 / 201.3 µs | 538.4 / 542.4 µs |
   | **np1L `PREFETCH=0`** | +1.0 / +1.6 µs | balanced | ≈0.01 | **240.3 / 240.1 µs** | **558.8 / 558.5 µs** |
   | ns1L `SIDE=0` | +0.6 / −0.4 µs | balanced | ≈0.02 | 203.0 / 202.4 µs | 546.3 / 548.0 µs |

   - Prefetch saves rank 0 42.7 µs (attn) / 20.9 µs (moe) a gather, and rank 1 39.1 / 16.3 µs. That difference is the
     whole residual imbalance. Prefetch is a large net win for both ranks (the 1-stream run takes 4.12 s with it,
     4.34 s without), so it stays on.
   - The prefetch code has nothing rank-dependent (fixed budgets: 16 MiB a launch, 8 MiB behind `wo_b`, 4 MiB before
     an Engram layer; single-CTA paced launches on a stream of its own, `round.zig:801–870, 1212, 1418, 1752–1776`).
     The asymmetry is stable across restarts (same sign and size in fx1L / fx1Lb), so it is systematic, not
     allocation luck. Why rank 1 gains less from the same prefetch is the open question.

## Consequences

- **Production:** the residual is ~0.4% of a 1-stream round and ~0 at 4 streams. No production change.
- **Benchmark harness:** `arms.sh`'s `TF_DS_ROUND_PROFILE=1` is rank-0-only. Arm-vs-arm comparisons in every past suite
  stay valid (same in every arm), but absolute per-rank timings include rank 0's profile clocks. Trace work must set
  `TF_DS_ROUND_PROFILE=0` (as `rowsplit/` and `mech/` do).
- **Next, if worth it:** a rank-local prefetch-budget knob (e.g. `TF_DS_PREFETCH_BUDGET_MIB` read per rank) to test
  whether a larger or earlier prefetch on rank 1 recovers its missing ~4 µs a gather (≤0.15 ms/round, 1-stream only).
  SCHEDULING.md row 1 (move the MoE-side prefetch fork) is the same lever. Expected value is small.
