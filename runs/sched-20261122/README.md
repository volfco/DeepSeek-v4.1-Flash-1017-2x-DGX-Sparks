# Scheduling round (from ringprobe/SCHEDULING.md), 2026-11-22

Plan derived from [SCHEDULING.md](../overnight-20261122/ringprobe/SCHEDULING.md). No scheduling change goes on hardware
until the trace says which signature applies (its change table is keyed on trace signatures; a moved wait can look like
a gain on one rank).

## Round A — no engine code
| Step | What | Where | Status |
|---|---|---|---|
| A1 | Standalone ring bench at the real payload: width 5120, rows 1/7/14/21/28 (+ throwaway), GID 3, PDL on, production HCAs per rank, `TF_RDMA_IDLE_S=2 TF_RDMA_NAP_US=500`, cpuset 15-19, **only the proxy thread** pinned to CPU 15 (the bench never passes `.cpu`, so pinning the process would put the launcher on the proxy's core). 3 repeats + one width-2560 pass. Engine stopped; restore + validate after. Hidden gathers are the plain `gather` (`prompt.zig` `gatherF32` → `Ring.gather`), so compare `us_gather`. | `rdmabench/` | **done** ([RESULTS-A](RESULTS-A.md)): 9.4 µs + 1.16 µs/row |
| A2 | Proxy scheduling, read-only, while serving: proxy TID mask, voluntary/involuntary switches, migrations, run-queue wait across a 1-stream and a 4-stream burst on both ranks; every engine thread allowed on CPU 15. Decides whether the CPU-reservation arm is warranted (SCHEDULING.md row 3 gates it on observed contention). | `proxysched/` | **done** ([RESULTS-A](RESULTS-A.md)): no contention, row 3 dropped |
| A3 | Probe metric: `suite_probe.py` now timestamps each SSE chunk → `chunk_gap_ms_p50/p95` (inter-chunk = inter-round as the client sees it, not inter-token), `client_span_s`, `client_tok_s` (finer than the engine's 0.1 s `decode_s`). `--stream-bursts` adds streamed x1/x4 rows (new labels; `burst()` unchanged so old x2/x4 rows stay comparable). `probetable.py --metric chunk_gap_ms_p95|client_tok_s`, plus a rounds-match column (SCHEDULING.md's "matched round/row counts"). | `suite_probe.py`, `probetable.py` | done; use `PROBE_ARGS='--narrow --sampled --stream-bursts'` from the next suite |

Not possible without code: the both-rank aggregate probe. Rank 1 never prints its ring probe words (`native.zig` prints
rank 0's profile only), so the two-sided view is in B.

## Round B — trace patch
0272 `TF_DS_RING_TRACE=<N>` (pi agent, `runs/kernel-round-20261122c/BRIEF-0272.md`): sampled per-sequence records on
both ranks keyed by sequence/round/layer/op/rows/bytes, GPU and proxy timestamps, drop counts, drained outside the
timed window, plus a same-clock summarizer (p50/p95 per op and rank). Gate: off identical, on exact, overhead measured.

## Round C — scheduling candidates (after B's trace)
In SCHEDULING.md's order, each gated on its trace signature, judged on probe-off graphs-on forward ms, sampled tok/s
and chunk-gap p95 at 1 and 4 streams, exact panel + seeded identity + matched rounds:
1. MoE-side weight-prefetch fork moved to just after hidden-stage publication (one fork point, bytes fixed). The only
   candidate with prior evidence (the rank asymmetry after MoE in the phase results).
2. Verify-frame / stretch-boundary preparation moved ahead of the boundary.
3. ~~Proxy CPU reservation~~: dropped, A2 showed no contention (0 involuntary switches, 0 migrations, ≤0.3 ms runq wait).
