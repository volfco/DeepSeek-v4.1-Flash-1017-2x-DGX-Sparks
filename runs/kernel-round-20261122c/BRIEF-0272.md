# BRIEF-0272 — sampled per-sequence ring trace on both ranks (diagnostic)

**Switch:** `TF_DS_RING_TRACE=<N>` (off by default; N = sample every Nth ring sequence, 0/unset off).
**Module:** `zig/src/families/dsv41/ring_trace.zig` (pure: parse, record layout, host-side pairing/summary helpers).
**Commit subject:** `0272 dsv41: TF_DS_RING_TRACE (off by default): sampled per-sequence ring timestamps on both ranks, drained outside the timed window`

This patch is diagnostic: it never changes a served value. Read `/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/overnight-20261122/ringprobe/SCHEDULING.md` in full first. It is
the plan this trace serves (its "Smallest experiment" step 2 and 3). The existing aggregate probe (`TF_DS_RDMA_PROBES`,
`rdma.zig` `probes_env` / `Ring.probes`, kernel side `zig/kernels/cuda/dsv41/rdma_gather.cu` `probe[0..4]`) is the
mechanism to **extend**, not replace: it sums stage / doorbell-to-flag / copy over probed gathers on rank 0 only, and only
rank 0 prints it (`native.zig` profile line). It pools every op, so it can't say which exchange waits or why.

## What to record

Per sampled ring sequence (both ranks, the same sampling rule so records pair by sequence), into a **bounded preallocated
per-rank buffer** (device-side ring of fixed records + a host-side one for the proxy), with a drop counter when full:
- key: ring sequence, round number, layer, op (hidden attention gather / hidden MoE gather / KV exchange / Engram / head /
  other: take the op from the call site, not guessed), rows, bytes;
- GPU (globaltimer, same clock as `now_ns()` in rdma_gather.cu): stage kernel entry, doorbell publication, collect entry
  after PDL, flag observed, copy-out complete; plus the previous gather's completion on the same stream (the alignment
  landmark: "GPU time from the previous gather's completion to this doorbell" is the producer interval);
- proxy (host CLOCK_MONOTONIC in preallocated host memory, no allocation, no syscalls beyond clock_gettime): doorbell
  observed, last write posted, each NIC part's receive completion observed, flag published.
- idle wakeups (first after a nap) flagged; graph replay vs eager flagged if knowable at the call site.

Drain **outside the timed window**: on the existing profile print points (when TF_DS_ROUND_PROFILE prints) or a dedicated
dump trigger (e.g. at engine stop, or every K rounds from the lanes loop after the round completes), writing JSON lines to
a file path from `TF_DS_RING_TRACE_OUT` (default `/tmp/ring_trace.<rank>.jsonl`), **both ranks** each writing its own
file. No per-gather host sync: the device ring is downloaded in bulk at drain.

Same-clock only: never subtract a rank-0 timestamp from a rank-1 one, or a GPU timestamp from a CPU one. The analysis
pairs records by sequence and compares durations. Provide a small host-side summarizer (python in the harness dir, or the
pure Zig module + a tiny CLI) that reads both ranks' files and prints p50/p95 per (op, rank) of: producer interval,
doorbell→flag wait, collect-entry→flag, copy; proxy doorbell→last-post, last-completion→flag; plus drops and sample counts.

## Contract

- Off: launch- and byte-identical (no new kernel args unless they're passed as null/0 in the off path the way
  `probe` already is; say exactly what changed in the kernel signature).
- On: outputs bit-identical (the exact panel must pass with it on). Overhead must be bounded; the trace writes a few
  words per sampled gather from block 0 thread 0, as the existing probe does. Graph capture: buffer addresses are fixed
  at load, so captured graphs stay valid; the sampling decision must be made on device from the live sequence number
  (graph replays re-run with new sequences), not baked in at capture time.
- Rank 1 writes its own trace file. Check how rank 1 runs (`tf-dsv41-lanes` on the worker) and where it can drain.
- Banner: log once per rank when the first record is written, with the buffer size and sampling interval.

## Harness (`h0272/`)

CPU emulation of the device-side record ring (sampling rule by sequence, wraparound, drop counting, multiple gathers per
round) against an independent reference; the summarizer run on a synthetic two-rank trace with known injected delays
(producer-late on rank 0, proxy-late on rank 1) showing that it attributes them correctly.
