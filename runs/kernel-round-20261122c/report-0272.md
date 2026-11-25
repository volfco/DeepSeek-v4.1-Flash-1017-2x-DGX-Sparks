# report-0272 — TF_DS_RING_TRACE (sampled per-sequence ring timestamps on both ranks, drained outside the timed window)

**Status:** implemented; premise holds. Commit **`8a57bf54a0b5a0d760787408b18aaa108a4d8e05`** (`8a57bf5`) on
`kernel-0272` (`/home/colum/Workspace/tfref-k0272`, parent/base `27baa93` = recipe-0264): exactly one commit, subject
as the brief's export format, working tree clean. Switch: **`TF_DS_RING_TRACE=<N>`** (off by default; unset/`""`/`"0"`
off, a decimal N ≥ 1 samples every Nth ring sequence, anything else prints `TF_DS_RING_TRACE: expected 0 or a positive
integer` and refuses startup via the `fromText` error). **`TF_DS_RING_TRACE_OUT`** names the output file (rank-local,
as `TF_DS_ABSORB_AFTER_PICK`; default `/tmp/ring_trace.<rank>.jsonl`).

## What it does

Serves SCHEDULING.md's "Smallest experiment" steps 2–3. Per sampled ring sequence (the same rule on both ranks and on
device and host alike, so records pair by sequence), two bounded preallocated record rings per rank (4,096 × 128 B
each, a drop counter when full):

- **device ring** (written by the ring kernels from block 0's thread): key (ring sequence, round number, layer, op,
  rows, bytes) and the GPU points on the rank's own globaltimer — stage entry (after PDL), doorbell publication,
  collect entry (after PDL), flag observed, copy-out complete — plus the **previous gather's completion on the same
  stream** (per-stream prev-done cell stamped by every collect, sampled or not: the producer interval's landmark).
- **proxy ring** (written by `tf-rdma-proxy` in preallocated host memory, CLOCK_MONOTONIC, no syscalls beyond
  `clock_gettime`): doorbell observed, last write posted, each NIC part's receive completion observed, flag published;
  the first record after a proxy nap carries `woke`.

Op comes from the call site (never the shape): `attn_gather` / `moe_gather` / `kv` / `engram` / `head` / `other`.
Sampling is decided **on device from the live sequence** (`q % n == 0`), so graph replays record with fresh sequences;
buffer addresses are fixed at create, so captured graphs stay valid. Launches under capture carry a graph flag
(every replay of them does), eager launches do not.

Drain is outside the timed window: every 4 work units (a `Model.pass`/`Model.verify` boundary — the round boundary
both ranks share; rank 1's `tf-dsv41-lanes` frame loop calls the same `Model` methods) and at engine stop, the device
ring is bulk-downloaded after the mapped streams' work lands (no per-gather host sync) and both rings are appended as
JSON lines to the rank's file (`meta`, `gpu`, `proxy`, `drain` records). The proxy's records emit only below both of
its release-stored progress marks, so every emitted one is complete; drops are counted as expected minus kept. The
summarizer (`h0272/summarize.py`) reads both ranks' files, pairs by `seq`, and prints p50/p95 per (op, rank) of the
producer interval, doorbell→flag, collect-entry→flag, copy, proxy doorbell→last-post and last-completion→flag, plus
drops and sample counts. A python summarizer in the harness dir was the brief's first option; the pure Zig module
keeps the tested record layouts, sampling rule and interval/percentile helpers.

## Files touched

- `zig/src/families/dsv41/ring_trace.zig` (new, pure): `env_name`/`out_env`/`parse`/`fromText`; `Op`, `Key`,
  `Trace` (the 64-byte kernel argument, `@sizeOf` test); `GpuRec`/`ProxyRec` (the 128-byte records, offsets tested);
  the sampling rule (`sampled`/`slot`/`sampledCount`); `gpuTimes`/`proxyTimes` (same-clock intervals, null where an
  endpoint is missing), `percentile`/`Stats`; `outPath` and the JSON line writers (exact-shape test). 7 tests.
- `zig/kernels/cuda/dsv41/ring_trace_rec.cuh` (new): `TfDsTrace` and the record writes (`tf_tr_claim` at the stage
  entry, `tf_tr_publish` at the doorbell, `tf_tr_entry`/`tf_tr_flag`/`tf_tr_done` in the collect; helpers self-gate
  on the sampling rule; the identity word goes in last). Compiled for the CPU by the harness (`TF_DS_TRACE_EMULATE`)
  and by `tools/dsv41_shard_emulate` (verified still passing).
- `zig/kernels/cuda/dsv41/rdma_gather.cu`, `zig/kernels/cuda/dsv41_shard.cu`: the five ring kernels call the
  record helpers at the listed points (one thread each; `__syncthreads` stays inside the uniform guard).
- `zig/src/families/dsv41/rdma.zig`: `Settings.trace_n`/`trace_out`; `TraceState` (both rings, the round cell and
  `[4]` prev-done cells in one 48-byte device header, the drain's download buffer — all preallocated at create);
  `traceArg`/`streamOrd`/`traceGraph`/`proxyRec`/`traceNote`/`traceTick`/`traceDrain`; the proxy loop's record writes
  and progress marks; the final drain in `destroy`.
- `zig/src/families/dsv41/model.zig`: the switch parsed into `openRing`'s settings (refusing startup on a malformed
  value), `tickTrace` at `Model.pass`/`Model.verify` entry.
- `zig/src/families/dsv41/round.zig`, `draft.zig`, `prompt.zig`, `shard_ops.zig`, `ring2d.zig`,
  `rdma_bench_main.zig`, `dsv41.zig`: call-site keys (`Engine.gatherF32`, `shard_ops.ringSend`/`ringRecv` gained a
  `ring_trace.Key`), `traceGraph` around the two capture sites, zero keys on the untraced 2D path and bench, export
  + `refAll(ring_trace)`.

## Kernel signature changes (the contract asks for these exactly)

One trailing by-value parameter `TfDsTrace trace` (64 B: `recs`, `round`, `prev` pointers, `n`, `cap`, `rows`, `op`,
`layer`, `stream`, `flags`) on `dsv41_rdma_stage`, `dsv41_rdma_collect`, `dsv41_rdma_collect_into`,
`tf_ds_ring_stage_rows`, `tf_ds_ring_collect_rows`. The off path passes all zeros (the `probe`-style null: the kernel
checks `recs != 0` and writes nothing). Launch configs, grids, blocks and streams are unchanged.

## Launch counts at the served shape (TP2, `TF_DS_DRAFT_BLOCK=6`, graphs on, up to four streams)

| | before | after |
| --- | --- | --- |
| ring launches a round (hidden attn/moe/head/engram gathers + KV exchanges) | unchanged | **unchanged** — the same five kernels carry one extra zeroed argument (off) or write a few words from one thread into the trace ring (on) |
| new launches | — | none; the drain is host-side (one 8-byte upload per work unit, one bulk download + file append per 4) |

## Contract items

- **Off: launch- and byte-identical.** Every `tf_tr_*` write is behind `tf_tr_on` (`recs != 0`); the probe words and
  arithmetic, the proxy's loop work and every launch are untouched; `traceTick`/`traceDrain`/`proxyRec` return
  immediately when off. The only load-time delta is the switch parse (refusing startup on garbage).
- **On: outputs bit-identical by construction** — the trace writes only into its own device/host rings and reads
  nothing the pipeline consumes. The exact panel must confirm on hardware.
- **Graph capture:** addresses fixed at create; sampling and the round number read live on device (the round cell is
  host-updated at the round boundary, outside capture); no host syncs or pageable transfers inside `body`/graph paths.
- **Banner:** `dsv41: TF_DS_RING_TRACE: every Nth ring sequence recorded in 4096 device and 4096 proxy records a rank
  (first at sequence q)` once per rank behind an atomic `swap(true)`, at the first record written (proxy claim) or
  first drained record, whichever comes first.
- **Both ranks alike:** the switch is read at `Model.openRing` on every rank, the ticks and drains run in the shared
  `Model.pass`/`Model.verify`/`Ring.destroy` paths (the follower's frame loop reaches the same methods), and each rank
  writes its own file.

## Design decisions (with the why)

1. **Extend the probe mechanism, no new kernel** (the brief's instruction over the common brief's `ops.zig` switch
   convention): the record logic lives in `dsv41/ring_trace_rec.cuh` shared by `rdma_gather.cu` and
   `dsv41_shard.cu` so both kernel pairs speak one layout and the harness extracts one file. This is the round's
   stated deviation from "kernel in `dsv41_ops.cu`".
2. **Ops taxonomy:** the target's hidden attention/MoE gathers, KV exchanges (fetch, queries, rotated output), the
   Engram gather and the head are tagged at their call sites; the **drafter's gathers are `other`** (with their draft
   layer index): at `TF_DS_DRAFT_BLOCK=6` their 6-row payloads would be indistinguishable from verify windows of 5–7
   rows, and the scheduling experiment needs the target's hidden gathers isolated. `other` + `layer` still separates
   drafter traffic in analysis.
3. **Sampling and drops:** `q % n == 0`, record `(q / n) % cap`; a full ring overwrites the oldest sample and the
   drain counts the loss as expected minus kept over its window. `cap` 4096 with the 4-unit cadence keeps N ≥ 1
   within the ring at the served shape; lower N with slower drains only loses counted samples.
4. **Producer interval's landmark:** the previous gather's **copy-out complete on the same stream**, stamped by every
   collect into a per-stream cell. Copy-out complete is block 0's copy end — the existing `probe[2]`'s attribution,
   which the brief's "from block 0 thread 0, as the existing probe does" sanctions; SCHEDULING.md itself calls the
   landmark one "with some residual uncertainty".
5. **Drain trigger:** the brief's "every K rounds from the lanes loop after the round completes" implemented as a tick
   at the next work-unit boundary (`Model.pass`/`Model.verify` entry) every 4 units plus engine stop. The proxy's
   records emit below `min(max_doorbell, max_flag)` (release/acquire), so a record's events are all in before it is
   written out; records that straddle a dump wait for the next one instead of losing fields.
6. **Round number is a live device cell**, not a baked argument: captured graphs replay with new rounds. Uploaded
   stream-ordered at the tick (a pageable `cuMemcpyHtoDAsync` exactly as `prompt.upload`'s per-round list upload).

## Independent review (pre-commit, subagent)

Three majors found and fixed before the commit: (1) the stream map allowed 8 streams into 4 prev-done cells (now
capped at 4, the served maximum); (2) proxy records kept stale fields across slot wraparound and the `woke` bit was
sticky (records are now zeroed at claim, before the event writes); (3) NIC part timestamps were only stamped for the
flag-publishing sequence and shared the flag's clock read (now every completion stamps its own sequence's
`parts[d]`, and the flag gets its own read, so last-completion→flag is a real interval). Nits fixed: the off
argument is literally all zeros; `shortcut:` comments added for the TP4 path being untraced, the side-stream
landmark limit and the pageable tick upload. ABI, off-path parity, the settle protocol and call-site coverage were
verified clean.

## Gate results

1. `zig build test`: **exit 0** (513 tests incl. the module's 7). The known `lane_host.test.prefill` interleave flake
   appeared once in eight runs; three consecutive reruns clean. `zig build` (every binary, incl.
   `tf-dsv41-rdma-bench` and `tf-dsv41-lanes`) compiles clean.
2. CUDA syntax check: the shared `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails on the pristine base**
   (missing `float4`/`__clz`/`__nv_bfloat162` stubs — same as every round since 0265); run with this round's fixed
   stubs `h0272/stubs/` (from h0269's) it prints **`cuda syntax check: ok`** on the committed tree. `rdma_gather.cu`
   is not in the check's file list; checked separately with the same stubs (its single pre-existing `%globaltimer`
   inline-asm complaint is byte-identical on base and this tree; with the asm lines dropped the file is clean).
   `tools/dsv41_shard_emulate` still builds and reports `PASS`.
3. **CPU-emulation harness (`h0272/`, `build.sh` + sources + `run.log`): ok.** `splice.py` extracts the shipped
   record logic into `trace_kernel_current.inc`; `trace_emu.cpp` drives it as the kernels call it against an
   independent reference written from the brief, over the case matrix: sampling by sequence (n = 1, 2, 7), wraparound
   with mid-window drains and drop counts, **unsampled sequences sharing slots with sampled ones** (this case caught
   a real pre-commit bug: unguarded publish/entry/flag writes would have corrupted sampled records), the live round
   cell across multiple gathers per round, and the per-stream previous-gather landmark. `gen_trace.py` writes a
   synthetic two-rank trace with known injected delays; `summarize.py` + `check_attribution.py` show the injected
   producer-late (rank 0 attn_gather, +50 µs) and proxy-late (rank 1, +30 µs) land in exactly their (metric, rank)
   buckets with no false attribution, and that drops and sample counts survive.

## Known gaps (only the hardware gate can settle)

- The exact panel with the switch on (writes are to separate memory by construction; the panel is the proof) and the
  served-shape overhead of the on path (bounded by design: a few words a sampled gather from one thread, one 8-byte
  upload per work unit, one bulk download every four; price it in the gate's A/B).
- Cross-rank and GPU-vs-host comparisons are same-clock only (each rank's globaltimer and CLOCK_MONOTONIC are
  uncalibrated against each other and between hosts); the summarizer never subtracts across kinds or ranks.
- The copy-out point is block 0's copy end (see decision 4), so producer intervals carry the landmark's residual
  uncertainty; a grid-wide copy-done would need a block counter in the collect.
- **Graph miss vs steady replay is not distinguished** (the flag is baked at capture, and a cache miss's first replay
  carries the same flag as steady replays). The brief asked for it "if knowable at the call site": only graph vs
  eager is knowable. Correlate misses with `Graphs.run`'s existing `CUDA graphs: N kept` logs if needed.
- A `sendRows` whose collect runs on another stream (the early query send) leaves the sending stream's prev-done cell
  at its last same-stream gather: those records show no producer interval (see the `shortcut:` in `streamOrd`).
- The drain's file is append-only; a second run with the same `TF_DS_RING_TRACE_OUT` appends (the `meta` line
  repeats). The rank-1 drain cadence against its frame loop and proxy wakeups under nap (`TF_RDMA_IDLE_S`) are
  runtime behaviour only hardware will show.
