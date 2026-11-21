# Engram staging to the device (item E, proper) — design verdict

Status: **not built, not buildable in this environment.** The prerequisite (item E's smaller sibling,
"both Engram layers' rows land at the first cut") **is** built — patch `0128`
(`TF_DS_ENGRAM_BOTH=1`, branch `decode-step-structure`, off by default, bit-identical: two stretches a
round instead of three). The full staging design needs a pair and GDS or a polling kernel to be settled
honestly. This note records the analysis so the next agent does not re-derive it.

## Where the Engram bytes live and how they move today

- The tables are ~203 GB on each Spark's own NVMe (shards 47/48 of the checkpoint, `HEAD_ENGRAM_DIR` /
  `WORKER_ENGRAM_DIR`; see `docs/zig-engine.md`, `docs/requirements.md`, `docs/configuration.md`). They
  are **not** in GPU memory and cannot be: GB10 device memory is far smaller.
- A round's rows are read by the host per rank (`engramAhead`/`engramLand`, `round.zig`): hashed once
  (both Engram layers, after the first stretch is issued), submitted via Linux AIO on O_DIRECT
  (`ah.aio`, `ah.ids[j]`), then **waited for, decoded row-by-row on the host** (`engram.decodeRow`,
  `FP8 -> bf16`) and uploaded H2D into `rd.e_in` (bf16 rows, `R * k * 2` bytes) before each Engram
  layer's stretch runs. The decode and the upload are the host steps; the read itself is async and
  overlaps stretch 0.
- A round is cut into stretches at each Engram layer (today: three; with `0128`: two) so the host can
  land the rows between stretches. The rows must be in device memory before the layer's projection reads
  them.
- Rank-1 profile (overnight, 8 streams): `engram_read_ms` ~0.67 ms, `engram_hash_ms` ~0.98 ms a round.
- Rank-0 profile, 2026-11-15 (`runs/ultrafast/prof/*.log`, graphs on): Engram phase **0.47 / 0.57 ms a
  round** (~1% of a 34–56 ms round; `forward` is 89% of `step`). The Engram host landing is **not** the
  critical path today.
- Both Engram layers' reads are already submitted at once (`engramAhead` handles all layers up front);
  nothing about the submit changes in any variant below.

## The candidate mechanisms

### A. GPUDirect Storage (cuFile) — the only clean no-host version

`cuFileReadAsync` reads the tables' row ranges DMA straight into a device (or pinned) buffer on a CUDA
stream, stream-ordered with the stretch kernels that consume them. The GPU-side then needs:

1. a device-side decode kernel replacing `engram.decodeRow` (mechanical port: per row, per block: read
   the FP8 codes and the scale bytes, unpack, multiply, write bf16 — the exact rounding is already known,
   `engram.zig`), and
2. the AIO/wait/decode/upload replaced by: open the cuFile driver + register the device buffer once,
   keep per-table handles, submit one `cuFileReadAsync` per layer's row range at `engramAhead` time,
   and let the engram layer's first kernel launch wait on the same stream (no event, no host, no H2D).

Open questions that **cannot be answered without the pair and without cuFile**:

- Whether the pair's NVMe + driver actually do GDS at usable rates (GB10, Blackwell, driver 580.x —
  unverified; cuFile has its own GPU/NVMe/OS matrix).
- Whether `cuFileReadAsync`'s stream-ordered semantics survive CUDA graph capture on GB10 (graphs span
  every stream of a stretch in this engine), or whether the read must stay outside the captured graph
  with the decode kernels inside (still no host step, just two submission points).
- The FP8-decode port has to be byte-equal to `engram.decodeRow` (exact.py gate on the pair, not
  buildable or testable here).
- Error handling: a cuFile failure at round time must end the pair like every other rank fault
  (`EngramRankFault` precedent).

Cost if it works: zero host landings a round, one stretch (one graph) for the whole round, ~0.5–1 ms a
round on the rank-0 figures, plus one less graph per round. Gain: **small**, and it is overlapped work
anyway — exactly the class of change the profile (`forward` 89% of `step`, host ~1%) says is not the
lever.

### B. Device kernel polling a pinned host completion word — possible, not recommended

`libaio` completions land in userspace host memory (the CQ ring); a device kernel could poll a pinned
word that the AIO path advances on completion, then decode the rows from pinned memory (zero-copy over
the CPU↔GPU link) and be the first thing the Engram stretch waits on — no cuFile dependency, and it
would also avoid the host round trip. Why it is rejected anyway:

- The SM spins for the whole table-read window (~100–700 µs typical), competing with the SMs the round
  needs — the prefix exists because decode GPU time is scarce.
- Correctness hinges on the exact libaio CQ layout and overflow flags — a subtle, rank-pair-coupled
  spin protocol written blind (no pair to test on), the same "assumed but not seen" risk class as `0122`'s
  `_score` arithmetic, without `0122`'s upside.
- Zero win over option A if GDS works; a fragile win over A if it does not.

### C. Overlap what already exists (done)

0128: both layers decode+upload at the first cut. Layer 14's landing no longer gates the layer-14
stretch; one stretch boundary removed. Bits identical. This is the whole of the measured cheap win.

### D. Tables in device memory at start — Baseten's layout, not portable

Baseten holds tables in GPU memory (`engram_fork`, `model.rs:3238`, forked at round start, joined before
the layer; a round stays one graph). Here the tables are ~203 GB per rank; GB10 does not have the
memory. Not an option, and not fixable by sparsity (the reads are random by hash).

## Also checked: the drafter's KV projections

Baseten's drafter shares one grouped KV projection across its three blocks (`mtp_wkv`,
`model.rs:3695-3702`). Here the drafter's three blocks run **sequentially** (stage 0 → 1 → 2), each
block's `wkv` consumed by the next stage's input, so a cross-block grouped launch is not available;
the absorb path already groups all three stages' `wkv` into **one** `grouped` call
(`draft.zig:505-511`, `stage_kv`), and each block's `attn_in` groups its own `wq_a`/`wkv`
(`draft.zig:677-678`). Nothing to change.

## Verdict

- Do not build A without (1) a 50-line GDS read benchmark on the head/worker proving cuFile works on
  this pair at round-read rates, and (2) the pair measuring the Engram landing's share with the `0123`
  serve-profile line (`TF_DS_ROUND_PROFILE=1`, scrape `/metrics`). If the landing is not on the critical
  path after 0128, A has no justification.
- Do not build B at all.
- 0128 is the answer item E gets until a pair measurement says otherwise.
- If A is ever built: it replaces AIO with cuFile, `decodeRow` with a device twin, and the landing with a
  stream-ordered read; every Engram consumer then reads `rd.e_in` as it does today. Gates: exact.py on the
  pair, the AIO path kept behind `TF_DS_ENGRAM_AIO=1` as the fallback for hosts without GDS, and a GDS
  failure at init refusing the start (never silently half-GDS and half-AIO in one round).
