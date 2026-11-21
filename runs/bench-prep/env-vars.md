# env-vars — the new switches, as implemented (patches `0129`-`0145`)

**Status: written against the patches, 2026-11-16.** Unlike the first draft of this file, every row below is read
off a landed patch (`zig/patches/0129`-`0145`, exported by `scripts/port-patches.sh` from engine `bench-prep` over
`decode-step-structure`), not off the plan's prose. Where the plans numbered a patch differently (they said
`0130`-`0147`), the **plan number** is given in the row so the older docs still map.

Patch-number map (plan → landed): map-warm 0130→**0129**, `DECODE_THREADS` 0131→**0130**, `POOL` 0132→**0131**,
`CPUS` 0133→**0132**, `AIO_BATCH` 0134→**0133**, `engram_host_ms` 0135→**0134**, `SPILL_SHARED_MIN_TOKENS`
0136→**0135**, row-cache map 0137→**0136**, `RehashMap` 0138→**0137**, `queue_s`/`restore_s` 0139→**0138**,
driver pins 0141→**0139**, `EXPERT_BYTES` 0142→**0140**, `PREFILL_PROFILE` 0143→**0141**, probe 0144→**0142**,
`ALLOCATE_LEARNED` 0145→**0143**, `pass_send_ms` 0146→**0144**, `RDMA_PROBES` 0147→**0145**.
Two planned items are **not landed**: `TF_DS_ENGRAM_SPLIT=2` (two striped AIO rings — `LEAD` already owns the second
ring and `aioGather` writes contiguous slices, so striping is not a minimal change) and the streamed-reply text scan
(W10-b, plan `0140`).

## The rules every row is held to

- **Off by default.** Unset leaves the replies byte-identical and `/metrics` untouched, except the two done-line
  fields, which are appended after every existing field (the `0070` prefix pattern) and are attribution only.
- **Rank scope.** `rank-local` = each rank's process reads it, no fingerprint bit, no `switches` frame: the two ranks
  are set alike by convention and an uneven pair is still bit-identical. `pair-wide` = both ranks must agree, the
  value rides the features string / `switches` frame, and an uneven pair refuses the start.
- **Gate.** `tools/validate.py … --compare runs/matrix/base.validate.json` (it prints `all passed`, exit 0; the
  matrix labels that outcome `IDENTICAL`), plus `tools/exact.py` where the switch claims bit-identity beyond what
  `validate.py`'s 8 prompts reach. `0129`-`0133`, `0135`, `0136`, `0137`, `0139`-`0145` are identity-class;
  **`0143` is sequential-class** (it changes a round's row counts, so `validate.py`'s concurrent check matters as
  much as `exact.py`).
- **Keep/drop bar.** One bench block cannot promote anything. A switch that passes its gate but reads neutral is
  *kept as a measured row and dropped as an option*.

## The table

| Patch | Switch / field | Default (unset) | Scope | Semantics | Gate | Expected effect | Keep/drop bar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0129** (`0130`) | `TF_DS_ENGRAM_SCALES=map-warm` | `io`; `map` unchanged | rank-local | `map` plus a detached thread that walks the scales mapping in 64 MiB `madvise(MADV_WILLNEED)` pieces; open does not wait | validate + exact | removes `map`'s cold page-fault path (overnight: `map` +3.5-8.8 % prefill in steady state, -82 %/-41 % on the first two requests after a restart) | `map-warm` ≥ `map`'s steady state **and** its cold first/second request no longer collapse; then `map`'s keep/drop can be signed |
| **0130** (`0131`) | `TF_DS_ENGRAM_DECODE_THREADS=1..16` | `8` | rank-local | sizes the `FAST` row-decode pool (FP8 → bf16) | validate + exact | the container has 5 fast cores: fewer decode threads may leave the engine more | a decode/prefill gain outside the band at one depth without loss at another; otherwise measured-only |
| **0131** (`0132`) | `TF_DS_ENGRAM_POOL=1..256` | `64` | rank-local (`ENGRAM_AIO=0` only) | sizes the pread pool the ring-off path uses | validate + exact | read-ahead stall vs pread throughput | a gain against `AIO=0` alone; a loss vs the ring (the base) is expected — this row is `AIO=0`'s tuning, not a win over the ring |
| **0132** (`0133`) | `TF_DS_ENGRAM_CPUS=a-b,c-d` | unset: no affinity syscall | rank-local (with `CPUSET`) | every Engram host thread (read pool, decode pool, warm walk, `LEAD` readers, prompt read-ahead) pins itself at spawn; a bad spec refuses the start | validate + exact | moves hashing/decoding/reaping off the cores the engine uses | a gain at 1 and 4 streams with no loss at the other; the only CPU-contention experiment |
| **0133** (`0134`) | `TF_DS_ENGRAM_AIO_BATCH=n` | unset: `slots/4` | rank-local (prompt path only) | bounds the rows one prompt AIO batch submits; the decode round's per-layer batch is untouched | validate + exact | read wall vs reap churn | any gain outside the band on 8K/32K prefill |
| **0134** (`0135`) | `engram_host_ms` (rides `TF_DS_ENGRAM_METRICS`, on by default) | field absent with `METRICS=0` | rank-local (each rank's own line) | per-request delta of the Engram host clock (hash+decode+upload wait, not the AIO wait), on the done line and the driver's request line | none (attribution) | none on time; makes the host items visible without `--profile` | n/a — keep (zero cost, counters already existed) |
| **0135** (`0136`) | `TF_DS_SPILL_SHARED_MIN_TOKENS=n` | off | **pair-wide** (`spill_shared_min=N;` in the features, named only when on) | a keep at a shared cut is written from `n` tokens even below `SPILL_MIN_TOKENS` (floor = lower of the two) | validate + exact | a 2048-token shared block survives a restart; short sessions recover instead of re-prefilling | a restart between rounds leaves the shared cut resumable and the spill churn stays bounded |
| **0136** (`0137`) | — (`TF_DS_ENGRAM_ROWCACHE_MB` sizes it) | n/a | rank-local | W1: dense array map + `swapRemove`, so a full cache's misses stop probing the whole table | validate + exact + the W1 kill test | removes the progressive regression the overnight pass saw after ~370-470K uncached tokens | keep unconditionally if the regression is gone; if it persists, the cache is ruled out and production runs without it |
| **0137** (`0138`) | — | n/a | rank-local | W10-c: `core.RehashMap` (rehash at a quarter of capacity) on the spill tier's `points` and the Engram AIO's `batches` | validate + exact (fix; no switch) | bounded probes on the long-lived maps | keep (a fix); no measurable speed claim |
| **0138** (`0139`) | `queue_s`, `restore_s` on the done line; `ttft_s`/`prefill_s` at 3 decimals | fields present, `-` when the request never ran | rank-local | queue = admission → prefill start (includes a parked restore); restore = time parked on a restore | none (attribution) | none on time; decomposes the 36 s median TTFT | n/a — keep |
| **0139** (`0141`) | `TF_DS_FIX_K`, `TF_DS_DRAFT_DEPTH2..5` in `tf-dsv41-lanes` | unset: nothing changes | driver only (a profiling instrument) | the batch driver now reads the same pins the served process does (same refusal on two pins) | none (instrument) | enables W4's fixed-rows cells in the driver | n/a — keep (needed by W4) |
| **0140** (`0142`) | `TF_DS_EXPERT_BYTES=1` | off | rank 0's process | per round: places and their trellis bytes, printed beside `expert_prune`'s line | none (attribution) | none on time; makes `moe` ms regressable on bytes | n/a — keep (needed by W4) |
| **0141** (`0143`) | `TF_DS_PREFILL_PROFILE=1` | off | each rank prints its own | per-chunk phases + attention/router/dense/expert sub-marks, ms a chunk on the `--profile` line | none (attribution; the marks sync) | none when off; sizes the prefill and M6 | n/a — keep (the instrument W4 lacks) |
| **0142** (`0144`) | `tf-dsv41-probe` (tool) | not run | rank-local binary | pure-read GB/s over expert trellis ranges: `distinct-sorted`/`distinct-shuffled`/`repeat-first` × prefetch off/on | **UNVERIFIED-BUILD** — `probe.cu` has never been through nvcc; the image build is its first compile | settles bus-bound vs trellis-decode-bound (W4's decision) | the probe's GB/s vs the MoE's measured rate: ≈equal → bus-bound (M5 dead); ≥1.5x → decode-bound |
| **0143** (`0145`) | `TF_DS_ALLOCATE_LEARNED=1` | off | rank 0 (the depth rule is rank-0 host state) | prices a round's rows from the depth rule's learned costs, per-row fallback to the static table | validate (incl. concurrent) + exact | the static table reads 30 ms at one row and 2.6-3.6 ms/row past five against ~11.4 + 4.9 measured: wrong grants, worst in wide rounds | ≥2 % at 8 streams with no loss at 1; the known risk is that `observeCost` is fed by single-stream rounds only |
| **0144** (`0146`) | `pass_send_ms` (profile column; rides `--profile` / `TF_DS_ROUND_PROFILE`) | n/a | rank 0's profile line | the pass frame's send, from issue to the pass timer's start | none (attribution) | none on time | n/a — keep |
| **0145** (`0147`) | `TF_DS_RDMA_PROBES=1` | off (keys print as zeros so both lines keep equal key sets) | both ranks (each its own ring) | `Ring.probes()`' first caller: stage/flight/copy of each decode gather in the round profile | none (attribution) | none when off; splits a gather that costs more than its bytes | n/a — keep |

## The no-env patches (retest / metric only)

| Patch | What it is | Why there is no env | Gate | Reading |
| --- | --- | --- | --- | --- |
| row-cache map fix (`0136`) | the row cache's `std.AutoHashMapUnmanaged` → dense `AutoArrayHashMapUnmanaged` + `swapRemove` | it is a bug fix (`TF_DS_ENGRAM_ROWCACHE_MB` already sizes it) | `validate.py`, `exact.py`, then the W1 kill test at 64 and 256 MB | the onset of the overnight regression should move with the budget if this was the cause |
| rehash helper (`0137`) | `core.RehashMap` on `spill_tier.points` and `engram_aio.batches` | it is a fix with no policy choice | `validate.py`, `exact.py` | no timing claim; the value is that long runs stop degrading |
| `queue_s`/`restore_s` (`0138`) | two done-line fields + 3-decimal TTFT | attribution only | none | watch the first real traffic: `queue_s` is the number the repo never had |

## Not landed

| Plan item | Why |
| --- | --- |
| `TF_DS_ENGRAM_SPLIT=2` (plan `0134b`) | `paio` is one prompt ring, `LEAD` already owns the second (`paio2`/`lead_ring`), and `aioGather` submits contiguous id slices into contiguous destinations: striping needs per-row destinations or a bounce copy plus a second batch-id set |
| streamed-reply text scan (plan `0140`, W10-b) | not implemented; the allocator question (`splitThinking` handed the generation's arena each round) is unresolved and the change must be bit-identical for stop sequences |

## Caveats for the arm runner

- The pair's **first request after a restart deterministically fails** (`cuLaunchKernel: CUDA_ERROR_INVALID_VALUE`,
  `prompt=8 tokens=1`) on every recent image. It is environmental: warm up once before timing anything cold-sensitive
  (`map`'s and `map-warm`'s first-request cells above).
- `0142`'s `.cu` is uncompiled here. If the image build fails on `probe.cu`, remove that patch and rebuild: it is a
  tool, nothing served depends on it.
- `0134`, `0138`, `0144` change log/profile output, not behaviour: an arm that only enables them measures nothing —
  read their fields off the base arm.
