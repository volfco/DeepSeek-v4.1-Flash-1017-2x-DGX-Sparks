# Benchmark gaps, Baseten follow-ups and the MoE byte budget: findings and action items

Review of 2026-11-15. Three questions: what `to-benchmark.md` leaves untested that could change its results, the same
for the Baseten-derived work (`docs/baseten-ultrafast-plan.md`, patches `0119`-`0122`, items A-E), and what can be
done to the MoE step. The proposals and estimates below record the November 15 review. W1, W4's instruments, W5 and W10's cache/TTFT work subsequently landed. The row-cache soak, scales warm-up, learned allocator and eager expert-bandwidth results are in [combo2](../runs/combo2/RESULTS.md) and [W4](../runs/w4/RESULTS.md). The [capped T15 trial](../runs/review-round/RESULTS.md#applied-cap-t15) found no material wall-time gain at eight fixed rows. A generated prune-off regression set now exists. Quality on independent references and sampled concurrent tasks, and the speed of the cap itself versus production, remain open. See the [current evaluation order](../to-benchmark.md#current-evaluation-order) for work still justified on the pair.

How to read it:

- **Grade.** *measured*: a figure in a log under `runs/`. *derived*: arithmetic on measured figures (the arithmetic is
  shown). *unverified*: a hypothesis from reading code, with the experiment that settles it.
- **Bits.** *identical*, *sequential* (a reply keeps its bits alone; round composition changes, so gate with
  `validate.py`'s concurrent check too), *LOSSY* (opt-in only), as in the Baseten plan.
- **Where.** *repo* (this repo's Zig or own CUDA, patchable), *kit* (the kit's recorded kernels), *pack* (the weight
  file's format), *config* (settings and scripts only).
- **Code references** are paths under `zig/src/` of the engine checkout `../tfref-next` as read on its
  `decode-step-structure` working tree (`8ff4797` plus uncommitted work that belongs to another agent). Re-check a
  line before editing it. That checkout was read only; nothing in it was changed.
- **The expert kernels' source is not on this machine.** `grouped_cp_kernel`, `decode_prep_kernel`, `grouped_mma_kernel`
  and `grouped_rows_kernel` are cubins in the kit. A search of `tfref-next/zig/kernels/`, `tools/`, `docs/` and the git
  history of `tfref-next` and `../TensorFold` found no source. Everything said about them comes from the launch
  wrappers and their doc comments (`families/dsv41/exl3_experts.zig`). `../TensorFold/zig/kernels/cuda/exl3_block/` is
  a different EXL3 kernel family with source; it is not what the served build runs.

## 0. What changes what to do next

| # | Finding | Grade | Package |
| --- | --- | --- | --- |
| 1 | The Engram row cache (`0114`) evicts from a hash map and never rehashes it. This is the bug `0119` fixed in the token cache. It fits the overnight run's progressive slowdown (8-stream decode -17%, two rank-pair breaks). `to-benchmark.md` still signs `0114` **keep** | code read; symptom measured; link unverified | W1 |
| 2 | Two result sets disagree and only one is in `to-benchmark.md`. `runs/overnight/RESULTS.md` (untracked) rules the row cache out, finds `SCALES=map` positive in steady state, `AIO=0` positive on prefill, `T25` -12% at 8 streams, and the old 125-word corpus ~10% optimistic on prefill | measured | W2 |
| 3 | The base of every row is lossy. `.env` sets `TF_DS_T13_EXPERT_PRUNE=1`. The prune removes ~3 of ~16.5 expert places a layer, worth ~10-12% of a round. No arm has it off, and there is no quality number | measured (speed), none (quality) | W3 |
| 4 | A decode round's MoE time tracks the expert bytes it reads at ~200-220 GB/s, which is at or above what informal measurements give this memory bus (~205 GB/s on a test that also writes). If the bus is the limit, fusing or reordering launches cannot help, and time is bytes per accepted token | derived; cause unverified | W4 |
| 5 | Attention is ~30% of a 1-stream round, not ~6%. `to-benchmark.md` reads the `attn` leaf alone (1.8 ms); its sibling leaves bring it to 9.8 ms, nearly independent of rows | measured | section 1 |
| 6 | The exact-resume failure (`0121`) has a concrete candidate cause: kernel paths are chosen by a chunk's row count (16, 64, 128), and a resumed off-grid chunk has a different row count from the cold chunk that holds the same rows | unverified (thresholds read in code) | W8 |
| 7 | The head's slow start is probably its storage path, not the serial reader. The same reader reads the same 95.1 GiB in 20.2 s on the worker and 88 s on the head, whose weight file is under `/nfs1` | measured (times, mounts); cause unverified | W7 |
| 8 | Production load is unprofiled. `PARALLEL=8` is set; the round profile stops at 4 streams and ~20-token prompts | measured (absence) | W4 |
| 9 | The allocator's cost table does not fit this pair. A two-point fit of measured rounds gives ~11 + 4.9 ms a row; the table says 30.0 for one row and 2.6-3.6 ms a row past five | derived (two points) | W5 |
| 10 | Items A, D, E of the decode-step work have ceilings of about 1% each on the measured profile; C is 1-3% and unsized | derived | section 3 |

## 1. The measured round, and the byte budget

Source: `runs/ultrafast/prof/` (2026-11-15, `tf-dsv41-lanes` as both ranks, production settings with spill off,
`TF_DS_T13_EXPERT_PRUNE=1`). The phase timer's leaves are siblings, not a tree: they sum to `forward_ms`.

| Eager round (`--profile 2`, `GRAPHS=0`), ms | 1 stream, 3.88 rows | share | 4 streams, 7.45 rows | share |
| --- | --- | --- | --- | --- |
| MoE experts (`moe`) | 16.10 | 50% | 31.45 | 63% |
| routing (`m_rt`) | 1.14 | 4% | 1.33 | 3% |
| attention (`attn` + `a_in` `a_q` `a_idx` `a_core` `a_woa` `a_x` + `sh_fetch` `sh_rows`) | 9.84 | 31% | 11.43 | 23% |
| gathers (`gather_a` + `gather_m`) | 2.11 | 7% | 2.21 | 4% |
| mixes and post | 1.39 | 4% | 1.46 | 3% |
| head | 1.09 | 3% | 1.19 | 2% |
| Engram | 0.47 | 1% | 0.57 | 1% |
| sum (logged `forward_ms`) | 32.15 (32.16) | | 49.65 (49.67) | |

With graphs: `forward_ms` 30.54 / 47.78, `step_ms` 34.35 / 53.59, drafter pass 3.43 / 5.41, host path under 1%. The
GPU waits on the host before a stretch for 17-30 µs on rank 0 and 24-148 µs on rank 1 (`round_clock_us`).

**The byte model (derived).**

- Expert bytes: `rank cache: 95.1 GiB of expert entries` a rank, over 40 x 385 + 3 x 129 = 15,787 experts, is 6.47 MB
  an expert a rank. A rank's expert is 3 x 5120 x 1152 = 17.7 M weights, so 2.9 bits a weight.
- Places a round: `expert_prune` logs 12.50 distinct routed experts a layer after the prune at 1 stream (15.48
  before) and 25.97 at 4 streams (31.30 before). Add the shared expert: 13.5 and 27.0 places a layer, 540 and 1,079 a
  round.
- 540 x 6.47 MB = 3.49 GB in 16.10 ms is **217 GB/s**. 1,079 x 6.47 MB = 6.98 GB in 31.45 ms is **222 GB/s**. Without
  the shared place: 201 and 214 GB/s. Either way ~29.5 µs a place.
- The bus: NVIDIA's figure for GB10 is 273 GB/s. Forum posts report 205-207 GB/s on a triad test on an Asus GX10 and
  ~178 GB/s of weight reads in a decode on a GB10 (single units, informal; taken from a search summary, the pages
  were not opened; links at the end).
- **The derived rate is above the measured one, and that needs an answer before anything is built on it.** Two
  explanations, not exclusive: a triad test writes as well as reads, so a read-only kernel can run faster; or the
  bytes a place are overcounted. 6.47 MB is the file's average, and experts differ: `weights.Experts` carries
  `gate_k2` / `up_k2` / `down_k2` for each expert with a 6-10 range (`k2_gu`, `k2_d`), so the experts a round picks
  need not be average-sized. W4 measures the picked experts' bytes instead of counting places.
- The dense linears point the same way but are not sized: attention costs 9.8 ms at 3.88 rows and 11.4 ms at 7.45
  (+0.44 ms a row), so it is a per-round constant. From `config.zig` a layer's EXL3 linears are ~68 M weights a rank
  (`wq_a` 5120x1280, `wq_b` 1280x16384, `wkv` 5120x512, `wo_a` 4 x 4096x1024, `wo_b` 4096x5120), about 1.0-1.4 GB a
  round at 3-4 bits. Their bit rate was not read.

**What the fit cannot say (do not build on it before W4).**

- Rows and distinct experts move together in both profiles: `moe` is 4.15 against 4.22 ms for each row of the round,
  and 1.19 against 1.17 ms for each place of a layer. The data cannot tell which one drives the cost.
- Bus-bound and decode-bound both scale with bytes. A rate this close to the bus's measured figures is evidence,
  not proof.
- This decides every caching idea. A decoded expert is ~5x the bytes of its trellis: a decoded-expert cache wins only
  if trellis decode is the limit and loses if the bus is.

**If the model holds, a round costs** about 12 ms of per-round constants (dense linears, head, shared expert,
gathers, mixes) plus ~4.9 ms a row, and the only levers are bytes per accepted token: fewer wasted rows, fewer places
a row (lossy), more accepted tokens a round, more streams sharing the constants.

## 2. Gaps in `to-benchmark.md`

Ordered by how much each could change a recorded result.

| # | Gap | Why it can change a result | Package |
| --- | --- | --- | --- |
| A1 | **`0114` is signed keep on a metric that is not speed, and a later run rules it out.** The matrix pass signed it on `engram_lookup_seconds` (-57% read time) with prefill at 2050 against 2122 (inside the drift band, wrong direction). The overnight pass (`runs/overnight/RESULTS.md`, blocks b07, b09) found a progressive regression that starts after ~370-470K uncached prefill tokens since the restart: decode -4% / -9% / -17% at 1 / 4 / 8 streams, code 8K prefill -39%, and both of that day's rank-pair breaks had the cache armed. The matrix block was too short to reach the onset | The sign-off is wrong as written, and production was restored with the cache on | W1, W2 |
| A2 | **The overnight verdicts are not in the matrix.** `SCALES=map`: +3.5 to +8.8% prefill in steady state, with the first request after a restart at -82% and the second at -41% (the matrix says -5% / -11% and "hold"). `LEAD`+`map`: +6-7% at 8K/32K, the largest prefill gain of either pass. `AIO=0`: +2-4% prefill (the matrix says neutral). `T25=0.05`: -12% at 8 streams | Three signs flip or sharpen. `map-warm` was the row meant to decide `map`; the overnight data says it would also remove the only cost | W2, W6 |
| A3 | **The base is lossy and unpriced.** Every base and every `IDENTICAL` gate runs with `TF_DS_T13_EXPERT_PRUNE=1` (`.env`). The prompt path never prunes (`prompt.zig:1467-1490` calls no prune), so a reply's tokens are computed with pruned experts and re-read unpruned as history. No row has the prune off, on `stats`, or with a cap, and `0120`'s score sets do not exist | The bit-exact engine's speed is unknown (derived: ~10-12% slower a round), and the quality paid for it is unknown | W3 |
| A4 | **Production flags are not written down.** `LEAD=1`, `ROWCACHE_MB=256`, `STRICT=1` were exported in a shell (overnight: "not in any file"); the 2026-11-15 arms restarted from `.env` alone, so the pair that produced the Baseten rows ran without them | "Production" names two different configurations across the result tables | W2 |
| A5 | **No measurement at the production load.** `PARALLEL=8`; live traffic had mean concurrency ~3.6 with prompts of 6.9K-209K, mostly cached. The matrix decodes at 1 and 4 streams, the round profile at 1 and 4 streams on ~20-token prompts. The overnight pass has 8-stream cells and they diverge from the 4-stream ones (T25: -7% at 4, -12% at 8) | Depth, calibration and prune arms price rows; row prices change with streams (section 4.4) | W4, W5 |
| A6 | **One traffic class for the draft-depth rows.** Bench prose drafts at 70% / 2.0 tokens a round; agent traffic at 88% / 2.35; the Tetris prompt at 80% / 4.8. The depth pins were scored on the first only | A depth that loses on prose can win on code; the verdict "pins lose" is one class's | W5 |
| A7 | **The corpus.** `p125` at 8K prefills ~10% faster than diverse text on every base block (overnight). The matrix's absolute prefill figures use the old corpus | Absolute figures are optimistic; deltas between arms may differ on diverse text (`code 8K` moved differently from `p125 8K` in several blocks) | W2 |
| A8 | **Drift and single blocks.** Bookends drifted -4.6% prefill in one pass; `LEAD` (+5-6%) and `AIO=0` sit near that band. No arm has the 3x alternation the protocol asks for | Small positives are unconfirmed | W6 |
| A9 | **`DEDUPE`'s metric counts gathers, not rows**, so "no effect" is partly "not visible". It is neutral on speed in both passes, and `DEDUPE`+`LEAD` breaks the pair 2/2 | Low value. The crash is the open part: it is in the path `LEAD` uses | todo.md (existing) |
| A10 | **`GRAPH_CAP` never reached its regime.** The bench keeps 64 graphs; production showed 128 kept and 64 destroyed in 80 minutes | Bounded: 64 recaptures in 80 minutes is negligible unless a capture costs seconds. Measure one capture's ms and close it | W4 |
| A11 | **TTFT in agent traffic is not decomposed.** The baseline says TTFT median 36 s against 2.1 s prefill; a warm 30K turn measures 1.23 s. Queue, restore, prefill and thinking are not separable on the `done` line (0.1 s resolution, no queue field) | The largest latency number in the repo has no attribution | W10 |
| A12 | **A 300K-token incident with no cause.** 16:00Z, 2026-11-13: 86.6 s TTFT for a 3.4K uncached tail, decode at 2.5 tok/s, then "no progress for 120 s" and exit 75. The row cache was armed and far past its onset | Fits A1's mechanism (a reader holding the cache's mutex through long probes); unverified | W1 |

## 3. Gaps in the Baseten-derived work

### 3.1 The four patches

| Item | State | Gap | Package |
| --- | --- | --- | --- |
| `0119` token cache | **+**: `encode_s` 0.108 -> 0.056 s on a resumed 95K turn | Complete enough. `render_s` is 0.002 s at 95K, so there is nothing to cache in the template render. 250K and 500K were never run; the saving scales with length and is small against a 0.7 s TTFT | none |
| `0120` scoring | built, host-tested | `tools/score_sets/` does not exist and no hardware smoke is recorded. It gates every lossy decision in this repo, including the one already in production (A3) | W3 |
| `0121` exact resume | fast (TTFT 1.31 -> 0.26 s on a 1-token append; 0.7 -> 0.3 s at 100K; `prompt_reuse` 88% -> 98%) but **not identical to cold** | Step 0 (`TF_DS_DEBUG_SPLIT` + `--logits-sha 1`) was never run. `to-benchmark.md` says turns 7-8 match cold; `runs/ultrafast/full/resume-report.md` marks turns 6-8 `NO`. Reconcile: once turn 6's reply differs, later prompts carry a different history. See 3.2 | W8 |
| `0122` speculative sampling | built; six blockers fixed in the patch | No distribution gate (`samp_test.py`, 5,000 seeds) and no speed row. `uf_probe.py sampled` now provides the sampled load the todo said was missing. Its bar is `TF_DS_T26_KEYED_DRAFT`, which measured -2 to -4% on sampled prose and about +2.5% on sampled code | todo.md (existing) |

### 3.2 Why exact resume is not identical: a candidate cause (unverified)

Three kernel choices in a prompt pass depend on the chunk's row count, and the file that holds two of them says so
("The two paths round differently, so a token's bits can differ between them", `prompt_rows.zig`):

| Choice | Rows | Code |
| --- | --- | --- |
| router logits | up to 16: `rowmmGate` chunk sums; above: one cuBLAS GEMM | `prompt.zig:1475-1484`, `tri.zig:10` |
| routed experts | under 64: the decode window (`exl3_experts.decode`); 64 and above: the prompt path (`exl3_experts.prompt`) | `prompt.zig:1488-1489`, `prompt_rows.zig` |
| dense linears | up to 128: the grouped EXL3 linear; above: `rot_in` + `unpack` + Triton GEMM | `prompt.zig:1005`, `exl3_prefill.zig:103-107` |

The failing turn (turn 6, prompt 8,281, resumed at 8,228):

- cold pass: the last chunk is `[8192, 8281)`, 89 rows: cuBLAS gate, prompt-path experts, grouped linears;
- warm pass: the chunk is `[8228, 8281)`, 53 rows: cuBLAS gate at another shape, **decode-window experts**, grouped
  linears;
- the decoder replay differs too: cold runs rows `[8153, 8192)` as the tail of the previous chunk (39 rows, window
  path) and `[8192, 8281)` separately; warm prepends the tail and runs `[8153, 8281)` as one 128-row chunk.

So the same rows take different kernels warm and cold. Grid resume never does this: its chunks are the cold pass's
chunks. This also restates review finding 18's "identity-bound" as an engineering constraint rather than a law: the
contract is coupled to chunk shape, not to state.

### 3.3 The decode-step items (A-E) against the profile

Ceilings from section 1 (1 stream / 4 streams, share of `step_ms`). `0123` and `0124` are named in
`docs/device-accept.md`; `0125` and `0126` are inferred from the commit order of `decode-step-structure`.

| Item | What it could remove | Ceiling | Verdict |
| --- | --- | --- | --- |
| A. One graph a step, fewer host hops | `send` + `logits` + `sample` + `absorb`: 0.4 ms | ~1% | Keep `0123` (the served round profile: it is the instrument for W4). Do not build the one-frame round |
| B. Compressor and indexer on a side stream | part of `a_idx` (0.57 ms) and the compressor's launches inside `a_q` | under 2% | Measure `0125` once; expect noise |
| C. PDL in the hand-written kernels | launch gaps of the repo's own kernels. Graphs already save 1.6 of 32.2 ms over eager launches | 1-3%, unsized | One arm of `0126` with the nsys gap script, then stop |
| D. Fuse the ratio-2 compressor chain | ~20 launches a round | under 1% | Only if C shows launch gaps are real |
| E. Engram without a host landing | the GPU's wait before a stretch: 17-148 µs | under 0.5% | Close. The reads are already hidden |
| Second pass: absorb reorder | `absorb_ms` 0.26-0.29 | under 1% | Close |
| Second pass: greedy rounds exchange argmax | part of `head` (1.1 ms, most of it the head's own weights) | under 1% | Close |
| Second pass: static expert grid, rows x 7 places | unknown: depends on what an empty place costs in the kit's kernel | unsized | Folded into W4 (the `PLACE_BOUND` arm answers it) |
| Second pass: start time "is one serial read" | 88 s on the head | large, but see section 0 item 7 | W7 first; parallel readers only if the head's disk has headroom |
| Second pass: whole reply text reprocessed every round | host CPU and memory on 32K-token replies; not in any profile (the profile has no HTTP or text path) | unsized | W10 |

The profile that produced "the host path is ~1%" is the batch driver: no HTTP, no SSE, no text path, 1-4 streams,
~20-token prompts. It does not cover what the last row names.

## 4. The MoE step

### 4.1 What runs

- **Decode window (under 64 rows), a layer:** the gate's chunk sums (`rowmmGate`), `route`, the prune kernel, then
  three launches (`exl3_experts.zig:357-431`): `decode_prep` (groups the picks into places, rotates gate/up's input
  rows), `grouped_cp` for gate and up with their epilogue, `grouped_cp` for down with the rows' combine, the last two
  as programmatic dependents. Grid x is the place count: `min(rows x 7, 385)`, or the prune's bound under
  `TF_DS_T15_PLACE_BOUND`. Rows that pick the same expert share one read of it (16-row member tiles).
- **Prompt chunk (64 rows and more), a layer:** eight launches (`exl3_experts.zig:194-272`): count, place, work
  list, `rot_in`, `grouped_mma` for gate and up over (place, 64-row tile) pairs, the epilogue, `grouped_rows` for down
  over (place, 32-row tile) pairs, the combine. Each pair is one pass over that expert's trellis. The shared-dense
  GEMM path is off (`skip_e = -1`).
- **Prefetch that exists:** a paced L2 prefetch on its own stream (`round.zig:685`), 8 MiB behind `wo_b` (the FFN
  mix's weights, the router, the shared expert's three trellises) and 16 MiB behind the experts (the next layer's
  mix and input linears). A layer's routed experts are 80-90 MB; a prefetch cannot hold them.

### 4.2 "MoEs in parallel at TP=2": the direct answer

They already are. Each rank computes every picked expert on its half of the intermediate width (I = 1152 of 2304)
at the same time, and one exchange (`gather_m`) sums the two partials. Nothing in the MoE waits on the other rank.

- **Splitting by expert instead of by width** (rank 0 runs experts a, b, c; rank 1 runs d, e, f) reads the same
  bytes a rank and balances worse: with 6 picks split at random the busier rank averages ~3.9 experts' worth, not 3.
  It also needs every expert whole on both ranks (2x memory) or a static partition. Ruled out.
- **Overlapping the MoE with other work on the same GPU** helps only if the two contend for different resources. If
  W4 says the bus is the limit, the dense linears are on the same bus and there is nothing to overlap. If W4 says
  trellis decode is the limit, the picture changes: see 4.3, M5.
- **Across layers** there is no parallelism to take: layer l + 1's router needs layer l's output.

### 4.3 Candidates, with ceilings

| # | Candidate | Bits | Where | Ceiling (grade) | Step 0 that can kill it |
| --- | --- | --- | --- | --- | --- |
| M1 | **Price rows correctly.** Calibrate `TF_DS_ROUND_MS` from measured forward ms by rows, then feed the measured costs to `windows.allocate` instead of the static table | sequential | config, then repo | 1-2% at 1 stream; several % at 8 streams, where the table's price for a row is ~35% under the fit's (derived, two points) | W4's forward-by-rows table; W5 |
| M2 | **A row-aware expert cap.** `TF_DS_T13_EXPERT_CAP` is a cap on the whole round: 12 experts is mild at 1 stream (12.5 distinct today) and halves the places at 4 streams (26). Quality would fall with load. Make the budget scale with rows, and choose the kept picks to maximise kept routing weight under the budget | LOSSY | repo (the prune kernel is this repo's) | large: at 4 streams a 13-place round is ~-16 ms of 53.6 (derived); upstream reported 99 -> 112 tok/s on code for `CAP=12,1` | W3's score sets first. No quality number, no arm |
| M3 | **Draft triage: stop computing rows that are already lost.** A draft row whose predecessor will pick another token is wasted work, and dropping it mid-forward cannot change a reply (a dropped row is an unverified draft). A device kernel at a stretch cut marks rows dead; dead rows route nothing (`pick = -1` in every slot; `decode_prep` already combines rows that route no slot). Both ranks compute the mask from replicated inputs, as they already do the routing | sequential | repo + a trained probe | wasted rows are 0.78 a round (834 of 3,085 drafted rows), ~3.8 ms or 11% of a step; a perfect probe at the layer-14 cut saves 65% of their MoE: **~6%**, less with a real probe (derived) | An offline probe study: can a linear probe at layer 13 or 19 predict "the next draft is rejected" at 95% precision with 50% recall? If not, drop it |
| M4 | **A state for the identical prompt.** A retry, a regenerate or `n > 1` resends a prompt the cache holds in full, and still replays 128 decoder rows and the grid slack. Keep the pass's end state after the decoder (rings, the drafter's rings, the last row's logits) and resume there | identical (the state is a cold pass's own) | repo | the whole warm TTFT of such a turn (0.7 s at 100K) | The `tail_window` miss-cause counter (`0116`) on production traffic. Under ~2% of requests: drop |
| M5 | **A decoded cache for the hot experts, and routed-expert prefetch.** Only if W4 finds trellis decode, not the bus, is the limit. Then: keep the shared expert (in every row, every round) and the most-picked routed experts decoded, and prefetch by a per-token routing prior | identical | kit (a decoded-weights kernel does not exist in the kit) | unknown; zero or negative if bus-bound | W4's pure-read probe |
| M6 | **The prompt path's tile fill.** A 2,048-row chunk gives a routed expert ~32 rows of a 64-row gate/up tile (12,288 picks over 384 experts), and the 128-row decoder replay gives most experts 2-3. If a tile's cost is its trellis pass, not its rows, the path is paying for empty tiles | identical at fixed shapes | kit | unknown: no prefill phase profile exists at real chunk sizes | W4's prefill profile at 256 / 512 / 1,024 / 2,048 rows. `moe` flat in rows: tile passes dominate |
| M7 | **Batch several streams' warm tails into one prompt pass.** Agent traffic is short appends at concurrency 3.6-8; each pass pays the decoder replay (~0.15-0.2 s, derived) and its own expert passes. One pass over k streams' rows would share them | identical only if W8 shows row-invariance | repo, large | unsized | W8's step 0 and W4's prefill profile. Design note only until both exist |

Why M3 is safe, since it is the unusual one: a chain row is used only if every draft before it was accepted, a
row's exactness depends only on the rows before it, and rejected rows are already overwritten in place. A wrong
"dead" call costs one token of speculation; a wrong "alive" call costs what it costs today. The host must treat a
dead row as rejected, so the mask rides the picks' read-back.

### 4.4 The allocator's prices (M1), in numbers

- The window allocator maximises expected tokens over `cost(rows) + overhead` (`core/lanes/allocate.zig:9-48`,
  `core/lanes/windows.zig:159-181`). `cost` is the static table `draft.round_ms` (`draft.zig:779`) unless
  `TF_DS_ROUND_MS` is set (`native.zig:273`, never set here). `overhead` is learned as round time minus the table's
  forward, floored at zero (`depth.zig:262-272`), so a table that reads high leaves it at zero.
- Measured with graphs: 30.54 ms at 3.88 rows, 47.78 ms at 7.38 rows. The line through them is 11.4 + 4.93 x rows.
  The table gives 44.3 and 56.9 at those rows: 45% and 19% high.
- Shape, not scale, is what moves the choice. The table's intercept is 30 ms where the fit says ~16 for one row, and
  its slope past five rows is 2.6-3.6 ms where the fit says 4.9. Both make an extra row look cheaper than it is,
  most of all in wide rounds: at 16 rows the marginal cost over the round's cost is 3.1 / 84.4 in the table and
  4.9 / 90.3 in the fit, so the landing chance the last row needs is about 45% higher than the table asks.
- Two points from different stream counts are not a calibration. They justify measuring one.

### 4.5 The sweep across other fields

What was looked at outside the usual LLM-serving toolbox, and where each idea ended up.

| Field | Idea | Verdict |
| --- | --- | --- |
| Computer architecture: the roofline model | Treat a round as bytes over a bus rather than launches or flops | The frame of this review; W4 tests it |
| Computer architecture: branch-target prediction | Predict a row's experts from its token and prefetch them | M5: only if decode, not the bus, is the limit; a layer's experts are 5x the prefetch budget |
| Computer architecture: early branch resolution | Resolve a draft's fate before the last layer and stop paying for it | M3 |
| Queueing and marginal analysis | Price a row at its marginal cost for the round's width and stream count | M1 / W5 |
| Combinatorial optimisation: knapsack, coverage | Keep the most routing weight under a budget of places, or of bytes | M2 / W9 |
| Information theory: rate and distortion | Put a number on what a dropped expert costs, in nats a token | W3 |
| Information theory: entropy coding | Recompress the trellis losslessly so the bus carries fewer bytes | One `zstd` test (section 5); expected to close |
| Numerical analysis: reproducible summation | Make a row's result depend on the row alone, never on how many rows ride along | W8; it is also what M7 needs |
| Data structures: open addressing | Deletions without rehashing degrade every miss | W1, W10 |
| Data structures: approximate membership | A Bloom or cuckoo filter in front of the row cache's `has()`, so a miss takes no lock | After W1, and only if the lock still shows |
| Caching theory | A least-recently-used cache under cyclic access never hits; the full hit is the one state not kept | Unpacked dense weights ruled out; M4 |
| Parallel computing: wavefront schedules | Run half the rows a half-layer behind so unlike phases overlap | Ruled out (section 7) |
| Distributed systems: partitioning | Split the experts, not their width, between the ranks | Ruled out (4.2) |
| Distributed systems: replicated state machines | Both ranks derive the same decision from the same inputs, with no exchange | Already how routing agrees; M3's mask rides on it |
| Statistics and interpretability: linear probes, the tuned lens | Read the model's coming pick from a middle layer | M3's step 0 |
| Linear algebra: shared basis plus low-rank or sparse deltas | Store experts as a common part and small differences, so a round reads less | A new pack and a quality study; not planned, recorded so it is not rediscovered |
| Graph theory: co-activation clustering | Lay experts out on disk by which fire together | Ruled out unless W4 shows a penalty for scattered ranges |
| Compilers: loop fusion and interchange | Fuse the MoE's launches; go row-major instead of layer-major | Ruled out: the cost is a place, not a launch; row-major re-reads the constants |
| Control theory: online estimation | Let the allocator use the round costs it already measures | W5, second half |
| Batch scheduling | One prompt pass for several streams' short tails | M7, after W8 |
| Coupling and rejection sampling | Accept a sampled draft with probability min(1, p/q) | Built (`0122`), unmeasured |

## 5. What can be cached, and for how long

| Lifetime | Item | State | Note |
| --- | --- | --- | --- |
| a request | tokenised pieces | `0119`, + | done |
| a request | template render | nothing to win | `render_s` 0.002 s at 95K |
| a request | the reply's text scan (stops, thinking split) | whole text every round | incremental scan; W10 |
| a request | the router's widened gate, a layer a chunk | recomputed | 7.9 MB a layer: negligible |
| a request | unpacked dense linears in prefill (`exl3_prefill.zig:103-107` unpacks on every call) | recomputed | ~6 GB of traffic a chunk if bus-priced, under ~3% (derived: ~68 M weights a layer, 136 MB unpacked, written once and read once, 20 encoder layers). A cache of them is 2.7 GB a rank for the encoder layers against ~3.6 GiB of head room, and an LRU over 20 layers visited in a cycle hits nothing. Not worth it unless W4 shows unpack is compute-bound |
| a process | Engram rows (`0114`) | broken as built (W1) | after the fix its value is still unproven: the AIO ring already hides the reads |
| a process | Engram scales in the page cache (`SCALES=map`) | + in steady state | warm it at open (W6) |
| a process | CUDA graphs (`GRAPH_CAP`) | fine | 64 recaptures in 80 min |
| a process | prompt states, shared cuts, edit marks | `0110`-`0116` | targeted probes still owed (to-benchmark.md) |
| a process | the end state of an identical prompt | missing | M4 |
| a process | measured round costs by rows | learned per depth, not used by the window allocator | M1 |
| disk | prompt states (spill tier) | exists | `SPILL_MIN_TOKENS=2048` recovers short sessions (overnight K) |
| disk | rank 0's weight file on local disk | on `/nfs1` today | W7 |
| disk | a lossless recompression of the trellis | untested | one `zstd -19` of a 1 GB slice of the rank cache settles it; under 3% saving: close |

## 6. Work packages

Rules, inherited from `todo.md` and unchanged:

- Engine work on a branch of `../tfref-next` from `series-0119`. The `decode-step-structure` branch there holds
  `0123`-`0126` and uncommitted work: do not rebase, stash or reset it. Number new patches from `0130`.
- Do not add files to `zig/patches/` in this repo and do not commit here; hand patches over from a scratch folder
  (`git format-patch`). Host tests with `zig build test`; type-check CUDA paths with
  `zig build install native -Dfatbins=<dir of dummy *.fatbin>`.
- The pair is the operator's. Each pair step below is to be delivered as a script under `runs/<name>/` with its
  restore step, not run.
- Every switch off by default; a switch both ranks must agree on rides the `switches` frame; a new `Model` field goes
  in `Model.open`'s init list.

Order: W1-W3 are correctness and bookkeeping and gate the rest. W4 is the instrument for W5-W9.

### W1. Row cache: fix the map, retest past the onset

- **Bits** identical. **Where** repo. **Switch** none new (`TF_DS_ENGRAM_ROWCACHE_MB`).
- **Evidence.** `engram_io.zig:281-310`: `put` evicts with `rc.map.remove` and inserts with `rc.map.put` on a
  `std.AutoHashMapUnmanaged`. In Zig 0.17's `std/hash_map.zig`, `removeByIndex` (`:1229`) leaves a tombstone and
  gives the slot back to `available`, so the map never grows or rehashes at a steady count, and `getIndex` (`:968`)
  probes past tombstones until a free slot or the whole table. Once the cache is full every put leaves a tombstone;
  misses (`get`, `has`, and `put`'s own `contains`) drift toward a full-table probe under the mutex. The token cache
  has the fix (`core/tokenizer/cache.zig:79`, `0119`'s review note: "tombstones made every miss probe the table"). It
  is the only `rehash` call in the engine.
- **A prediction the logs already bear on.** The slowdown should start when the cache first fills, so its onset
  should scale with the budget. At 256 MB it came after ~370-470K uncached tokens. At 64 MB (overnight b23) the first
  cold 100K ran at 702 tok/s against ~1,950 and the pair broke 9 s later: a quarter of the budget, the wall about
  four times sooner. One event, so it is support, not proof. **Kill test:** if a run at two budgets shows the onset
  does not move with the budget, this is not the cause.
- **Change.** Use `std.AutoArrayHashMapUnmanaged(Key, *Node)` with `swapRemove` (no tombstones, no periodic pause
  under the mutex). If that does not fit, count removals and `rehash` when they reach a quarter of the capacity.
- **Host tests.** Fill to the budget, then run 20x the capacity in put-and-evict cycles with interleaved missing
  keys. Assert on structure, not wall time (a timing ratio will flake): the map's never-used slots stay above a
  floor after the churn, or a probe counter compiled in for tests stays within 2x of the fresh cache's a miss. The
  test must fail on the current code: run it there first and record what it reads.
- **Pair gate (script).** Before the fix, the kill test above at 64 and 256 MB (it costs one block each and tells
  whether the fix is aimed at the right thing). After it, the overnight b07 cells (`tools/bench_div.py`,
  `scripts/bench_overnight.py`) driven past 500K and past 1.5M uncached prefill tokens since the restart, base
  blocks either side: dec1 / dec4 / dec8 and
  code 8K inside the band; a cold 100K after `ROWCACHE_MB=64` leaves the pair up; `validate.py --compare` identical.
- **Stop rule.** Regression still there after the fix: the cache is ruled out for good; its read-time win never
  moved tok/s.
- **Meanwhile.** No row cache in production; A1's sign-off reads "withdrawn pending W1".

### W2. One set of results, and production flags in a file

- **Where** config and docs.
- Fold `runs/overnight/RESULTS.md`'s verdict table into `to-benchmark.md` (rows A1, A2, A7 above), with the block
  ids as evidence. Hand the edit over as a diff; tracking `runs/overnight/` is the operator's step. That folder and
  `runs/ultrafast/prof/` hold whole container environments (`production.rank*.env`, `*.inspect.json`, `*.cmd`): check
  them for keys and tokens before anything there is tracked or shared.
- Make `tools/bench_div.py`'s diverse corpus the matrix's prefill corpus; keep `p125` as a labelled comparison.
- Put the intended production switches in the head's `scripts/local.sh` (it does not exist on the head) or `.env`.
  Proposal, the operator's call: `TF_DS_ENGRAM_LEAD=1`, `TF_DS_ENGRAM_STRICT=1`, no row cache, `SCALES=io` until W6.
- State on every result table which configuration its "production" arm was.

### W3. A quality number for the lossy base

- **Where** tools, then pair. Needs `0120`'s hardware smoke first.
- Build `tools/score_sets/` with provenance: (a) self-consistency: greedy replies recorded with
  `TF_DS_T13_EXPERT_PRUNE` **unset** on the `bench_div` prose and code prompts and a tool-call set, 60-100 items of
  256-512 tokens; (b) held-out human text split into prompt and continuation.
- Arms: prune unset, `stats`, `1`, and `1` with `TF_DS_T13_EXPERT_CAP=12,1`. For each: `score.py` NLL mean ± se,
  argmax match, paired deltas against unset, by class; and the bench block for speed, paired decode at 1 / 4 / 8.
- **Gates.** Unset scores its own replies at 100% argmax match and repeats bit for bit, or the tool is wrong.
- **Decision this buys.** Whether production keeps the prune, and the price list M2 needs.

### W4. The MoE step 0s (one instrument run)

- **Where** repo (profile-only code) and a pair script. Attribution only: it syncs every round.
- Extend `runs/ultrafast/uf_prof.sh` / `uf_profreq.py`:
  0. **Fixed rows in the batch driver.** `TF_DS_FIX_K` and the depth pins are read by the served process only
     (`native.zig:302`, `:308`); `lanes_main.zig` reads neither. Add the same two reads there (profile-only), or
     take item 2 from the served process with `0123`'s `TF_DS_ROUND_PROFILE`.
  1. **Places at fixed rows.** Rows pinned at 3 drafts, crossed with prune unset / `stats` / `1` and
     `TF_DS_T13_EXPERT_CAP` 8,1 / 12,1 / 16,1, each with `TF_DS_T15_PLACE_BOUND` off and on, `--profile 2 --graphs 0`.
     Profile-only: after each layer's experts, read back the decode scratch's `ids` and `count` (the round already
     syncs) and add each place's trellis bytes (`k x n x k2 / 16` for gate, up and down, from the k2 tables) to a
     per-round counter printed beside `expert_prune`'s line. Regress `moe` ms on rows and on **those bytes**, not on
     place counts. This separates the two, prices an empty place, and corrects the 6.47 MB average.
  2. **Rows.** 0-5 drafts pinned, alone, graphs on: `forward_ms` by rows, the table for W5.
  3. **8 streams.** The `g` and `e` arms with an `r8` request file.
  4. **Real prompts.** Requests of 2K, 8K and 32K natural tokens, so `prefill_phases_ms` is taken at 2,048-row
     chunks, plus single chunks of 256 / 512 / 1,024 rows (M6) and a 1-token append onto an 8K history (the decoder
     replay's cost).
  5. **One graph capture's ms**, from the first round at a new shape (A10).
- Add, profile-only: phase marks in the prompt path matching the round's (`a_in`, `a_q`, `a_idx`, `a_core`,
  `a_woa`, `m_rt`), a split of the dense path into `rot_in` / `unpack` / GEMM, and of the prompt experts into
  gate-up / down.
- **A pure-read probe.** Kernels in `zig/kernels/cuda/probe.cu` (it has none that read memory at scale) that sum
  the words of given address ranges, in two forms: a plain non-caching global load, and `cp.async` copies into
  shared memory through a small ring (what `grouped_cp`'s doc comment describes: 4 warps, 3-stage rings). The grid
  is shaped like the decode launch (places x I / 128 blocks). A small main feeds them one layer's expert ranges from
  `weights.Experts` (`gate_ptr`, `up_ptr`, `down_ptr`). Report GB/s for: 13 distinct experts; one expert 13
  times; ranges in address order against shuffled; with and without the L2 prefetch stream running.
- **Reading.** Compare the probe's GB/s with the MoE's rate on its measured bytes (item 1). About equal: the MoE is
  bus-bound, M5 is dead and the byte model stands. Probe well above it (1.5x or more): trellis decode is the limit,
  M5 opens and section 4.2's overlap question reopens.

### W5. The allocator's prices

- **Bits** sequential. **Where** config, then repo.
- From W4.2 set `TF_DS_ROUND_MS` to the measured forward ms for 1-16 rows (linear past the last measured row).
- Arms against unset: prose, code and thinking-on agent prompts at 1 / 4 / 8 streams; record accept, tokens a
  round, rows a round, tok/s. Gate: `exact.py`, `validate.py` including the concurrent check.
- Keep if 8 streams gains 2% or more with no loss at 1. Then the patch: let `windows.allocate` read the costs the
  rule already learns (`depth.zig` `observeCost`) instead of the static table.

### W6. `map-warm`, then the confirmation runs

- **Bits** identical. **Where** repo. **Switch** `TF_DS_ENGRAM_SCALES=map-warm`.
- After the mapping is made, a background thread walks it with `madvise(MADV_WILLNEED)` in pieces (3.07 GB a rank,
  clean page cache: reclaimable, counted as available by the guard). The load does not wait for it.
- Pair script: first and second request after a restart against `map` and `io`; steady state; `MemAvailable` low;
  then `LEAD=1` + `map-warm` against base with the protocol's 3x alternation on the diverse corpus. Do not combine
  with `AIO=0` (the matrix measured `AIO=0` + `map` at -44%).

### W7. Rank 0's weight file

- **Where** config and scripts.
- Check on the head: `findmnt -T /nfs1`. The profile's containers show rank 0 reading `/nfs1/.../rank-cache`
  (`runs/ultrafast/prof/rank0.inspect.json`) and rank 1 a local bind; `.env` has `NFS_SERVER=10.50.0.1`,
  `NFS_PATH=/nvme1`.
- `scripts/config.sh` describes `NFS_SERVER` as the head's own address for the worker's mount. If `/nfs1` is the
  head's own `/nvme1` reached back through NFS, the fix is a path, not a copy: give rank 0 the local path of the
  same file.
- If it is a mount from another machine: add `HEAD_CACHE_DIR` beside `WORKER_CACHE_DIR` (`scripts/config.sh`,
  `prepare.sh`, `start.sh`, their tests), a local copy of rank 0's 98.9 GiB file. Expected: the 88 s read falls toward the
  worker's 20 s, the 100 s load toward ~35 s. If it is local disk: measure the device's sequential read; only then
  consider the plan's parallel readers.

### W8. Exact resume that equals a cold pass

- **Bits** to be shown identical. **Where** repo. **Switch** `TF_DS_EXACT_RESUME` (existing).
- **Step 0 (the plan's, sharpened).** With `TF_DS_DEBUG_SPLIT=p` and `--logits-sha 1`, on 3K / 10K / 70K prompts,
  choose `p` so that the cell's two parts fall in chosen classes:
  both parts above 128 rows; second part 65-128; second part 17-63; second part 16 or fewer; and cells whose whole
  size is in each class. Prediction if 3.2 is right: equal shas whenever each part's class equals the unsplit cell's
  class, different otherwise. Also run turn 6 with turn 5's cold reply forced as history, to settle turns 7-8.
- **If the prediction holds, phase 1:** resume off-grid only when every row-count choice of the warm chunk and of
  its decoder replay equals the cold pass's (an O(1) check on `len`, the resume point and the grid); otherwise use
  the grid state as today. Report the share of turns that qualify on agent traffic.
- **Phase 2, cold-shape emulation:** choose each path by the cold cell's row count and pad the launch with inert
  rows (experts: rows whose picks are -1; dense: zero rows; the gate GEMM at the cold M), and split the replay at
  grid boundaries as a cold pass does. Each padded path needs its own sha check.
- **Gate.** `resume_exact.py` against true cold rows (`--prompt-cache-gib 0`) identical on every turn in 3 passes;
  `exact.py`; `validate.py` (the image case is still refused: fix or keep refusing).
- **If the prediction fails:** the difference is inside a path; record which and stop. `0121` stays LOSSY and off.
- **Outcome (2026-11-16): the prediction failed; W8 is closed. Step 0 found bits move for EVERY off-grid cut
  measured** (`DEBUG_SPLIT` 8228/9300/9380/9440/9480; NLLs move both directions at every second-part size 16-2012;
  t3k bit-stable; turn 6's split reproduced the warm-resume sha `96d0fd7dcd8e` exactly). `--logits-sha 1` prints
  nothing on this build; teacher-forced NLL was the instrument. Phase 1 is vacuous (an off-grid keep whose shapes
  equal the cold pass's is grid resume); phase 2 cannot fix bits already baked into the restored parent-pass state.
  `0121` stays opt-in `TF_DS_EXACT_RESUME=1`, never default; grid resume is production. See `decisions.md` 2026-11-16;
  artifacts `runs/matrix/step0-cold-shape/`.

### W9. The row-aware cap (M2)

- **Bits** LOSSY. **Where** repo. After W3 only.
- `TF_DS_T13_EXPERT_CAP=rows:<a>,<b>[,floor]`: a budget of `a + b x rows` places for the round. Selection: keep each
  row's `floor` heaviest picks, then add picks in order of routing weight until the budget (the greedy for a
  coverage objective). Extend `tf_ds_expert_prune_kernel`, its host reference in `expert_prune.zig` and the rule
  fingerprint the ranks compare.
- A second form budgets bytes instead of places (`bytes:<MB a row>`): the cost is bytes and experts differ in size
  (their k2), so a pick is ranked by routing weight per byte of its expert's trellis. Only worth building if W4
  shows the picked experts' sizes spread by more than ~15%.
- Score at equal bytes against p85k: ΔNLL by class at 1 / 4 / 8 streams. The point of the design is that ΔNLL
  does not grow with streams.

### W10. Smaller ones

- **TTFT decomposition.** `queue_s` and `restore_s` on the `done` line after the existing fields, and 3 decimals for
  `ttft_s` / `prefill_s` (A11; the overnight J item could not separate them).
- **Reply text scan.** Check the allocator `splitThinking` is given each round (`server/chat.zig:423-428`,
  `server/reply_text.zig:472-487`); if it is the generation's arena, memory grows until the reply ends. Hold back a
  suffix instead of rescanning; gate with the server's golden tests.
- **Other maps with churn.** `spill_tier.zig`'s `points` and `jobs`, `engram_aio.zig`'s `batches`, `lanes.zig`'s
  `streams` remove from std hash maps and never rehash. The small ones are bounded by their capacity; `points` grows
  with the disk tier. A helper in `core/` that counts removals and rehashes, with a test, then apply it.
- **M3's probe study and M4's counter read** are step 0s, listed in 4.3. Neither is a patch yet.

## 7. Ruled out, and why

| Idea | Why not |
| --- | --- |
| Expert-partitioned TP (each rank runs half the picked experts whole) | Same bytes a rank, worse balance, 2x memory or a static partition (4.2) |
| Running half the rows a half-layer behind the other half, so one half's MoE overlaps the other's attention | Each half re-reads the dense linears (a per-round constant, ~30% of a round), and if the bus is the limit both halves want it |
| Approximating the tail rows (fewer experts, lower precision) and re-checking only matches | A row is exact only if every row before it is exact; an approximate row cannot be accepted and an exact recompute costs the row again |
| Skipping `up` and `down` for intermediate units whose gate is ~0 | The trellis stores weights in a sign-Hadamard rotated basis (`suh` / `svh`); a unit is not a row of what is stored |
| Fusing gate, route and experts into one launch (Baseten's `moe_fused`) | Launch count is not the cost: ~29.5 µs a place whatever the launch count; the kernels are the kit's |
| A full cache of unpacked dense weights | 2.7 GB a rank for the encoder layers (5.4 GB for all 40) against ~3.6 GiB of head room; cyclic access defeats an LRU; ceiling under ~3% |
| Prefetching routed experts by a routing prior | A layer's routed experts are 80-90 MB against a 16 MiB prefetch; no gain on a saturated bus. Reopens only under M5 |
| Reordering experts on disk by co-activation | Does not change bytes read; only matters if W4 shows a penalty for shuffled ranges |
| A larger draft block, deeper pins | Measured: pins lose on prose (to-benchmark.md), T25 loses at 8 streams |
| Render cache, gate-widen cache, Engram landing changes | Measured or derived negligible (sections 3, 5) |

## 8. Not verified

- Whether the bus or the trellis decode limits the MoE (W4), and the dense linears' bit rate.
- That W1's bug is the overnight regression. The code shows the bug; the link to the symptom is inference.
- Whether `/nfs1` on the head is a network mount (W7).
- 3.2's cause for `0121`: read from the thresholds, never run.
- The kit kernels' internals: no source here.
- Anything at 8 streams on the round profile, and any prefill phase split at real chunk sizes.
- M3's premise, that a mid-depth probe can see a rejection coming.

## Sources

- GB10 memory bandwidth, spec and community measurements:
  [NVIDIA forum: GB10 measured, with a tool](https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618),
  [NVIDIA forum: GB10 hardware baseline](https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851),
  [NVIDIA forum: DDR bandwidth lower than the claimed 273 GB/s](https://forums.developer.nvidia.com/t/the-ddr-bandwidth-is-significantly-lower-than-the-claimed-273gb-s/363238),
  [The Register on GB10](https://www.theregister.com/2025/08/27/nvidia_blackwell_gb10/).
- Everything else: `runs/ultrafast/prof/`, `runs/ultrafast/full/`, `runs/overnight/RESULTS.md`, `runs/matrix/`,
  `to-benchmark.md`, `docs/baseten-ultrafast-plan.md`, and the engine checkout named at the top.
