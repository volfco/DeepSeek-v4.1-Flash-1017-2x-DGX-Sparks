# Engine benchmark matrix

The dated results and current priorities below supersede the original experiment designs. The [November 18 follow-up](runs/engine-followup/RESULTS.md) covers scheduler policies, held-out pruning quality, concurrent deep decode, served prefetch, KV owner mode and the 500K tokenizer cache. The [November 17 review round](runs/review-round/RESULTS.md) supplies the earlier controls. The [runtime configuration record](best-runtime.md) tracks the incumbent and each challenger. For the deployed image and production switches use [continue.md](continue.md).

**Final pinned-image result.** For mixed short decode and persistent chat, use the deployed recipe with keyed drafts and the 32 MiB tokenizer cache. At 3K × 4 the cache alone stayed within 0.2% of deployed throughput. For cold-prefill-heavy traffic, the screened eight-switch bundle with unkeyed drafts delivered 2,070–2,072 input tok/s at 32K (+5.1–5.5%) and 2,021 at 100K (+3.9–4.3%). Four-stream warm output fell 5.9–7.6% at 32K and 5.5–11.6% at 100K. All 24 full-panel outputs matched both deployed controls. [Full crossover](runs/overnight-opt/RESULTS.md#full-32k-and-100k-confirmation)


The [overnight GB10 stage report](runs/overnight-opt/RESULTS.md) records paired missing-snapshot pressure, the graph-on finer-bucket crossover, per-board bandwidth calibration, per-stage read pressure and two long prefill captures.

For single-switch attribution, change one switch and hold image, context, pool, parallelism, round rows, graphs, fill settings, spill, KV sharding and prompt policy fixed. A predefined bundle such as `best` can also be the treatment; a bundle comparison measures only its combined effect. For sampled decode, use an identical-input replay to control output trajectory and confirm actual cache cuts.

## Current evaluation order

Use the [runtime record](best-runtime.md) for the general configuration and the [overnight report](runs/overnight-opt/RESULTS.md) for measured hardware limits and experiment outcomes.

| Decision | Evidence and next criterion |
| --- | --- |
| **Built, never run: fused-path follow-up 0200-0212** (no new switch - the existing `TF_DS_IDX_FUSED`, `TF_DS_FUSED_COMPRESS2_NORM`, `TF_DS_FUSED_ROPE_STORE`, `TF_DS_FUSED_ROUTE`; only `TF_DS_CHECK_INVALID=1` is new, and it is a diagnostic) | Thirteen patches from `todo.md`'s 0195+ review, written 2026-11-21 and not yet on a GPU: the dead fused-path scratch stores (229 KiB of stores at 0195's seven-row shape), the prompt indexer query on 0198's kernel, the `L == 32` norm barrier, a reader for `ch.invalid`, the recorded and asserted warp-alignment invariant, the 128-wide decode-key gate, the 0195 key/tile-maxima fold that drops `_score_keys` above 16,384 columns, the single-exponential finite pair softmax, the shared key-high-word cache, the rotation folded into the packed-column key store, the norm's held pair results, the prune prefix early-out and the wider prompt packing. Locally: `zig build test` on every suite and a stub-header `clang++` syntax check of the three hand-written `.cu` files in the work tree and in a tree reproduced from `TF_REF` + 0001-0212. **Gate before timing:** 39 exact cases, the long greedy hashes, PDL both ways; 0206/0208-0210 also owe the standalone fused-versus-unfused byte comparison (`runs/fused-kernel-audit-20261120/probe.py` predates the argument removals), 0207/0211 the same against the unfused path, and 0202/0210/0212 their SASS claims (barrier count; no spills; two-warp byte parity). Time one switch at a time with graphs off for attribution and on for a serving claim, since every patch rides a switch production leaves unset. [Rows](zig/patches/README.md) · [Decision](decisions.md) |
| **Deployed provisionally November 19; full testing owed at the next full benchmark round: the Engram lead reader** (`TF_DS_ENGRAM_LEAD=1` in production's `.env`, no rebuild) | On in production since 02:22 UTC on colum's instruction, ahead of its full test. Evidence so far: the screen (+7.5%, +11.6%, +10.1% cold prefill at 8K, 32K, 100K on random-word prompts) and the narrow run (+4.9%, +5.0%, +4.5% on natural text, decode at one and four streams unchanged, replies identical), each one process per arm under 15 minutes old. [Screen](runs/screen-20261118/RESULTS.md) · [Narrow run](runs/suite-20261119/RESULTS.md) · [Deploy record](runs/deploy-lead-20261119/deploy.log) **Owed at the next full round:** production (lead on) against a lead-off control, three rotated passes on private spill directories. Cells: cold prefill on the natural-text corpus at 8K, 32K, 100K and 300K; the 1–2K uncached tail of a warm turn on 100K and 300K histories; decode at one, four and eight streams; each before and after a soak of at least 1.5M uncached prefill tokens and an hour of process age; the long-lived deep-session run; both ranks' lowest `MemAvailable`. Gates: `exact.py` 39 cases, `validate.py` both ways, the 80 scores equal. **Keep if** natural-text prefill gains at least 3% at 32K and 100K in every pass, the warm tail is not slower, and no decode cell falls below the control gap before or after the soak; otherwise delete the `.env` line and restart. The lead reader was on during the November 13 slowdown (with the 256 MiB row cache, off here). Do not combine it with `TF_DS_ENGRAM_DEDUPE` (that pair broke the ranks on November 12). **Depth evidence, November 20:** lead on and off gave identical token hashes on a greedy recall probe at 4K, 64K, 256K and 530K, cold and cache-resumed ([results](runs/depth-quality-20261120/RESULTS.md)); this covers 300K-class output identity but not speed, soak or the exact panel. Owed separately: the repo baseline at sampled settings on the same probe (`sampled.py`), which isolates the lossy T13 prune. **Runner notes:** the campaign runner's base arm sets `TF_DS_ENGRAM_LEAD=0`, so base is no longer production: name the arms lead-off and production. Retake `production.rank0.env` and `production.rank1.env` before a run, or the restore's env check compares against the old production. `runs/suite-20261119/schedule.full.json` is the saved 18-block plan and still treats lead as the candidate. |
| **Queued with it: the draft-widening threshold** (`TF_DS_DRAFT_BLOCK_KEPT=3` and `2`, default 4, no rebuild) | Production widens a stream's pass to six drafts only after a round that kept four rows. Widening every pass gained 7.2% on the JSON prompt and 5.5% on the code prompt against 2.5% and 3.1% for the default, and cost about 1% on prose. A lower threshold may recover part of that. Replies are unchanged by construction. [Draft block](runs/draft-block-20261118/RESULTS.md) **Keep a value if** the code-like single-stream cells gain at least 2% over the default in every pass, with prose and the four- and eight-stream cells inside the control gap and every reply hash equal to control. |
| **Closed November 19: a seven-draft block, plain or with the draft ladder** (`TF_DS_DRAFT_BLOCK=7`, `TF_DS_DRAFT_LADDER`) | Measured on 21 code-like prompts with production's six as control: seven is 0.4% faster (±0.5% across prompts, 10 of 21 slower), and a ladder that offers the seventh draft only at a landing chance of 0.43 or more does the same (paired difference −0.03 ± 0.14%). A seven-row pass with the seventh unused is 0.9% slower than six by rounds and verified drafts, which is what cancels the seventh draft's gain. Per prompt seven runs from 6.1% faster to 3.5% slower, and not by language. The same run measured six 3.2% faster than five (slower on none of the 21 prompts; the 39-case exact panel was not run on the ladder). The ladder is built (bench image `bench-draft-ladder-0bd63fda950f` on both ranks, branch `draft-ladder` in `~/Workspace/tfref-draftblock`) and not in the recipe. [Results](runs/draft-ladder-20261119/RESULTS.md) Reopen only with an answer to why a seven-row pass makes the round loop verify 1.6% more of the first six drafts. |
| **Narrow run done, November 19: those three, one block each** | The full 18-block suite was cut to a 40-minute run: one block per arm on fresh processes, all gates passed. Nothing was deployed by the run; the lead reader was turned on in production afterwards (row above). **Lead reader:** cold prefill on natural text +4.9%, +5.0% and +4.5% at 8K, 32K and 100K, decode at one and four streams unchanged, replies equal. **Threshold 3 and 2:** no gain beyond the 0.7% gap between controls; threshold 2 gained 2.4–3.0% on the JSON prompt only. **Seven-draft block:** Python +6%, Go and TypeScript +1–3%, SQL and sampled Tetris −2%, two streams −1.9%, total equal to control; its keep rule is not met. (The per-language reading did not hold on 21 prompts: see the closed seven-draft row.) [Results](runs/suite-20261119/RESULTS.md) Not covered: pass-to-pass spread, the exact panel, the soak, 300K and the long-lived deep-session run, which the lead reader still owes. The full plan is saved as `runs/suite-20261119/schedule.full.json`. |
| **Closed November 19: fused kernels 0193–0198** (`TF_DS_FUSED_ROUTE`, `TF_DS_IDX_FUSED`, `TF_DS_FUSED_COMPRESS2_NORM`, `TF_DS_FUSED_ROPE_STORE`) | Short A/B on two bench images, controls bracketing each. Fused route (0193/0194): 31/39 exact cases differ, greedy single-stream −19% (Tetris) and −18% (code), four-stream −11%; likely because one block a row recomputes the full gate (source reading, not profiled). 0195+0197+0198: exact suite identical, rates inside the ~1.8% control spread, but the 1024-token greedy Tetris reply changed near token 984 (bundle; the drifting patch is not isolated). 0196 ran only with the route on. Everything on: same as route alone. [Results](runs/fused-bench-20261119/RESULTS.md) Reopen the route only with a gate split across expert blocks and a bit-identity fix; PDL (0126) untested. |
| **Parked: round-wide expert caps** (`TF_DS_T13_EXPERT_CAP=32,1` or `24,1`) | The screen raised output 8% and 13% at four mixed streams and 5% and 15% at eight Tetris streams, with single-stream likelihood inside the +0.020 line. Replies then depend on the other streams in the round (2 and 1 of 8 concurrent replies equalled solo), and quality at eight streams is unread. A per-stream cap would not bind at these sizes with several streams, and a cheaper first pass to choose experts could save 2–3% at most. Reopen only if load-dependent replies are acceptable; start with `32,1` on the eight-stream SQL and tool panels with reply text saved. [Screen and source reading](runs/screen-20261118/RESULTS.md) |
| General serving versus the deployed control | The eight warm switches improved historical uncached 8K/32K/100K prefill by 6.9%/5.3%/4.1%. On the pinned image, matched warm TTFT improved 5.8%/4.9% at 4K/100K with small mixed decode effects. On the pinned three-arm crossover the candidate lost 3.5% all-active output at 3K × 4 and 1.2% at 300K × 4. The latter improved median TTFT 4.2% and 11.2%. At 3K × 4 the isolated token cache changed output −0.1%/−0.2%, adding Engram yielded −4.1%/−2.1% versus deployed, and the eight-switch result depended on the seed. [Attribution](runs/overnight-opt/RESULTS.md#pinned-configuration-attribution-and-keyed-drafts) [Direct crossover](runs/overnight-opt/RESULTS.md#same-image-production-crossover-at-3k-and-300k) |
| Long-context decode and read bandwidth | Keep fetch-mode KV, the ordinary index bucket and L2 prefetch. Fixed-row indexer steps are reproducible, but the graph-on finer-bucket crossover did not yield a repeatable serving gain. Graph-on decode used 146–162 GB/s per rank. Four-way routed experts reached 93–96% of the pair-local scattered-read reference on both ranks. On identical four-stream outputs at 300K, adding CPU streaming reads raised physical round time 16.8% and 6.9% against the bracketing controls. Detailed speculative work adapted during the intervention. [Serving and phase counters](runs/overnight-opt/RESULTS.md#whole-engine-decode-with-serving-graphs-and-speculation) |
| Engram | Leave AIO batch size and reader affinity at their serving defaults. The repaired 64-entry batch slowed both 8K and both 32K samples. Verified reader placement on CPUs 10–14 under a 10–19 service allocation slowed 32K by 30–31% versus 15–19. These screens close the earlier startup failures. [Controls and live masks](runs/overnight-opt/RESULTS.md#engram-asynchronous-io-and-cpu-placement) |
| Pruning | Keep current uncapped T13. At width eight the cap improved SQL/tool throughput, but failed the +0.020-nat quality margin on two independent 20-source sets and lost the solved SQL case. The returning control reproduced 80 scores and 40 task-turn hashes. [Held-out quality and routing](runs/overnight-opt/prune/RESULTS.md) |
| Admission and eviction | Keep ten-second held-prefix patience for general serving. A guarded 500 ms policy prevented an observed 150 s cold revisit, while a delayed peer blocked its synchronous check for ten seconds. The natural 142-token lead finished in 3.48 s with short-held fixed grace and 10.82 s with short-held adaptive grace. The matched fully backed repeat took 29.36 s guarded and 30.14 s plain. Keep fixed finish grace. The alternative lease needs two-rank retention and invalidation guarantees. [Pressure controls](runs/overnight-opt/kv/RESULTS.md) |
| Snapshot size and exact resends | Trim passed 39 frozen and eight saved checks, cutting ordinary entry cost 44.1%. At 84 sessions the 512 MiB trial eliminated 12 evictions and eight disk restores, saving 275 ms over 12 revisits with all 96 outputs identical. Trim qualifies for this cache-pressure profile on the repaired image. The repaired full-end resend passed its five greedy pairs and four long score pairs. The off-grid 8,202-token append still changed its greedy reply. In the supported unsharded, unsliced, memory-only 0121 trial both odd-length appends diverged from cold. Keep 0121 off. [Snapshot findings](runs/overnight-opt/RESULTS.md#trimmed-ordinary-snapshots) |
| Head, greedy exchange and BF16 partials | Final-chunk head skipping slowed 8K and 32K. The original greedy rank reduction failed a concurrent exact gate and its GPU unit check. The repaired kernel passed 36 hardware cases, 12 matched serving outputs, 39 frozen cases and eight saved replies. Four-stream greedy throughput improved only 0.35%, below the full-benchmark gate. The BF16 bulk repair passed the source-paired +0.020-nat margin, but changed greedy outputs. Both 32K prefill gains were below 3% (+2.27%/+2.62%). Retain FP32 for general serving. [Odds results](runs/overnight-opt/RESULTS.md#distributed-greedy-argmax) |
| Hot prefetch and keyed drafts | Both ranks published 40-layer plans with hot:2 and hot:4, all 24 outputs matched, and both policies passed validation. Neither offered a consistent speed gain. Keep extra expert prefetch off. Keep keyed drafting. Turning it off on the eight-switch arm reduced tokens per stream per physical round on all four matched width/seed cells and hurt three output-rate comparisons. [Draft crossover](runs/overnight-opt/RESULTS.md#pinned-configuration-attribution-and-keyed-drafts) [Build corrections](runs/overnight-opt/build-audit.md) |
| Dense and routed prefill headroom | Cold dense projections reached 35%/67% of the same-shape FP16 reference on head/worker. Excluding the first occurrence at every shape left the two ranks at 34.9%/66.9%. Twelve of 78 repeated head calls at (2048,1280,16384) took 475 ms together and averaged 3.75 GB/s. Capture GPU start/end events against synchronized host intervals for this shape before changing the GEMM. [Repetition analysis](runs/overnight-opt/dense-routed-prefill.md) For routed gate/up and down, capture per-expert row counts and replay the same quantized grouped shapes; their current useful TFLOP/s has no matched arithmetic reference. [Per-rank measurements](runs/overnight-opt/RESULTS.md#both-ranks-at-matched-operation-shapes) |

Compare warmed controls on the same image with matching prompt bytes, cached cuts, start occupancy, output budgets and active stream counts. Use graph-on timings for serving claims. The eager and shape-instrumented kernels isolate causes. A failed record, missing comparison, short all-active window or mismatched output limits which conclusions a run supports. Higher-budget SQL recovery and production traffic frequencies answer different questions from the completed quality gates.

## Original experiment designs

The following sections retain the original build instructions, hypotheses and dated results. For a runtime-only comparison on the current deployment, use the image and switches recorded in the [handoff](continue.md).

### Before the first run

- The image must be built from this recipe's patch series (`./start.sh` rebuilds it; the tag changes with
  `zig/patches/`). Both Sparks need the Engram tables on local disk (`HEAD_ENGRAM_DIR` on the head,
  `WORKER_ENGRAM_DIR` on the worker): over NFS a random 4 KiB read was ~940 µs against ~90 µs local.
- `TF_DS_ENGRAM_METRICS` is on by default (1): `tensorfold:engram_lookup_seconds` is the read-latency histogram, with
  `_count` (reads) and `_sum` (seconds). It is the only per-read signal; use it for every Engram row below.
- Identity harness: `tools/exact.py` (greedy, 39 prompts through 100k tokens, resumes and 8 concurrent) must report
  IDENTICAL for each switch before its timing counts. `tools/validate.py` is the quicker smoke (text, thinking, tool
  call, image, 8 concurrent).
- Timing: `tools/bench.py` on **natural prompts** with a fresh `--salt` per run, medians of 3 alternating runs, and
  the draft acceptance rate recorded next to the speed (`docs/measurements.md`: filler prompts moved 30-60 % where
  natural text moved ~0 %). Treat differences under ~2 % prefill and ~3 % decode as noise.
- Record the head's and worker's lowest `MemAvailable` for every run: `SCALES=map`, a warm mapping and a row cache
  all trade memory or page cache for speed.

## Implemented switches (`zig/patches/0094`-`0098`)

All off by default; every one is bit-identical, so `exact.py` must pass on each.

| Run | Flags (export; `start.sh` gives them to both ranks) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Trim | `TF_DS_ENGRAM_TRIM=1` (mounts `/engram` rw; a hardlinked table is skipped) | `du -B1 -s $HEAD_ENGRAM_DIR` and `$WORKER_ENGRAM_DIR` before/after a start; `find -printf '%p %s\n'` unchanged; `exact.py`; prefill/decode unchanged | ~101 GB freed per Spark that owns a real copy, start still passes `port_engram_check`, replies identical. No speed change expected |
| Scales mapped | `TF_DS_ENGRAM_SCALES=map` | `engram_lookup_seconds` count and sum per prompt (bench.py `--metrics`), prefill 8K/32K, decode 1 stream | Count and sum fall; prefill or decode does not regress. If it regresses, the page-fault path is the cost (see `map-warm` below) |
| Dedupe | `TF_DS_ENGRAM_DEDUPE=1` | `engram_lookup_seconds_count` per prompt, prefill on prose *and* on a prompt with a repeated paragraph (the ceiling), decode; record on BOTH ranks (`engram_lookup_seconds` covers rank 0's repeat-rich 2-gram half only) | Count drops by the repeat rate with identical replies; prefill improves. The repeated-paragraph run shows the maximum the switch can buy |
| Fast | `TF_DS_ENGRAM_FAST=1`; optionally `TF_DS_ENGRAM_AIO_SLOTS=8192` | prefill 2K/8K/32K, decode; with `RANK1_PROFILE=1` the `prefill_engram_ms` split (rows/upload/mm/exchange/gate) | Any gain above noise. Expected under 1 % each; the switch exists to measure, not to claim |
| Strict | `TF_DS_ENGRAM_STRICT=1` | the pair starts, `exact.py` identical, no new log refusals | It must be a no-op on this checkpoint; a refusal means the tables are not what the decode assumes |

Combinations worth one run each (same identity rule): `SCALES=map`+`DEDUPE=1`, `DEDUPE=1`+`LEAD=1`, `FAST=1`+`DEDUPE=1`,
`AIO=0`+`SCALES=map`.

## Existing engine knobs (no patch; put these in the matrix)

| Run | Flags | Measure | Why |
| --- | --- | --- | --- |
| Lead readers | `TF_DS_ENGRAM_LEAD=1` (needs `ENGRAM_AIO=1`, the default) | prefill 2K/8K/32K, `engram_lookup_seconds`, `exact.py` | Reads each layer a chunk ahead on a ring of its own; measured in the combination campaign below. The cleanest test of whether overlap is the limit, and it shares the dedupe path |
| Buffered reads | `ENGRAM_AIO=0` and `ENGRAM_AIO=0 TF_DS_ENGRAM_SCALES=map` | prefill 2K/8K/32K, decode | O_DIRECT bypasses the page cache; repeated 2-gram rows in decode could be served from cache. Once measured at ~0.5 s against ~0.07 s per chunk — retest only with the scales mapped |
| Metric overhead | `TF_DS_ENGRAM_METRICS=0` | prefill/decode against the default | Control: two clock reads per gather, not per row. Confirm it is noise before trusting histogram deltas |

## Proposed (implemented as `0129`-`0145`; see "Ready to benchmark" below)

> Every row of this table is now built and, with two exceptions, benchmarked by the patch numbers in the section
> "Ready to benchmark: patches `0129`-`0145`" further down. `TF_DS_ENGRAM_SPLIT=2` was not landed (`LEAD` already owns
> the second AIO ring and `aioGather` writes contiguous slices), and the row-cache entry here is `0136`'s map fix.

| Run | Flag | What it would tell you |
| --- | --- | --- |
| Warm scales | `TF_DS_ENGRAM_SCALES=map-warm` (`madvise(MADV_WILLNEED)` on the mapping) | Whether `map`'s page-fault path is the cost: warmed, every scale row is a resident load instead of a fault-in. This decides whether `map` is a win or a wash |
| Decode threads | `TF_DS_ENGRAM_DECODE_THREADS=4` / `5` / `8` | The container has 5 fast cores; 8 decode threads oversubscribe and steal from the engine's threads |
| Pool threads | `TF_DS_ENGRAM_POOL=8` / `16` / `64` (with `AIO=0`) | The pool's size when the ring is off |
| CPU placement | `TF_DS_ENGRAM_CPUS=10-14` with `CPUSET=10-19` | Moves hashing, decoding and AIO reaping off the X925 cores the engine uses; the only CPU-contention experiment |
| Read depth | `TF_DS_ENGRAM_SPLIT=2` (two rings) or `TF_DS_ENGRAM_AIO_BATCH` | Whether 0.07 s per chunk is the drive or the single reaping thread |
| Host timing | `engram_host_ms` on the `done` line or in `EngramStats` | Makes the hash/decode items of `FAST` visible without `--profile`'s stream synchronization |

## Hit-rate patches (`zig/patches/0110`-`0116`) and finding 4

`docs/prefix-cache-hitrate-review.md`'s ranked findings as seven env-toggled patches (host-tested and clean-room
validated). Same one-switch-at-a-time rule. Every switch is cache policy, spill manifests or
metrics (replies keep their bits), but run `tools/exact.py` on each arm that lands anyway. Keep `TF_DS_CACHE_METRICS`
on (its default) in every arm — its counters (`prompt_cache_memory_*`, the resume-depth histogram, the miss causes)
are the measuring stick for the other six; `usage.prompt_tokens_details.cached_tokens` is the per-request signal.
The Row cache row of "Proposed" above is implemented as `0114` here, with its scope corrected: a process-lifetime
cache (cross-chunk, cross-round, cross-request, cross-turn), keyed (layer table, row id) — row-id ranges restart
per layer — and the decode rounds hooked at the AIO path (they bypass `readRows` under `ENGRAM_AIO=1`).

**Pass results (2026-11-13, one signal-grade block per arm, image `0110`-`0116` + the `0114` null-init fix).**
All nine validate comparisons matched the previous image's `base.validate.json` on
hardware (this pass is also the crash regression's first hardware check). Bookends drifted over the run
(hr-base 2121.7 → hr-base-final 2023.8 prefill8k, −4.6%; decode −2.2%), so arm deltas inside that band are noise.
`prompt_reuse` read BELOW90 on every arm including base (turns resume 88-89%): the probe's 90% bar sits above the
structural ceiling at its 12K size (finding 18: grid slack + 128-row replay ≈ 11-17% recompute per turn) — clean
turns cannot show the edit/rewind wins. `prefix_fanout` gave 0 duplicate prefill tokens at concurrency 1/2/4/8 on
every arm (the uniform-tools case; finding 1's per-agent-tools win needs its own probe).

| Run | Gate | Prefill 8K | Decode | engram read sum (s) | Sign | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| hr-base | IDENTICAL | 2121.7 | 63.7 | 23.80 | — | baseline, cross-image identity |
| hr-cuts | IDENTICAL | 2084.2 | 62.9 | 23.76 | (probe gap) | no regression; win case (per-agent tools) untested |
| hr-edit | IDENTICAL | 2107.1 | 63.3 | 23.76 | (probe gap) | no regression; win case (history edit) untested |
| hr-protect | IDENTICAL | 2109.0 | 65.3 | 24.10 | (probe gap) | no regression; win case (8-stream pressure) untested |
| hr-stride | IDENTICAL | 2118.6 | 62.1 | 24.33 | (probe gap) | no regression; win case (6-10 turn rewind) untested |
| hr-rowcache | IDENTICAL | 2050.3 | 64.1 | **10.16** | read metric | **−57% Engram read time at equal lookup count; costs ~0.7 GiB MemAvailable low at 256 MB** |
| hr-plan | IDENTICAL | 2091.7 | 60.7 | 23.90 | (probe gap) | no regression; win case as Keep protect |
| hr-metrics0 | IDENTICAL | 2084.8 | 61.9 | 24.73 | **+** (default on) | the counters cost nothing (2638/24.73 ≈ base) |
| hr-base-final | IDENTICAL | 2023.8 | 62.3 | 24.40 | — | drift bookend |

The 256 MB row cache cut Engram read time, but the later 0136 long soak showed no serving gain that justifies its memory. Leave it off. `0116` remains enabled. `0110`/`0111`/`0112`/`0113`/`0115`: **no regression, wins unproven** — sign after their targeted
probe rows below (findings 1/2/3/5/7 win cases are edit, rewind, tool-drift and pressure probes the generic block
does not exercise). For any toggle kept after its probe: the strict 3x alternation, per the protocol.

| Run | Flags (export; `start.sh` gives them to both ranks) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Shared cuts | `TF_DS_SHARED_CUTS=1` | `tools/prefix_fanout.py` with per-agent tool lists (base: ~0 reuse) and with a shared first-user document; duplicate tokens per follower | Followers resume the shared block (duplicates within the 2,048/follower tolerance) where the base run re-prefills everything |
| Edit mark | `TF_DS_EDIT_MARK=1` | an edit/regenerate of an early history message in a growing session, then 3+ more turns; `cached_tokens` of the turns after the edit | Later turns resume at the edit point instead of the system cut; clean sessions unchanged |
| Keep protect | `TF_DS_KEEP_PROTECT=1` | 8-stream divergent-tail agent traffic interleaved with `tools/prefix_fanout.py --expect-one-fill` over 1-2 h; fan-out duplicate tokens late in the run; the eviction cause counters | The shared prefix survives pressure (fan-out stays one-fill after hours) where the base run degrades to N fills |
| Spill stride | `TF_SPILL_STRIDE=8` (also `4`) | a long conversation: rewind/regenerate 6-10 turns back; the rewind turn's `prefill_s` and `prompt cache: resumed at`; disk in `SPILL_DIR` | The rewind resumes a kept cut (re-prefill ≈ the delta) instead of the middle; disk cost near zero (manifests only) |
| Row cache | `TF_DS_ENGRAM_ROWCACHE_MB=64` / `256` | prefill/decode on prose and code with `engram_lookup_seconds` count+sum (both ranks), a re-prefill after eviction or a restart; `MemAvailable` | Read count and sum fall (cross-turn/cross-request repeats), prefill improves; no memory pressure |
| Shared plan | `TF_DS_SHARED_PLAN=1` | as Keep protect (it protects the same states); with the spill tier on, also `Tier.share` quota accounting | Same as Keep protect; no further gain over either alone is expected at light load |
| Cache metrics | `TF_DS_CACHE_METRICS=0` vs default | control arm only | The counters' overhead is noise |

Combos worth one run each: `SHARED_CUTS=1`+`SHARED_PLAN=1`+`KEEP_PROTECT=1` (the fan-out stack), `SPILL_STRIDE=8`
with the spill tier on (`SPILL_GIB=64`), `ROWCACHE_MB=64`+`0096`+`LEAD=1`.

**Finding 4 — `SPILL_MIN_TOKENS` (no patch; measure before any change).** Agent sessions below ~10K tokens never
reach disk today (`SPILL_MIN_TOKENS=8192` plus the keep ceiling `floor2048(len-128)`): eviction or a restart loses
them entirely.

| Run | Flags | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Lower the floor | `SPILL_MIN_TOKENS=2048` (in `scripts/local.sh`; same on both ranks) | a 4-9K-token agent session: stop/start the pair, resume; the first turn's `cached_tokens`; spill counters (`short`, writes); disk churn in `SPILL_DIR` | Short sessions recover from disk instead of re-prefilling; block sharing across turns keeps the churn low |
| Exempt shared cuts (proposed, needs `0099`+) | — | the Shared cuts run with a pair restart between rounds | A shared cut ≥ 2048 tokens survives a restart and serves every follower — worth a patch that exempts shared points from `min_tokens` |

## Baseten-ultrafast candidates (`0119`, `0121`) and the Tetris row

`docs/baseten-ultrafast-plan.md` items 1 and 3 phase A, after the single-pass signal (`0119` saved ~46 ms of
`encode_s` on a 95K resumed turn; `0121` cut small-append TTFT in the initial screen; the cold comparison below found divergent replies). Same one-switch rule.
`0121` refuses spill, KV shard and the prefill slice, so its base is production with those three off, not production.
The generic `bench.py` block (cold, single-turn prompts) cannot show either win; it is the no-regression half. The
win is in the targeted probes: `encode_s` on a resumed 100K turn (fresh seed per arm, spill keeps states across
restarts), `tools/resume_exact.py` and `tools/prompt_reuse.py 12000` (its 88-89% ceiling is what exact resume should
lift). Driver: `runs/ultrafast/uf_full.sh` on the head (3 passes, arms alternated in each).

| Run | Flags (export) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Token cache | `TENSORFOLD_TOKEN_CACHE_MB=32` against production | bench block (no regression); `encode_s`, `render_s`, `ttft_s` of the resumed 100K turn (done lines) | `encode_s` falls on resumed turns by more than run-to-run noise, nothing regresses |
| Exact resume | `TF_DS_EXACT_RESUME=1` against `SPILL_GIB=0 KV_SHARD=0 TF_DS_PREFILL_SLICE=0` | bench block; `resume_exact.py warm` per pass (cached = previous prompt, reply shas = the cold rows'), `prompt_reuse.py 12000`'s resume %, the resumed 100K turn's TTFT | TTFT falls on every append size with every reply identical. Known gap: `validate.py`'s image case fails (images are refused, `lanes.zig:1064`), and production cannot use it until it works with spill/KV shard |
| Tetris (like-for-like with Baseten's 5.6 tokens a step) | production (`TF_DS_T26_KEYED_DRAFT=1`), and `TF_DS_T26_KEYED_DRAFT=0` | `uf_probe.py tetris`: "implement the tetris game. output HTML code directly - no explanation.", thinking off, 1 stream, 2,048 tokens, greedy once and temperature 1 (top_p 0.95, top_k 20) for 3 seeds; tokens_per_round, accept and tok/s from the done lines | No keep/drop: a figure to compare with Baseten's before reading anything into the gap (this repo's matrix: 2.0-2.35 a round on prose and agent traffic) |
| Keyed draft on/off, sampled | same two arms | `uf_probe.py sampled` (temperature 1, 256 tokens, 1 and 4 streams, 2 reps) | Settles the single pass's reading that keyed-off was ~4% faster than keyed-on on sampled streams (2 reps) |

**Results (2026-11-15, 16:44-18:10Z, image `exl3-tp2-f3f69b951cd3` in every arm, 3 passes alternated).** Artifacts:
`runs/ultrafast/full/` (`summary.txt` from `uf_analyze.py`, `status.log`, per-arm logs), cold reference rows from an
arm with `--prompt-cache-gib 0`. Medians over 3 passes, [min-max].

| Arm | Gate | Prefill 2K / 8K / 32K | Decode | Multi 4 | Resumed 100K turn | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| production | IDENTICAL x3 | 1794 / 2083 / 2131 | 64.0 | 123.1 | encode 0.108 s, TTFT 0.7 s | base |
| token cache 32 MB | IDENTICAL x3 | 1798 / 2079 / 2139 | 63.2 | 122.8 | **encode 0.056 s** [0.056-0.057], TTFT 0.7 s | **+**: −52 ms a resumed turn (~7% of a 0.7 s TTFT), no cold-turn gain, no regression |
| resume base (spill/shard/slice off) | IDENTICAL x3 | 1829 / 2127 / 2167 | 64.5 | 125.4 | cached 94,208, TTFT 0.7 s | base for exact resume |
| exact resume | **FAILED: image** x3 | 1834 / 2129 / 2173 | 65.0 | 126.1 | cached ~94,866, **TTFT 0.3 s** | faster, but **not bit-identical** (below) |

- **Exact resume is LOSSY — final (step 0 done 2026-11-16).** `resume_exact.py` against true cold rows: turn 6
  (1-token append, resumed at the previous prompt's end, 8,228 of 8,281) gives `96d0fd7dcd8e` where cold and grid resume give
  `9c5a438c3fb3`, in all 3 passes (deterministic); turns 7-8 match the resume arm's own divergent turn-6 byte stream
  (both arms' later prompts carry the divergent reply). The `--logits-sha 1` instrument printed nothing on this build;
  the teacher-forced score found bits move for every off-grid cut measured (DEBUG_SPLIT 8228/9300/9380/9440/9480, both
  prompt sizes, both directions, every tail class 16-2012; t3k bit-stable). `0121` stays opt-in (`TF_DS_EXACT_RESUME=1`),
  never default; grid resume is production. See `decisions.md` 2026-11-16 and `runs/matrix/step0-cold-shape/`. Also:
  turns 2-4 (prompts of 121-2,727 tokens) resume nothing in either arm, and turn 5 resumes 2,048 (not 2,727) with the
  switch on.
- Exact resume's speed is real: small-append TTFT 1.31 → 0.26 s (turn 6), 1.41 → 0.41 s (turn 7); `prompt_reuse.py
  12000` 88% / 1.3 s → 98% / 0.7 s a turn, all 3 passes.
- Bench blocks: every arm inside the noise band against its base.

| Tetris (1 stream, 2,048 tokens, thinking off) | tokens/round | accept | tok/s |
| --- | --- | --- | --- |
| greedy (keyed on or off: identical) | 4.8 | 80.3% | 117.6 |
| temperature 1, keyed draft on (production), 9 seeds | 4.7 [4.5-4.9] | 80.6% | 115.2 [111.7-119.6] |
| temperature 1, keyed draft off, 9 seeds | 4.6 [4.4-4.9] | 77.7% | 112.4 [110.2-119.5] |

Baseten reports 5.6 tokens a step on this prompt; this pair does 4.7-4.8 (2.0-2.35 on prose). The gap is ~15%, not
the 2.5x the prose figure suggested.

| Sampled prose (agg tok/s, 6 runs each) | 1 stream | 4 streams |
| --- | --- | --- |
| keyed draft on (production) | 54.2 [53.3-55.3] | 105.3 [104.5-106.1] |
| keyed draft off | 56.4 [55.3-57.7] | 107.8 [107.6-107.9] |

Keyed draft costs 2-4% on sampled prose (ranges do not overlap at 4 streams) and gains ~2.5% on sampled code (Tetris,
ranges overlap). No `.env` change on this evidence; the traffic mix decides it.

**Rank-0 round profile (2026-11-15, `runs/ultrafast/prof/`).** `tf-dsv41-lanes` as rank 0 (batch, `--requests`) and
rank 1, production settings with spill off; attribution only (it syncs every round). Every batch reply equalled the
served one (`all_equal: true`). Per round, graphs on:

| Load | rows/round | step | forward | draft pass | enqueue | send + absorb + logits + sample |
| --- | --- | --- | --- | --- | --- | --- |
| 1 stream (7 requests in turn) | 3.9 | 34.4 ms | 30.5 (89%) | 3.4 (10%) | 1.0 | 0.4 (1%) |
| 4 streams | 7.4 | 53.6 ms | 47.8 (89%) | 5.4 (10%) | 4.1 | 0.4 (<1%) |

Eager rounds (`--profile 2`, `GRAPHS=0`), GPU time by phase a round, 1 / 4 streams: MoE 16.1 / 31.5 ms, attention
1.8 / 2.0 (indexer `a_idx` 0.6), gathers 2.1 / 2.2, head 1.1 / 1.2, Engram 0.5 / 0.6, mixes 1.3 / 1.4. Reading: the
host path is ~1% of a round, so plan items A (one graph a step) and C (PDL) have little to win; MoE (EXL3 experts,
the kit's kernels) is 60-70% of the GPU time and is the lever; the drafter pass is the next 10%. The overnight
`enqueue_ms` 14.3 (rank 1, 8-stream rounds) does not reproduce at 1-4 streams (1.0-4.2 ms).

## Decode-step structure (`0123`-`0128`) — the plan's "To explore" items and their original tests

`docs/baseten-ultrafast-plan.md` items 1, A-E, on `decode-step-structure` of the engine fork (export:
`/home/colum/Workspace/tfref-next-handover/`, patches `0123`-`0128` + `README.txt`, applies onto the
`series-0119` base = TF_REF fbbc971 + 0001-0122). All off by default, all bit-identical by construction.
The group was subsequently screened and the combined candidate exact-tested in [combo](runs/combo/RESULTS.md). Individual effects require individual gates. The rank-0
profile (above, "Rank-0 round profile") is the sizing evidence; 0123 is the instrument that measures the
served process the same way live (`/metrics`, attribution only, syncs every round).

| Run | Flags (export) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Serve profile | `TF_DS_ROUND_PROFILE=1` (rank 0's server), scrape `/metrics` during any arm | the profile line shape vs `tf-dsv41-lanes --profile` (must be key-equality equal apart from `step_ms`); `render_s`/`encode_s` unaffected; sanity of `rounds`/`pass_ms` during live traffic | nothing to keep — it is the instrument; if the line is missing or torn, fix forward before any other arm reads it |
| Serve profile with phases | `TF_DS_ROUND_PROFILE=2 GRAPHS=0` | `phases_ms` at 1 and 4 streams vs the batch `--profile 2` eager figures (MoE 16.1/31.5, a_idx 0.57/0.61) | instrument only; validates the serve path measures what the batch driver measures |
| Device accept | `TF_DS_DEVICE_ACCEPT=1` | `exact.py` IDENTICAL (expected; the kernel is a re-implementation of the chain accept); then `--profile` send/logits/sample shares and round wall vs base | it removes the host accept walk, not the sync — likely no measurable win; if IDENTICAL holds and nothing regresses, keep with 0123's line as the share of the host path it could ever reclaim; if not, it is a wrong kernel |
| Compressor side stream | `TF_DS_COMPRESS_SIDE=1` | `exact.py` IDENTICAL; `--profile 2` eager phases: attention phases minus the compressor's share; tokens/round unchanged | the layer's launches overlap its query path; keep only if exact.py holds, the phase profile shows the hide, and nothing regresses |
| PDL in our kernels | `TF_DS_PDL=1` | `exact.py` IDENTICAL; then `tools/nsys_breakdown.py` on an nsys trace (gap between consecutive kernels, or lack of one); wall vs base at 1 and 4 streams | only if the gap table closes a measurable share of step_ms; profile says the host path is ~1%, so expect little; if nsys does not work on GB10 (unchecked), drop without guilt |
| Fused compressor | `TF_DS_FUSED_COMPRESS2=1` | `exact.py` IDENTICAL; `--profile 2` `a_idx` share and launches a phase (with `GRAPHS=0`); 1 and 4 streams | ~20 launches a round off; keep only if the phase profile or wall shows it; the MoE share (60-70%) says this is not where the time goes |
| Engram both at first cut | `TF_DS_ENGRAM_BOTH=1` | `exact.py` IDENTICAL; one chat smoke with `GRAPHS=0` and one with `GRAPHS=1`; eager phases: Engram 0.47/0.57 ms a round; stretch count in the log (2 graphs a round) | zero-cost structural win (one host step fewer a round); if IDENTICAL holds, this is the only one of the series that is a plausible keeper |

**Update 2026-11-16:** the five switch rows (`0124`-`0128`) now have `exact.py` IDENTICAL and a paired A/B (the two
`0123` serve-profile rows were not run). See "Combined configuration
(2026-11-16)" under Results.

**Results (2026-11-15, image `exl3-tp2-b40ea899e367` = TF_REF + 0001-0122 + the six patches, switches unset
unless named).** Gate is `validate.py --save runs/matrix/b40*.validate.json --compare runs/matrix/base.validate.json`
(campaign's base save, image `f3f69b951cd3`); it reports IDENTICAL 8 of 8 with the switches off and with
`TF_DS_ENGRAM_BOTH=1` on. One bench block per arm, n=1 (noise band ±5-10%), `prefill-sizes 2000,8000,32000`:

| Block | prefill 2K / 8K / 32K (tok/s) | TTFT 2K/8K/32K (s) | decode tok/s | multi.1 tok/s | multi.4 tok/s | errors |
| --- | --- | --- | --- | --- | --- | --- |
| base (six switches off) | 1771 / 2095 / 2140 | 1.09 / 3.67 / 14.33 | 62.4 | 65.7 | 122.4 | 0 |
| `TF_DS_ENGRAM_BOTH=1` | 1760 / 1998 / 2103 | 1.10 / 3.84 / 14.58 | 62.8 | 61.7 | 120.7 | 0 |

Δ vs base: prefill −0.6% / −4.6% / −1.7%, decode +0.7%, multi −6.1% / −1.4% — all inside the noise band;
**no measurable improvement, no regression**. Per the keep/drop bar, nothing graduates from this block;
`TF_DS_ENGRAM_BOTH` passes its gate and stays measured-only. The `0123` instrument (serve profile at /metrics)
verified end-to-end on the live server (the log shows the `profile` + `profile_note` line at every scrape with
cumulative `rounds`/`pass_ms`/`enqueue_ms`/`forward_ms`; the scrape also carries a `-1` `step_ms` sentinel by
design). `tf-dsv41-lanes --profile` and `tools/nsys_breakdown.py` were not run. This initial screen used validate. The subsequent combo campaign passed individual resumed exact comparisons on 0124–0128 (c02/c03/c05/c06/c08). Note: the pair's **first request after a restart deterministically fails**
(`cuLaunchKernel: CUDA_ERROR_INVALID_VALUE`, `prompt=8 tokens=1`) on this image *and* on the previous one —
a pre-existing environmental condition, not a regression from the series; identical retried requests pass.
(Artifacts: `runs/matrix/b40-base.bench.json`, `runs/matrix/b40-both.bench.json`, `b40.validate.json`,
`b40-both.validate.json`; logs: `runs/matrix/*.restart.log`, `runs/matrix/bench-b40-*.log` was not kept.)

Notes: `tools/nsys_breakdown.py` is untracked here (run it as-is against `nsys export --type sqlite`);
whether nsys works on GB10/sm_121 is unchecked. Exact resume's lesson (`0121`, LOSSY until root-caused)
applies: the identity gate is the price of admission for all six. `tools/samp_test.py`-style sampled
distribution tests were outside this early screen. Combo subsequently checked four seeded sampled hashes on selected arms.

## Commands (Engram arms)

```bash
# identity first, per switch (the record must reuse the base salt and shapes)
python3 -I tools/exact.py record --out runs/exact-base.json --label base
TF_DS_ENGRAM_FAST=1 ./start.sh restart
python3 -I tools/exact.py record --out runs/exact-fast.json --label fast --like runs/exact-base.json
python3 -I tools/exact.py compare runs/exact-base.json runs/exact-fast.json

# timing, alternating with the base run, fresh salt per run (so the prefix cache does not hide the Engram work)
python3 -I tools/bench.py --suite prefill --prefill-sizes 2000,8000,32000 --reps 3 --metrics --label base --out runs/engram-base.json
python3 -I tools/bench.py --suite decode,multi --streams 1,4 --reps 3 --label base-dec --out runs/engram-base-dec.json
python3 -I tools/bench.py --compare runs/engram-base.json runs/engram-fast.json

# the read histogram (rank 0 serves /metrics on this deployment's API port, PORT=8000 in .env — export PORT=8000
# or pass --url, or bench.py/exact.py fall back to 8888), before and after one known prompt
curl -s http://127.0.0.1:8000/metrics | grep 'tensorfold:engram_lookup_seconds'

# trim: blocks, not apparent size; the listing must not change
du -B1 -s "$WORKER_ENGRAM_DIR"; stat -c '%s %n' "$WORKER_ENGRAM_DIR"/*
find "$WORKER_ENGRAM_DIR" -maxdepth 1 -type f -printf '%f %s\n' | sort
```

## Draft depth and calibration

Two questions: whether the round loop's depth choice fits this traffic, and whether the depth under test is the depth
that runs. The toggles (`0103`-`0106`) pin every head draft, so a curve is a constant rather than a rule output;
`TF_DS_T25_DRAFT_CAL` is the existing per-stream calibration (off in `.env`).

Baseline on this pair (300 recent agent requests, thinking on, `PARALLEL=8`, from the `0070` done lines): acceptance
median 88.1% (p10 82.5, min 59), 2.35 tokens/round median (p90 3.4, max 4.5), 60 ms/token per stream, TTFT median 36 s
against 2.1 s prefill. The same fields are on every done line (`accept`, `tokens_per_round`, `s_per_token`, `rounds`),
so an arm is scored from the log, not from a client average.

| Run | Flags | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Calibration | `TF_DS_T25_DRAFT_CAL=0.05`, then `0.2` | accept, tokens/round, s_per_token, decode tok/s at 1/4/8 streams, natural prompts, the quality set | more tokens/round and faster decode at the same acceptance. Author's line: greedy prose 79.7 → 85.4 tok/s, code unchanged. Not bit-identical when set: a sampled stream's draws change |
| Depth 2 | `TF_DS_DRAFT_DEPTH2=1` | tokens/round, accept, s_per_token, rows/round, `MemAvailable` low | a depth whose tokens/round per second beats the unpinned run |
| Depth 3 | `TF_DS_DRAFT_DEPTH3=1` | as depth 2 | as depth 2 |
| Depth 4 | `TF_DS_DRAFT_DEPTH4=1` | as depth 2 | as depth 2 |
| Depth 5 | `TF_DS_DRAFT_DEPTH5=1` | as depth 2 | as depth 2 |
| Pinned + calibration | `TF_DS_DRAFT_DEPTH3=1 TF_DS_T25_DRAFT_CAL=0.05` | tokens/round must equal the pinned run's | control: a pin bypasses the allocator, so T25 cannot move it; a difference means the pin is not hard |
| Copies off | `TF_DS_NO_COPIES=1` alone, then with a pin | how many rounds are copy rounds (1 row) and how many are head rounds | a depth curve read without copy rounds confounding it |
| Fix-K control | `TF_DS_FIX_K=3` against `TF_DS_DRAFT_DEPTH3=1` | tokens/round, accept | shows the older knob feeds chances (the allocator can still grant fewer) while the pin is hard; keep one |

Rules for these arms:

- At most one depth toggle may be on; two refuse the start (`error.BadDraftPin`). The checkpoint's draft count and the
  stream's room clamp a pin, so a 5-deep toggle on a checkpoint with fewer draft blocks drafts fewer.
- Greedy replies keep their bits under a pin (same accept/reject path), so `tools/exact.py` is still the identity gate;
  it does not exercise sampled draws, so score sampled arms on the quality set and `accept` as well.
- A pin changes the round's rows; `TF_DS_PLAIN_ROWS` keeps the mixed pack's cuBLAS shapes fixed, so leave it at its
  default. Copies draft 1 row and are not governed by a pin: set `TF_DS_NO_COPIES=1` when a curve must be head drafts
  only, and report which arm had it.
- `TF_DS_T25_CAL_LOG=n` prints every n-th calibrated verify (diagnostic only).

```bash
# one arm per restart; the log is the evidence (rank 0's done lines)
TF_DS_DRAFT_DEPTH3=1 ./start.sh restart
docker logs dsv41-1017-exl3-tf 2>&1 | grep 'done ' | tail -300 > runs/done-depth3.log
# mean acceptance and tokens/round of the arm
sed -n 's/.*accept=\([0-9.]*\)%.*tokens_per_round=\([0-9.]*\).*/\1 \2/p' runs/done-depth3.log |
  awk '{a+=$1; t+=$2; n++} END {print n, a/n, t/n}'
```

## Results

Measured 2026-11-12 on the pair (image `exl3-tp2-6d345f841886`, all rows one switch off/on against it). Protocol
(signal grade, operator's call; `tools/bench_matrix.sh`): one bench block per row (prefill 2K/8K/32K + decode/multi
1,4, reps 3, fresh salt), identity gate `tools/validate.py --save/--compare` against the base save ("ident" below;
per-switch `exact.py` records are deferred), quality set not scored (undefined in this repo), TRIM skipped
(destructive). Cross-image gate first: `exact.py` old vs new image, all switches off: **IDENTICAL, 39/39 cases**.
Noise band, from the `metrics0` control and the `base-final` drift bookend (2088.9, -1.2% after 1.5 h): prefill
about +-4%, decode about +-3%. Engram rows are pass 2 (21:39-22:27Z); pass 1 (20:04-21:33Z, same arms) is quoted
in the notes where it changes a read. counts/sums are the engine histogram delta over the bench block (bench.py's
own snapshots truncate before the series; see `scripts/bench_matrix.sh`).

### Engram

| Run | Flags | Prefill 8K tok/s | Decode tok/s | engram reads (count) | engram read sum (s) | exact | MemAvailable low (H/W GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | - | 2097.5 | 61.2 | 2608 | 23.7 | - | 6.1/7.8 | pass 1: 2113.9/63.5. `base-final` bookend 2088.9 (-1.2%) |
| trim | `TF_DS_ENGRAM_TRIM=1` | not run | | | | | | skipped (operator): punches ~101 GB per Spark irreversibly (nlink=1 both); deferred to todo.md |
| scales=map | `TF_DS_ENGRAM_SCALES=map` | 1981.1 (-5%) | 62.8 | 2662 | 20.5 | ident | 6.1/7.8 | -11% on pass 1 (cold mapping), -5% warm: the page-fault path is the cost. count unchanged, read time -13%. The proposed `map-warm` run decides keep/drop |
| dedupe | `TF_DS_ENGRAM_DEDUPE=1` | 2083.2 | 62.5 | 2668 | 22.2 | ident | 6.0/7.3 | no count reduction anywhere: same-prompt cold probe is 114 = 114 (see below). Neutral: not worth keeping as measured |
| fast | `TF_DS_ENGRAM_FAST=1` | 2072.5 | 64.3 | 2638 | 23.5 | ident | 6.0/7.7 | neutral, as the doc expects (measures, does not claim) |
| strict | `TF_DS_ENGRAM_STRICT=1` | 2078.9 | 63.9 | 2586 | 23.7 | ident | 6.1/7.8 | pair starts, no new log refusals: no-op as required |
| lead | `TF_DS_ENGRAM_LEAD=1` | 2227.9 (+6%) | 63.0 | 2676 | 26.3 | ident | 6.0/7.7 | **+5.0% (pass 1) and +6.2% (pass 2) prefill: the one reproducible win.** read sum up (extra read-ahead) but hidden |
| aio=0 | `ENGRAM_AIO=0` | 2121.3 | 61.9 | 2604 | 18.4 | ident | 6.0/7.7 | per-read time -22% (pass 1: -45%) but prefill unchanged: the AIO ring already hides the reads |
| metrics=0 | `TF_DS_ENGRAM_METRICS=0` | 2075.0 | 64.8 | n/a | n/a | ident | 6.1/7.7 | control, in band. histogram series absent -> flag passthrough proven end to end |

### Combinations (one run each)

| Run | Flags | Prefill 8K tok/s | Decode tok/s | engram reads (count) | engram read sum (s) | exact | MemAvailable low (H/W GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| map+dedupe | `SCALES=map DEDUPE=1` | 1711.8 (-18%) | 62.1 | 2682 | 21.4 | ident | 6.1/7.7 | pass 1: -25%. map's cost composes; dedupe adds nothing |
| dedupe+lead | `DEDUPE=1 LEAD=1` | (2101.5 prefill only) | crashed | 19 | - | ident | 6.0/7.8 | **rank pair broke in decode, both passes** (`NCCL communicator aborted`, rank 1 closed): reproducible engine bug, do not ship the combo |
| fast+dedupe | `FAST=1 DEDUPE=1` | 2091.4 | 62.8 | 2670 | 22.2 | ident | 6.1/7.6 | neutral, composes cleanly |
| aio=0+map | `AIO=0 SCALES=map` | 1163.4 (-44%) | 64.9 | 2582 | 15.4 | ident | 6.1/7.4 | pass 1: -35%. worst row of the matrix: buffered reads + mapped scales are anti-synergistic |

### Dedupe same-prompt probe

Natural prose and the repeated-paragraph ceiling (the doc's measure), identical prompts in every leg, prompt
cache flushed between legs and `cached_tokens` 0 throughout (`scripts/dedupe_probe.sh`; one prefill block 8K/32K
x3 per leg, one 2431-token repeated-paragraph request). Counts are deterministic on identical prompts (base1 ==
base2), so the delta is exact:

| Leg | prose block reads | repeated-paragraph reads |
| --- | --- | --- |
| base (cold) | 114 | 10 |
| base repeat (cold) | 114 | 10 |
| dedupe (cold) | 114 | 10 |

**Zero change on either probe.** The doc's "count drops by the repeat rate" does not materialize even on the
maximum-repetition paragraph: either the switch's rows-within-gather saving is invisible to
`engram_lookup_seconds_count` (which may count gather events) or the dedupe path does not engage on this
checkpoint. With the timing rows also neutral, dedupe signs **no measurable effect**; keeping it would need a
rows-level metric showing the saving. (An earlier probe attempt showed 114 -> 12 - that was the prompt cache's
disk tier, not the switch; the flushed design above supersedes it.)

### Draft depth

accept/tokens_per_round/s_per_token are done-line means over the bench window (median accept in the notes where
it differs); quality set not scored (operator). Pins break **concurrent** identity (`concurrent == solo` fails in
validate.py) while keeping sequential identity ("equal to the saved replies" passes) - the doc's "greedy replies
keep their bits" holds per request only. Traffic note: bench prose drafts less than the doc's agent baseline
(69.9% / 2.00 here vs 88.1% / 2.35 there); rows compare within this session.

| Run | Flags | accept % | tokens/round | s_per_token | decode tok/s | quality set | MemAvailable low (H/W GiB) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | - | 69.9 | 2.00 | 0.0239 | 68.2 | not run | 6.1/7.8 | multi-4 132.7 |
| T25 0.05 | `TF_DS_T25_DRAFT_CAL=0.05` | 68.7 | 1.93 | 0.0241 | 66.6 | not run | 6.1/7.6 | tokens/round -3.5%: the author's "more tokens/round at the same acceptance" does not reproduce here |
| T25 0.2 | `TF_DS_T25_DRAFT_CAL=0.2` | 69.2 | 1.91 | 0.0237 | 65.9 | not run | 6.1/7.6 | same direction: neutral to negative |
| depth 2 | `TF_DS_DRAFT_DEPTH2=1` | 49.4 | 1.98 | 0.0250 | 62.0 | not run | 6.1/7.8 | accept -20 pp for no speed: net negative |
| depth 3 | `TF_DS_DRAFT_DEPTH3=1` | 37.7 | 2.13 | 0.0255 | 62.7 | not run | 6.0/7.9 | multi-4 112.5 (4 stream flakes) |
| depth 4 | `TF_DS_DRAFT_DEPTH4=1` | 29.1 | 2.17 | 0.0176* | 56.8 | not run | 6.2/7.9 | *survivor bias (n=6 done lines); all 12 multi-4 requests errored |
| depth 5 | `TF_DS_DRAFT_DEPTH5=1` | 25.7 | 2.27 | 0.0179* | 56.4 | not run | 6.2/7.8 | as depth 4; multi-4 dead |
| depth 3 + T25 | `DRAFT_DEPTH3=1 T25_DRAFT_CAL=0.05` | 37.7 | 2.14 | 0.0267 | 61.5 | not run | 5.9/7.8 | control passes: tokens/round and accept equal depth 3's - a pin is hard, T25 cannot move it |
| copies off | `TF_DS_NO_COPIES=1` | 67.2 | 1.94 | 0.0244 | 65.3 | not run | 6.1/7.8 | identity kept (gate ident); mildly negative - copy rounds are cheap |
| copies off + depth 3 | `NO_COPIES=1 DRAFT_DEPTH3=1` | 38.3 | 2.13 | 0.0266 | 64.3 | not run | 6.1/7.8 | equals depth 3: copy rounds do not confound the depth curve |
| fix-k 3 | `TF_DS_FIX_K=3` | 38.7 | 2.14 | 0.0274 | 61.9 | not run | 5.8/7.8 | depth-3's numbers with identity kept (gate ident, concurrent too): if a fixed-K curve is wanted, keep FIX_K over the pins |

Cross-checks from the lab's OpenObserve store (`tools/obs_arm_metrics.py`, per-arm windows of both passes):
`TF_DS_ENGRAM_METRICS=0` zeroes the histogram series; `aio0` halves read seconds; MTP accepted/drafted collapses
to 45-55% under pins (matching the done lines). Its 60 s scrape cadence is too coarse for the count columns
above (the runner's own histogram scrape is the source) and `pair_broken` cannot see the dedupe+lead crash (the
dying container is never scraped); the engine log is the evidence there.

### Combined configuration (2026-11-16, `runs/combo/RESULTS.md`)

Image `exl3-tp2-b40ea899e367` (TF_REF + `0001`-`0128`). Runner `scripts/bench_overnight.py` with
`CAMPAIGN_OUT=runs/combo`. The base of every block is the head's production `.env`: `TF_DS_T13_EXPERT_PRUNE=1`,
`T26_KEYED_DRAFT=1`, `T27_DEVICE_DRAW=1`, `PREFILL_SLICE=5`, `SPILL_GIB=180`, `KV_SHARD=1`, `PARALLEL=8`. Blocks
alternate ABAB with a base on each side, decode is paired per prompt, and the 2400 MHz clock cap held throughout.
Gates per arm: `validate.py --compare runs/matrix/base.validate.json`, `exact.py` (resumed via `--like`, and cold
with `SPILL_GIB=0`), and seeded temperature-1 `token_sha` (sampled identity).

- **Missing `0124`-`0128` rows, one block each.** Every one passed validate and exact (39 cases, IDENTICAL).
  `DEVICE_ACCEPT` also kept the sampled shas.
  - `DEVICE_ACCEPT`, `COMPRESS_SIDE`, `PDL` and `FUSED_COMPRESS2` are neutral, within ±0.5% on decode.
  - `ENGRAM_BOTH` costs -1.0% at dec8.
- **Stack, 3 rotated passes.** The `known` arm is `LEAD=1 SCALES=map STRICT=1 TOKEN_CACHE_MB=32`. `best` is known +
  `0124`-`0127`. `full` is best + `ENGRAM_BOTH`.
  - `best` against production: prefill 8K **+6.9%**, 32K +5.3%, 100K +4.1%, p125 8K +9.6%, code 8K +4.7%,
    TTFT 8K +8.6% better, decode 1/4/8 +0.5/+0.5/+1.0%. The base-to-base gap was 0.2-1.8%.
  - `best`/`known` is within ±1.2%, either sign: `0124`-`0127` neither add nor take away.
  - `full` < `best` on 8 of 12 metrics (code 8K -2.9%). **Keep `ENGRAM_BOTH` off.**
- **Soak.** 15 uncached 100K prefills, about 1.5M tokens.
  - `best` holds +4.0% over base for the whole soak.
  - Decode after the soak equals decode before (±0.4%) in both arms.
  - A common-mode prefill step of -1 to -2% appears at rep 13 in both arms.
- **Hit-rate cluster on top of `best`.** Flags: `SHARED_CUTS EDIT_MARK KEEP_PROTECT TF_SPILL_STRIDE=8 SHARED_PLAN`.
  - It is identical on exact (cold only: p03's `--like` record found its prompts evicted; resumed identity still owed)
    and on sampled identity.
  - 3 rotated pairs against `best` (p03/p04, q-blocks; `runs/combo/phase4.md`): cold prefill is a wash, median about
    0 [-0.9..+1.6%]. `TF_SPILL_STRIDE=8` alone: one of two blocks (q06) ran low at every size, warm cell included.
    That looks like a whole-block state effect (the 3 cluster blocks, also stride 8, read normally). Unresolved, n=2.
  - Hit rate unchanged: `prefix_fanout` cached 186,368 tokens at every level (1/2/4/8) in all 9 q-blocks, and the
    `prompt_reuse` turns match base turn for turn (91/89/88%). The probes
    cannot show the cluster's win cases (edit, rewind, pressure); that probe gap remains.
- **Thinking at 4 streams.** 5 of 19 LEAD+map blocks read t4 below the base floor, one rep each. A follow-up ABAB
  (r01-r07: base/best ×3, 6 paired prompts each) measured best/base at a median **+0.4%** (per block +0.5 / +0.0 / +0.6%).
  The low tail is one prompt (rep 0, the one the `think` cell uses): -3.4% in 2 of 3 blocks. The other five prompts
  were neutral to +1%. There is no workload-level loss.
- **Cold exact** (`SPILL_GIB=0`): `best` IDENTICAL, and `best` + the cluster IDENTICAL, against a fresh base record.
- **`0122` spec sample.** An earlier, unrecorded pass (2026-11-15 08:27-08:59Z, `runs/ultrafast/status.log` on the
  head) measured sampled agg tok/s at 1/4 streams of 42-48/89-91, against production's 53-55/105-106. That is
  about 20% slower. Its `samp_test` gate is degenerate (one first-token id). Not stacked.
- **First-request `CudaFailed`.** Every restart with `TF_DS_DEVICE_ACCEPT=1` (21 of 21) skipped it. Every restart
  without it (29 of 29) hit it. This is a lead toward the root cause.
- **Benchmark candidate:** production + `best` (`LEAD=1 SCALES=map STRICT=1 TOKEN_CACHE_MB=32` + `0124`-`0127`), the arm the soak,
  cold exact and sampled gates ran on. `0124`-`0127` are harmless rather than wins; `DEVICE_ACCEPT` avoids the
  first-request failure.
  - Combo2 subsequently measured the cold first-request cost. Both mapped modes add roughly six seconds versus I/O.
  - Open at the end of this campaign: `0119` encode at 250K/500K (carried over from this session's owed list; no row for it here), the
    hit-rate win-case probes, and `0123` profile phases (`ROUND_PROFILE=2`).

## Open questions this should answer

- Is the read-ahead hiding the Engram I/O completely, or is the drive/CPU contention visible in prefill?
- Does the scale half really cost a page per row, and does warming remove it?
- How much do repeated n-grams buy on real prose and on code, and is a cross-chunk cache worth its memory?
- Are 8 decode threads on 5 cores a loss, and does moving the host work off the engine's cores help?
- Does `LEAD` beat the plain read-ahead on this recipe, and does it compose with `DEDUPE`/`FAST`?

## Ready to benchmark: patches `0129`-`0145` (the "not built" rows, now built)

The rows above that said "needs `0099`+" and the W1/W4/W5/W10 work packages of
[`docs/benchmark-gaps-and-moe-review.md`](docs/benchmark-gaps-and-moe-review.md) are implemented as `zig/patches/0129`-`0145`
(exported by `scripts/port-patches.sh` from engine `bench-prep` over `decode-step-structure`; verified: TF_REF + every
patch in filename order equals `bench-prep`). Pair results are in [combo2](runs/combo2/RESULTS.md), [W4](runs/w4/RESULTS.md) and [depth](runs/depth/RESULTS.md). 0132 rejected valid CPU IDs and 0133 failed at small batches. The learned allocator lost its screen.
The striped-AIO proposal remains unbuilt. The later 0163 reply-scan screen was neutral on 6,000-token replies.

Every row keeps the rule at the top of this file: one switch at a time, everything else identical. Per-row details,
defaults and rank scope are in [`runs/bench-prep/env-vars.md`](runs/bench-prep/env-vars.md); the arm driver is
[`runs/bench-prep/run_arm.sh`](runs/bench-prep/run_arm.sh) and the plan is [`runs/bench-prep/README.md`](runs/bench-prep/README.md).
The pinned production image already contains this series.

| Run | Flags (export; `start.sh` gives them to both ranks) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Warm scales (`0129`) | `TF_DS_ENGRAM_SCALES=map-warm` | first and second request after a restart against `map` and `io`, then steady-state prefill 8K/32K, decode 1/4, `engram_lookup_seconds`, `MemAvailable` low | `map-warm` holds `map`'s steady state (overnight: +3.5-8.8 %) **and** its cold -82 %/-41 % first requests are gone. Decides `map`'s hold |
| Decode threads (`0130`) | `TF_DS_ENGRAM_DECODE_THREADS=4` / `5` / `2` | prefill 2K/8K/32K, decode 1/4, `engram_host_ms` (see `0134`) | any gain at one depth without a loss at another: 5 fast cores, 8 decode threads oversubscribes |
| Pool threads (`0131`) | `TF_DS_ENGRAM_POOL=16` / `32` / `128` with `ENGRAM_AIO=0` | prefill 2K/8K/32K against `AIO=0` alone, `engram_lookup_seconds` | a gain against `AIO=0`; it is that path's tuning, not a win over the ring |
| CPU placement (`0132`) | `TF_DS_ENGRAM_CPUS=10-14` with `CPUSET=10-19` | prefill and decode at 1 and 4 streams, both ranks' `MemAvailable`, the engine's own thread CPU time | the only CPU-contention experiment: hashing/decoding/reaping off the engine's cores, no loss elsewhere |
| Read depth (`0133`) | `TF_DS_ENGRAM_AIO_BATCH=64` / `256` / `1024` | prefill 8K/32K, `engram_lookup_seconds` count+sum, `engram_host_ms` | a prefill gain outside the band; says whether the read wall or the reap churn is the limit |
| Shared cut on disk (`0135`) | `TF_DS_SPILL_SHARED_MIN_TOKENS=2048` | a 4-9K agent session: restart the pair between rounds, resume, the first turn's `cached_tokens`; spill counters (`short`, writes); disk churn in `SPILL_DIR` | the shared cut is restored after a restart instead of every follower re-prefilling it, with no `short`-refusal storm |
| Row cache retest (`0136`) | `TF_DS_ENGRAM_ROWCACHE_MB=64` and `256` (the W1 kill test), then production with no cache | drive past 500K and 1.5M uncached prefill tokens since the restart; dec1/dec4/dec8 and code 8K either side | the progressive regression (overnight: -4/-9/-17 % decode at 1/4/8 streams) is gone **and** the onset moves with the budget. If it persists, the cache is ruled out for good |
| Rehash helper (`0137`) | none (a fix); arm on the same session as `0136` | the same long-session cells; the spill tier's `points` on a disk-tier run | no timing claim: it is there so long runs stop degrading |
| TTFT decomposition (`0138`) | none (new fields) | the `queue_s` / `restore_s` columns of live traffic; `ttft_s`/`prefill_s` at 3 decimals | not a keep/drop row: it is what makes the 36 s median TTFT attributable |
| Fixed-rows driver (`0139`) + `0140` + `0144` | `TF_DS_FIX_K=3`, `TF_DS_EXPERT_BYTES=1`, `TF_DS_RDMA_PROBES=1` on `tf-dsv41-lanes --profile` | `step_ms` by rows at fixed depth with the per-round expert bytes beside `expert_prune`; the ring's stage/flight/copy; `pass_send_ms` | instruments: they decide W4 (bus vs trellis) and size items A-E; nothing to keep or drop |
| Prefill phases (`0141`) | `TF_DS_PREFILL_PROFILE=1` with `--profile` | `prefill_profile_ms` at 2,048-row chunks on 2K/8K/32K prompts and single chunks of 256/512/1,024 rows | instrument: the first prefill split at real chunk sizes (M6) |
| Pure-read probe (`0142`) | `tf-dsv41-probe --synthetic` and `--spec` on one layer's picked experts | GB/s for `distinct-sorted` / `distinct-shuffled` / `repeat-first`, prefetch off and on | compare with the MoE's own rate on measured bytes: ≈equal → bus-bound (M5 dead, byte model stands); ≥1.5x → trellis decode is the limit and M5 opens. Built and run in W4. Blocking byte counts paired with separate nonblocking MoE times give an estimated rate; eight streams use the four-stream byte factor |
| Learned allocator (`0143`) | `TF_DS_ALLOCATE_LEARNED=1` | prose, code and thinking-on agent prompts at 1/4/8 streams: accept, tokens/round, rows/round, tok/s; the `alloc` trail | ≥2 % at 8 streams with no loss at 1. Sequential class: `validate.py`'s concurrent check as well as `exact.py`. Watch the known risk — `observeCost` is fed by single-stream rounds only |

Protocol notes for this set:

- `0129`-`0133`, `0135`, `0136`, `0137` are identity-class: run `tools/exact.py` on each arm that lands in addition to
  `validate.py`. `0143` changes round composition: its gate is `validate.py` including `concurrent == solo`.
- Warm up once before any cold-sensitive cell: the pair's first request after a restart fails deterministically on
  recent images (pre-existing, `cuLaunchKernel: CUDA_ERROR_INVALID_VALUE`), and the retry passes.
- If the image build fails on `probe.cu` (`0142`), drop that patch and rebuild — it is a tool, nothing served uses it.
- Restore after each session: unset the arm vars, restart, and confirm `validate.py --compare` still passes against
  `runs/matrix/base.validate.json`.

## Ready to benchmark: second pass (`0160`-`0171`)

The patches of `docs/baseten-ultrafast-plan.md` "Second pass" plus the MoE candidates M3/M4/M5/M7 and the
zstd/Bloom checks of [`docs/benchmark-gaps-and-moe-review.md`](docs/benchmark-gaps-and-moe-review.md) §4.3/§5, as
`zig/patches/0160`-`0171` (engine `second-pass-int`; verified: TF_REF + every patch in filename order equals the
branch). The pair screen is recorded below. Per-row details and the plan→landed number map are in
[`runs/bench-prep2/env-vars.md`](runs/bench-prep2/env-vars.md); the manifest is
[`runs/bench-prep2/SERIES.md`](runs/bench-prep2/SERIES.md). The row numbers are the **landed** patch numbers; the
plan's own item numbers differ for the named set (plan 0176-0181 = landed `0166`-`0171`: token align, M3, Bloom, M5,
M4, M7), so read the `env-vars.md` map before quoting a number from the plan. The plan's ten unbuilt odds (top-k,
gather overlap, buckets, bf16 partials; greedy argmax, expert grid, discarded head, Markov trim; Snap decoder-ring
trim, restore lookahead) are numbered `0182`-`0191` in `todo.md` "second-pass follow-ups" - none has code, so none has
a row here.

Same rule as every row above: one switch at a time, everything else identical. Arms marked **concurrent** need
`tools/validate.py`'s concurrent check (the switch changes round composition or shapes), not `exact.py` alone.

| Run | Flags (export; `start.sh` gives them to both ranks) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Absorb reorder (`0160`) | `TF_DS_ABSORB_AFTER_PICK=1` | round wall / tok-s at 1 and 4 streams, `absorb_ms` 0.26-0.29 | the round wall moves outside the noise band (the plan's ceiling is under 1 %) |
| Engram submit early (`0161`) | `TF_DS_ENGRAM_SUBMIT_EARLY=1` | round wall, `engram_host_ms`, `engram_hash_ms` at 1/4 streams | the hash (0.98 ms/rank/round) leaves the critical path measurably |
| Load readers (`0162`) | `TF_DS_LOAD_READERS=4` then `8` | the `rank cache: … read whole in … s` line and `loaded in … s` at start | the start wall moves toward the disk's parallel headroom (the plan's step 0: is there any?) |
| Reply scan suffix (`0163`) | `TF_DS_REPLY_SCAN_SUFFIX=1` | rank 0 CPU and `MemAvailable` over a 32K-token streamed reply | CPU or memory moves; else measured-only (the scans are small) |
| Prompt-cache auto (`0164`) | `TF_DS_PROMPT_CACHE_AUTO=1` | `prompt_cache_held_bytes`, eviction counters, `MemAvailable` low | more conversations stay warm without tripping memguard (on the pair expect the 0.5 GiB floor — a machine-with-room arm) |
| Server knobs (`0165`) | `TF_DS_MAX_TOKENS_CLAMP=1`; then `TF_DS_BODY_LIMIT_MB=64` | counts of 400s/413s, reply lengths vs requested | clamping turns refusals into replies with no surprises |
| Token align (`0166`) | `TF_DS_TOKEN_ALIGN_MB=64` | `prompt_tokens_cached` on resent conversations (a retry/regenerate harness) | prefix-cache hits rise on resends; the plan's step 0: the share of turns whose ids diverge inside the previous reply |
| **M3** draft triage (`0167`) **concurrent** | `TF_DS_DRAFT_TRIAGE=conf:0.2` (sweep 0.05/0.1/0.2) | tok/s and the round wall at 1/4/8 streams, `validate.py` concurrent | worth ≥ half the ~6 % ceiling with the concurrent check clean; the probe study (95 % precision / 50 % recall at layer 13/19) decides the candidate's life |
| Bloom front (`0168`) | `TF_DS_ENGRAM_ROWCACHE_FILTER=1` | `engram_rowcache_{has,fast,lock}_total`, `engram_lookup_seconds`, decode wall with the row cache on | the locked probes and the wall drop; if the lock does not show, the filter goes back off |
| **M5** hot prefetch (`0169`) | `TF_DS_EXPERT_HOT_CACHE=hot:3` | `moe` ms and the round wall; then compare with `0142`'s probe GB/s | any `moe` movement; W4's verdict (bus vs trellis) keeps or kills the candidate. `decoded` is refused until the kit grows kernels |
| **M4** end state (`0170`) | `TF_DS_PROMPT_ENDSTATE=2` | a retry/regenerate/`n>1` harness: TTFT of the resend vs the cold turn; `tail_window` misses | the resend's replay TTFT (0.7 s at 100K) is gone; and the `tail_window` counter shows ≥ 2 % of requests are such resends |
| **M7** tail rows (`0171`) **concurrent** | `TF_DS_TAIL_BATCH=4` with `0170` on | warm-tail TTFT (the 128-row replay is ~0.15-0.2 s a turn), `validate.py` concurrent, exact.py's cold-delta | the replay goes; the batching step (several tails one round) is the follow-up its commit body designs |

Checks beside the arms (not switches): `just zstd-check` (one 1 GiB slice of the rank cache through `zstd -19`,
lossless-verified; under 3 % saving the trellis-recompression idea closes), and `0168`'s counters for the row-cache
lock question. Restore after each session: unset the arm vars, restart, confirm `validate.py --compare` still matches
`runs/matrix/base.validate.json`.

### Results 2026-11-17: second-pass screen (`runs/combo3/RESULTS.md`)

One block per arm on the best config (image `234f9080b326`, 0129-0171 + head-only build fixup `0171z` for compile
errors in 0163/0164/0165). All-off a01 passed exact 39/39; armed blocks passed validate. **No switch gained beyond the
control spread; best config unchanged; all `0160`-`0171` switches stay off.** Single-block signals: absorb (8K prefill -27%, unconfirmed),
hot cache hot:2/hot:4 (decode/think -2 to -5%; hot:4 prefill -11%), rowcache 64 (think -3 to -4%) and +filter
(prefill -11 to -13%), endstate/+tail (8K-only -21/-28%, unconfirmed). Neutral: submit-early, scan-suffix,
token-align, triage conf:0.05/0.2, load readers 4/8 (restart 121 -> 120 s), DEVICE_ACCEPT (speed-neutral; kept for
the first-request fix). cache-auto: budget stays at the 0.5 GiB floor on a Spark (by design). Endstate: an identical
resend still resumes at the grid point (6144 of 8.1K, 30720 of 32K) and armed logs contain planner BadOp. Exact-length lookup and off-grid KV keeps have separate rejection paths. See [resend evidence](runs/combo3/RESULTS.md). `prompt_reuse`'s 88% FAIL is pre-existing (same on the old production image).

## Ready to benchmark: the odds (`0182`+)

The plan's odds in plan order, as landed after the second-pass series: `0182`-`0186`, `0188` and `0190` are in
`zig/patches/`. `0189` is a host test (plan 0166's granted+slack stop already ships); plans 0164 and 0166 are closed, and
the original T15 arms for plan 0164 omitted the required expert cap. The [capped retest](runs/review-round/RESULTS.md#applied-cap-t15) used the pinned serving image. The only plan item without code is `0191` (restore lookahead);
numbers and gates: `todo.md` "second-pass follow-ups", `runs/bench-prep2/env-vars.md`.

Same rule as every row above: one switch at a time, everything else identical. Arms that `env-vars.md` calls
**sequential** need `tools/validate.py`'s concurrent check, and `0185` is LOSSY: its `exact.py` record is expected to
differ.

| Run | Flags (export; `start.sh` gives them to both ranks) | Measure | What would make it worth keeping |
| --- | --- | --- | --- |
| Top-k fast (`0182`) | `TF_DS_TOPK_FAST=1` | `a_idx` at layers 24-36 (where scoring is constant) and the round wall at 1/4 streams; a 100K+ prompt for the prefill half | `a_idx` drops outside the noise band and nothing else moves. The path reads the score row 3 times against 10, so a flat `a_idx` means the select is not read-bound and the row closes |
| Gather overlap (`0183`) | `TF_DS_PREFILL_GATHER_OVERLAP=1` | the chunk wall and tok/s at 2K and 8K prompts (prefill), the two Engram layers' `engram` phase, and `MemAvailable`; then 4 streams | the chunk wall moves outside the noise band. Only the two Engram layers have anything to run beside their gather (~70 ms of host read a layer against a measured 8.31 ms an exchange), so a flat wall closes the row: the other 38 gathers have no slack |
| Finer buckets (`0184`) | `TF_DS_IDX_BUCKET_FINER=1`; then with `TF_DS_TOPK_FAST=1` | `a_idx` at ~65,000 against ~66,000 tokens (the bucket step) and at a 70K context; `MemAvailable` (more captured graphs) | `a_idx` drops where the bucket used to jump and the select's extra launch does not eat it. Without `0182` the finer bucket leaves the fused select, so the pairing matters |
| bf16 partials (`0185`, **LOSSY**) | `TF_DS_BF16_PARTIALS=1` | `gather_a`, the chunk wall and tok/s at 2K/8K prompts, 1 and 4 streams; the reply's bytes (it is lossy) | `gather_a` drops and the chunk wall moves; the plan expects little, so a flat wall closes it. exact.py will differ: record the delta and gate on validate.py |
| Argmax exchange (`0186`) | `TF_DS_GREEDY_ARGMAX=1` | the `head` phase and the round wall at 1/4/16 streams; `exact.py` IDENTICAL | the `head` phase drops and the round wall follows outside the noise band. Nothing else reads logits on a greedy round, so a flat head closes it (the gather is 258,560 B a row a rank against 8 B) |
| Head last chunk (`0188`) | `TF_DS_HEAD_LAST_CHUNK=1` | prefill tok/s and the chunk wall at a prompt length just past a chunk multiple (so the window straddles), 2K/8K | the chunk wall moves; the plan calls it "a few ms a request", so treat a flat wall as the finding |
| Snap trim (`0190`) | `TF_DS_SNAP_TRIM=1` with the spill tier (needs `TF_DS_REPLAY_FLOOR=1`, the default) | `prompt_cache_held_bytes`, the eviction counters and `MemAvailable` over a session of keeps; `exact.py` across a keep and a resume | more conversations stay kept in the same memory budget, and the restore/LOCAL path is exact. The first enable makes the existing store's points foreign: note the one-off loss |
| Place bound (`T15`, patch 0053) | `TF_DS_T13_EXPERT_PRUNE=1 TF_DS_T13_EXPERT_CAP=12,1`, toggle `TF_DS_T15_PLACE_BOUND=0/1` | Completed with eight fixed rows, graphs on and 127 shared decode rounds per timed wave | Median decode 4.18/4.18/4.17/4.19 s off/on/on/off. The active path gave no repeatable material gain at this shape. Component-level MoE timing and the quality of the cap versus production are separate measurements. |

On a repaired experimental image excluding 0190 until its default path passes smoke, run `start.sh restart` + one chat completion, `tools/exact.py` (IDENTICAL except `0185`, whose delta is the
point), then `tools/validate.py --compare`. Restore after each session: unset the arm vars, restart, confirm
`validate.py --compare` still matches `runs/matrix/base.validate.json`.

### Results 2026-11-17: the odds screen + combination search (`runs/combo4/RESULTS.md`)

Image = combo3 set + `0182`-`0189` + two head-only bench fixups (`0184z`: Round.init never sets `rd.hot`, segfault
once `0184` changes the layout; `0186z`: forward declaration of `tf_ds_block_argmax_v`, or `0186` does not compile).
`0190` excluded: illegal address on the first request at default flags. With every switch off: exact IDENTICAL,
validate passed, sampled shas unchanged. **Best config unchanged.**

- Top-k screened neutral. Phase D reproduced the exact failure for finer alone and with top-k. The T15 screens omitted the expert cap on both ranks, so the grid bound remained inactive. Gather overlap lost ~0.8% at 32K in the repeated comparison.
- Broken: `BF16_PARTIALS=1` and `GREEDY_ARGMAX=1` give a garbage smoke reply at startup; `HEAD_LAST_CHUNK=1` hangs
  the rank pair on an ~8K prompt.
- Combos (4 alternating passes vs 8 best blocks): the stack of neutral host switches (map-warm, ENGRAM_FAST, CPUSET,
  SUBMIT_EARLY, SCAN_SUFFIX, TOKEN_ALIGN_MB=16, LOAD_READERS=4) costs 0.4-0.8 % decode in every block, with or
  without triage; best + the four odds is flat on decode, ~0.7 % lower steady 8K prefill, and its exact run gets
  HTTP 500 MissingTritonVariant (`_index_score`), after which every request gets CudaFailed.
- Phase D (`runs/combo5/`): finer alone reproduces the crash (so it is `IDX_BUCKET_FINER`), though finer(+topk) was the
  only prefill hint (+0.5-0.9 %, 1 block). best + TOPK_FAST with inactive T15 is IDENTICAL and flat; adding GATHER_OVERLAP costs
  ~0.8 % pd.32K in every block. The stack's decode loss lives in its Engram half (map-warm + ENGRAM_FAST +
  SUBMIT_EARLY: ~1 % dec8); the host half is neutral. No tested combination beats best.
