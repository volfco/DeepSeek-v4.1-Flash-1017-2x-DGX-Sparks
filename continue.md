# Continue

**Current state (2026-11-26 19:47 UTC): production runs image `tensorfold-dsv41:bench-fused-next-1421187` (= `2bb052f` + 0280–0283) with `TF_DS_MARKOV_CACHE=4096`, `TF_DS_DRAFT_PREFETCH=1` and, since 2026-11-26, `KV_SHARD=0` (pool 1,048,576 positions; `PREFILL_SLICE=5` now in effect), plus `TF_DS_TAPS_PACKED=1`, `TF_DS_IDX_PICK_WIDE=1`, `NCCL_MAX_CTAS=4`, `KV_FETCH_REUSE`, `CONF_PACK`, `TOPK_MERGE`, `PLACE_BOUND_ALL`, `DRAFT_BLOCK=6`, `DRAFT_BLOCK_KEPT=2`. KV_SHARD=0 deployed 19:47 UTC: exact 39/39, validate rc=0. Undo chain: `.env.bak-20261126-pre-kvshard0` → `.env.bak-20261125-pre-mwb` → `.env.bak-20261123-pre-2bb052f`.**

**CUDA kernel process & GB10 hardware reference:** Follow [the CUDA kernel compilation, block microbenchmarking, and opcode evaluation process](docs/cuda-kernel-microbenchmark-process.md) for any new or modified CUDA kernel (`sm_121`, `--fmad=false --ftz=false`, byte-parity & sanitizer gate, tiered CUDA-graph/ablation → `clock64()`/`%globaltimer` → `ncu` SASS stall attribution, and opcode optimality criteria), backed by [the GB10 (`sm_121`) hardware performance & kernel design guide](docs/gb10-hardware-performance-guide.md) (measured ceilings, opcode behaviors, counter quirks, and bit-exactness rules).

## KV_SHARD=0 in production; live review of 1421187 (2026-11-26)

[Review](runs/observe-20261126/REPORT.md) · [Deploy](runs/deploy-kvshard0-20261126/)

- **Review (19:29 2026-11-25 → 04:22 2026-11-26 UTC, before the switch).** Decode is on spec: one-stream rounds +6–8%
  over the line (as in E2), and 5–8-stream rounds with prompt passes removed are 86–93 ms against `dec8` ~87 ms.
  The 8-stream figure is off 2.3× (212.6 ms a round, 68–70 tok/s), and that comes from the workload: a closed loop
  of about 21 outstanding requests, each needing ~1.9K uncached tokens, with prompt passes taking 60–64% of the GPU.
  Interleave only happens between 2,048-row chunks, so 62% of passes never yield. The GPU is serialized, so making
  passes yield moves latency, not throughput.
- **Dashboard:** the six prompt and generation tok/s panels show `sum(tensorfold:requests_running)` on a right-hand
  axis (`runs/observe-20261126/scripts/dash_running.py`; undo restores `data/tensorfold-dash-before.json`).
- **Switch (operator request):** `KV_SHARD=1` → `0`, applied by a watcher after 90 s of zero traffic (fired
  19:41:55, done 19:47:24 UTC). Exact **IDENTICAL 39/39** against the sharded reference, so unsharded output and
  slicing on are bit-identical; validate rc=0. Turning sharding off also turns `PREFILL_SLICE=5` back on.
  MemAvailable at 22:04: 6.3 GiB head, 7.8 GiB worker.
- **Cost, still to measure:** the pool is 1,048,576 positions, against a 1.59M-row peak on 2026-11-26 under sharding.
  Long kept conversations will be evicted sooner. Docs put sharding's decode cost at 1–4%.
- **Open:** see the todo.md entry for 2026-11-26.

## Prefill/decode interference measured (2026-11-25)

The first cell with a cold prefill beside decoding streams. 7 decoders at 8K depth, one 131,072-token cold
prefill launched 1.5 s into their decode, 9 knob pairs + a no-prefill control, all exact IDENTICAL. [Report](runs/mixed-20261125/REPORT.md)

- **One cold 128K prefill costs the decoders 62% of throughput** at production's 250/5: 9.37 vs 24.40 tok/s
  per decoder, 251 vs 95.5 ms per round. That is the first direct measurement of the live interference and
  it sits at the top of the 1.7–3.4× band the live logs implied.
- **`TF_DS_PREFILL_SLICE` is inert.** 5/10/20 identical at every interval.
- **`TF_DS_PREFILL_INTERLEAVE_MS` is a priority dial, not a win.** 250 ms keeps the prefill at 1662 tok/s
  (81% of its 2,040–2,100 solo) and starves decoders (49% of solo); 1000 ms reverses it (decoders 13.27
  tok/s, 70% of solo; prefill 1231 tok/s, 60%). 500/5 is dominated: prefill 1367 (18% worse than 250) with
  decoders 9.24 (no better than 250's 9.37).
- **The knob cannot fix the loss.** Even the best arm leaves decoders 46% below the no-prefill control, and
  decoders' `cached` stayed 6144 in every arm — so the loss is structural bandwidth/queue contention, not
  scheduling granularity and not cache eviction. A structural fix (eviction reservation / lookahead 0191,
  or not co-scheduling cold prefills with decodes) is the only lever left.
- **Adopt nothing.** Production's 250/5 is prefill-optimal and correct while TTFT binds (422 s mean, 113/153
  clients gone at 600 s). 1000 ms only makes sense for a decode-heavy queue with TTFT headroom.
- **Open:** the 250→500→1000 decoder curve is flat-then-jump (9.37 / 9.24 / 13.27) and unexplained; it needs
  the traced attribution the [drafter-degradation report](runs/drafter-degradation-20261124/REPORT.md) §3
  already asks for.

## k296 recorded (2026-11-25): a suite that ran on 2026-11-24 and was in no doc

`runs/k296/` (2026-11-24 01:09–02:06 UTC, image `bench-fused-next-87e92a2` = 1421187 + 0294–0296) had a full RESULTS.md and six
arms of artefacts but appeared in no document, and it contradicted four open items. Now folded into `todo.md`:

- **0295 `CONF_LEADER_ONLY`: exact, neutral — adoptable under the exact-and-neutral rule.** Its saving is follower GPU work,
  invisible to any rank-0 metric.
- **0294 `ROUND_MS_FIT`: no win** (probe −0.15%, one-stream forward +1.5 to +3.3%). **`DRAFT_MS`: inert** at the served
  shapes. **And the knob's unit is per-head-step (~0.65 ms at one stream), not the whole pass** — every reader multiplies
  it by drafts/steps/held_levels, so the staged `draftms39`/`draftms72` arms would have been ~5× too dear.
- **0296 union-fetch gate: FAILED** (`distinct/picks` 0.582 at 32K, 0.612 at 128K in both controls, above the ½ rule) —
  do not brief the union fetch. The counter stays as instrumentation.
- Also useful: its control arms prove the cross-built image's off path is exact, which retires the "the new executables have
  never been executed" doubt the 2026-11-23 staging note carried.
- **Process lesson:** a suite whose verdicts live only in its own directory is lost. `runs/k296/` was untracked and
  unrecorded while `todo.md` still said "awaiting the window" for all three patches.

## M3 draft triage closed: no signal (2026-11-25)

The largest unmeasured bit-exact decode lever in the record, now measured and **retired**. M3's step 0: can a linear probe on a
layer-13 or layer-19 hidden state predict "this draft row will be rejected" at 95% precision / 50% recall? [Report](runs/m3-20261125/REPORT.md)

- **No.** Layer 13 AUC **0.621**, layer 19 AUC **0.651**, against a free positional baseline of 0.588 / 0.590
  (~40k draft rows, honest round split). **No threshold reaches 95% precision at either layer**; even a leaky row
  split reached it only at 0.2% recall.
- **Why:** the target is a sampling outcome at a *different* row (row r dies when row r−1's pick misses its draft),
  and row r's hidden state does not encode row r−1's draw. Rows after the first break are dead by construction,
  which is what position alone captures. So the existing `conf:` arm's −0.5% was never a probe-quality problem.
- **Consequence:** do not build the `file:` probe path; `0177`'s dead-row mechanism stays built and unarmed;
  production unchanged. The decode lever is acceptance (workload-dependent) and contention, not drafting.
- Kept as a reusable instrument (off by default): engine branch `probe-dump` (`27090ee` + `535b40d`), image
  `bench-probe-dump`, `runs/m3-20261125/{build.sh,capture.sh,capture.py,train.py}`. Two traps recorded there: the
  startup smoke test opens the dump before the workload (never `rm` it after a restart), and `find -xdev` skips the
  `/spill` bind mount.

## Staged for the next hardware window (2026-11-23, two-lane campaign — NOT on hardware)

Two delegated lanes finished 2026-11-23 with seven patches staged **out of series** (they are based on `1421187` =
`TF_REF` + `0001`-`0283`, but the exported series stops at 0264, so they must not be appended to `zig/patches/` yet —
`zig/Dockerfile` applies that directory in filename order). Records: [tier0](runs/lane-tier0-20261123/REPORT.md) ·
[sched](runs/lane-sched-20261123/REPORT.md). Decision: `decisions.md` 2026-11-23.

**Tier 0 (blockers).** `runs/lane-tier0-20261123/patches/`:

| Patch | Switch | What | Window gate |
|---|---|---|---|
| `0297` | `TF_DS_KEYS_PROFILE` | the pruned selection's width + key-encoding class (`kit`/`fold`) on a `key_profile` line | Stage A: unfixed `TF_DS_IDX_FUSED` bundle → expect 4/39 DIFFERENT and `fold` records only inside 16,384 < width <= 65,536 |
| `0298` | none (dead until parked 0206 adopts it) | `kitKey`/`key_kit_flip`: the key fold writes the kit's `_score_keys` (signed-order) encoding | Stage B: fix folded into re-rolled 0206 → 39/39 IDENTICAL + `probe.py` keys **and** `tmax` byte parity vs the kit |
| `0299` + `0299z` | rides `TF_DS_ROUND_PROFILE`/`TF_DS_CLOCK_MARKS` | `clock_fold.zig` host fold + clk-init pins; 0299z fixes the round count above the degenerate-buffer guard | graphs-on profile reads nonzero phases |
| scoring | — | per-round distributions (`work.py` samples; `table_gpu.py` mean/median/p95/p99; `arms.sh` `*.frames.txt`) | every future suite scores this way |
| `runs/k280/window.sh` + `just k280-window` | — | the staged 0280-0283 suite, README order, gated on `restored: validate rc=0` | run with production idle (~3.5 h) |

**Scheduler cluster.** `runs/lane-sched-20261123/patches/`:

| Patch | Switch | What | Note |
|---|---|---|---|
| `0294` | `TF_DS_DRAFT_MS`, `TF_DS_ROUND_MS_FIT` | the drafter allowance knob (unset keeps 3.5) + the trace-fitted round table (`34.656 ms @ 6 rows + 3.1536 ms/row`); `chooseK`'s clamp fixed to the tail slope | measure knob and re-fit **apart**; J1: wide-round (>= 24-row) replay expectations shift |
| `0295` | `TF_DS_CONF_LEADER_ONLY` | the follower skips the drafter confidence head (output-invariant) | gate: reply/rounds/accepted equal + follower draft IDs, depths, stashed rows identical |
| `0296` | rides `TF_DS_ROUND_PROFILE` | distinct picks by depth per fetch | brief the decode KV union fetch only if `distinct/(rows*512) < 1/2` at 32K/128K |

**0293 produced no patch** — the drafter block-row clamp is **not** bit-exact (non-causal block attention; rows past
`steps` feed the scored rows). Its approved safe subset is already 0265 `TF_DS_DRAFT_HEAD_SCORED`, whose commit
`c04f020` is not an ancestor of `1421187` (series call). Operator declined the full clamp 2026-11-23.

**Test state:** both lanes' full `zig build test` green (the first attempts' failures were the known load-flake family,
verified green standalone: 75/75 lanes/stream, 23/23 tokenizer, 530/530 dsv41); focused tests green for every patch.

## Status summary (2026-11-23 05:30 UTC)

**Production:** `bench-fused-next-2bb052f`, which is recipe-0264 `27baa93` + 0272–0275 + 0278 + 0279. The `.env` switches are `KV_FETCH_REUSE`, `CONF_PACK`, `TOPK_MERGE`, `PLACE_BOUND_ALL`, `DRAFT_BLOCK=6`, `DRAFT_BLOCK_KEPT=2`, `TAPS_PACKED`, `IDX_PICK_WIDE` and `NCCL_MAX_CTAS=4`. The exact reference is `runs/deploy-10c9866-20261122/exact.json`; every deploy since has been IDENTICAL against it.

**Adopted overnight (Nov 22–23), all exact 39/39:**

| When (UTC) | Change | Gain | Record |
|---|---|---|---|
| Nov 22 13:49 | `DRAFT_BLOCK_KEPT=2` | decode +0.3% | `runs/overnight-20261122/kept/` |
| Nov 22 21:25 | image 10c9866: 0273 capture-join fix, 0274 tool-call first-token fix | bug fixes | `runs/deploy-10c9866-20261122/` |
| Nov 23 02:22 | `NCCL_MAX_CTAS=4` | cold prefill +0.8% | `runs/overnight-20261122/nccl/` |
| Nov 23 05:20 | image 2bb052f with 0278 `TAPS_PACKED` and 0279 `IDX_PICK_WIDE` | prefill +0.5%; 0279 neutral (>262K decode) | `runs/k278/` |

**Rejected or held (exact unless noted):**
- 0265, 0266, 0268, 0269 and 0270: neutral.
- 0267 `MARKOV_FUSED`: −6 to −10% greedy.
- 0271 `RESUME_ASSEMBLE`: inert, since it needs lossy `EXACT_RESUME`.
- 0272 `RING_TRACE`: a diagnostic.
- 0275 `COPY_MIN`: neutral.
- 0276 `CACHE_REPLACE_STALE`: inert; every kept state has a spill point.
- 0277: not built (kit cubin).
- `SOLO_OVERHEAD`: neutral.
- Drafter D1: −2.4% decode.
- NCCL `PROTO`/`CHANNELS`/`BUFFSIZE`/`QPS`/`CHUNKSIZE`/`MIN_CTAS`/`MAX_CTAS` 2 and 8: none better than `MAX_CTAS=4`.
- `ENGRAM_SCALES=map` and `GREEDY_ARGMAX`: slower.
- M1 measured `TF_DS_ROUND_MS`: −0.24%.

**Root causes and findings:**
- The k272 rank-1 wait was the harness's rank-0-only `ROUND_PROFILE`. The real residual is ~0.4% at 1 stream, from L2 prefetch helping rank 0 more (`runs/sched-20261122/b2/`).
- Multi-turn cost is the 2,048-row resume-grid remainder, about 1.1 s TTFT per ~8K turn (`runs/cache-20261122/`).
- Spill restore after a restart is 1.5 s for 300K and 2.7 s for 600K, against 152 s / 332 s cold (`runs/k278/`).
- NCCL carries only all-gathers and send/recv, so its knobs can't move bits.

**New at 06:30 UTC:** the [audit review below](#audit-review-and-re-score-of-rounds-02590279-2026-11-23-0630-utc) corrects several round-time records above (0264, k265, k269's noise floor) and adds four candidates. Read it before trusting a `forward_ms a round` number.

**Open, in rough priority order:**
1. **Pool-pressure cache replay.** About 100 distinct ~22K documents fill the 2.1M-position pool, then revisit the early ones. It exercises staleness and spill-restore under memory pressure, and times restore against recompute. `runs/cache-20261122/replay.py` with `K=100 WORDS=16000 R=2`.
2. **`top_k` -1/0/>56 with concurrent sampled streams:** correctness and slow-path check.
3. ~~**More than 4 concurrent streams** (8, 16): total-throughput scaling; no suite has gone past 4.~~ **CLOSED 2026-11-25 (capacity window): the box saturates at 8 streams.** 8→13→16 streams gives 98.4→104.8→108.8 tok/s aggregate (+10.7%) while per-stream falls 12.3→8.1→6.8 tok/s and TTFT rises ~2.4×. Round cost is row-bound, so extra lanes convert to extra rows rather than throughput. `ROUND_ROWS` 32/48 at 8 streams is neutral/−10% — the "drafts 2–5 are free" premise is false. Two constraints found: `DRAFT_BLOCK=6` caps the server at 13 streams (`max_rows=80`, `model.zig:565`), and 16 streams needs `DRAFT_BLOCK=5`. The live 422 s TTFT with 35 waiting is therefore not fixable by raising `PARALLEL`; the remaining lever is acceptance rate (M3 triage). [Capacity report](runs/capacity-20261125/REPORT.md) (Correction, 07:27 UTC: the November 21–22 `run_suite.py` suites have a `dec8` cell, and live traffic ran 8 streams with 35 waiting on November 23; see "OpenObserve review results".)
4. **Operator decisions (not exact against the current reference, per decisions §0121):**
   - 4,096-row prompt passes (bertholomus, +3–10% prefill);
   - `TF_DS_EXACT_RESUME` for multi-turn (~1 s TTFT a turn).
5. **k280 measured 2026-11-25: adopt 0281's Markov cache at 4,096 rows and 0282's `dpf1` prefetch — after a from-source control.** Both clear their gates within the cross-built image: 0281 `pass_ms` −0.20/−0.22 ms (−5.0%, miss 52.2%→24.2%, both controls, drafts IDENTICAL 19/19); 0282 `dpf1` −0.18/−0.17 ms (−4.2%, no forward cost). 2048 rows misses the 0.15 ms gate at −0.13/−0.14 ms but is 6.7× the control spread; `dpf2` is a wash on forward+pass and should not be adopted. **The blocker is the measurement's own caveat:** the image's executables are cross-built while production's came from a GCC 13.3 from-source build, so a win against a slower in-image control may not transfer. Either add a from-source control arm or rebuild 0281/0282 from source before deploying. [main report](runs/k280/main/REPORT.md) · [passprof](runs/k280/passprof/)
6. **Low value, parked:**
   - per-rank prefetch budget (≤0.4% at 1 stream);
   - deferred 0266/0267 microbenchmarks;
   - idx selection past 65,536 blocks (hierarchical).

**Housekeeping:**
- `todo.md` is stale on 0195 small-row/`max_nbk` (done) and the sampled top-k exchange (that's 0261 `TOPK_MERGE`).
- `scripts/config.sh` has a `DATA_SUFFIX` hook (no-op unset). The head's copy also carries an uncommitted CPUSET block.
- Local commits since `25581bd` are not pushed.
- Uncommitted edits to `todo.md` and `runs/suite-fused-next2-20261121/query128/bench.py` predate this session; they were left alone.


## Audit review and re-score of rounds 0259–0279 (2026-11-23 06:30 UTC)

A review of [the fused-kernel audit](docs/fused-kernel-rebenchmark-audit.md) and of every suite since 0259, from the saved runs and the engine worktrees. Nothing was run on the pair. **Verdict: no large kernel win was missed. The patches remove microseconds from a 42.8 ms cycle, and the suites' round-time metric had artifacts that produced the apparent wins and losses.** Four candidates came out of it (round 0280–0283, below).

### A decode cycle, and what each patch could remove

- **Cycle identity (measured).** One-stream cycle = `forward_ms` + one drafter `pass_ms` (+ `absorb_ms` 0.26, `logits_ms` 0.05). k265 ctl1: 38.27 + 4.17 ms. The sampled probe reads 42.55 ms a round (tetris, `decode_s` / rounds). `draftFn` runs the pass synchronously before the next verify (`lanes.zig:1823–1828`); passes ≈ 1.006 × rounds. Nothing else is hidden: no row leaves a positive residual.
- **The drafter pass is absent from the stage ceiling table below** (that table covers the forward only). It is about 10% of a one-stream cycle and serial.

| Patch | Where it runs | Best case a cycle | Basis |
|---|---|---|---|
| 0259 `KV_FETCH_REUSE` | 30 of 38 exchanges a round | 0.16–0.71 ms realized | measured; wall −0.3 to −1.6% in all six cells |
| 0260 `CONF_PACK` | once a pass, in the pass graph | 1.8 µs | 4 launches × 0.457 µs measured in-graph node floor |
| 0261 `TOPK_MERGE` | once a sampled round, head stretch | not recorded | no selector timing; the whole head stage is 1.16 ms |
| 0264 `PLACE_BOUND_ALL` | once a layer | 0 at 1 row, R − 1 places a layer | not timed alone |
| 0265 `DRAFT_HEAD_SCORED` | once a pass, only when steps ≤ 4 | 9–18 µs at one stream | estimate; its 9 µs a row slope rests on row counts that are not in `p2g0.json` |
| 0266 `COMPRESS2_PAIR_ONCE` | **3 layers a round** (2, 8, 14), not 40 | ≤ 0.3 µs | `comp_wkv` layers only (`weights.zig:659`) |
| 0268 `DRAFT_QKV_NORM` | 3 a pass | 1.4 µs | launch floor |
| 0269 `DRAFT_TAPS_DIRECT` | once a round, eager absorb | ~5 µs | derived; all of `absorb_ms` is 0.26 ms |
| 0270 `PROMPT_PICKS_ONCE` | chunks with `kk % 16 != 0` | **0 on a 2,048-row chunk** | `kk` is 512 once a chunk ends past 1,024 tokens |
| 0271 `RESUME_ASSEMBLE` | never | 0 | `EXACT_RESUME` is refused at startup with spill, `KV_SHARD` and `PREFILL_SLICE`, all armed |
| 0277 (not built) | 2 launches a round | 6–10 µs | estimate |
| 0278 `TAPS_PACKED` | once a prompt, decoder chunk | ~44 µs a prompt | derived; see "Unexplained" |
| 0279 `IDX_PICK_WIDE` | once a round past 262K tokens | 1–4 µs | estimate |

### Scoring artifacts in the `arms.sh` suites

1. **`forward_ms_per_round` cells are quantized.** `workload.py:84,99` differences a lifetime mean that the engine prints to 0.01 ms, times lifetime rounds. From k265 on, the probe runs before the workload (~6–7K lifetime rounds), so a 97-round cell is off by up to ±0.6 ms.
   - k265 4K×1 on the host metric: controls 33.88 / 33.87, 0265 33.30 / 33.26, 0268 33.29 / 33.17 (the "−1.75% / −1.89%").
   - The same cells on the GPU clock (`tail` alone): 33.45 / 33.44, 33.48 / 33.46, 33.40 / 33.24.
   - Identical-code pairs over 48 one-stream cells: p90 2.06% on the host metric, 0.64% on the GPU clock (medians 0.33% / 0.28%). k269's "±2–3% a cell" noise floor is mostly this artifact.
   - **Use `per_round.tail[0]` (µs; the forward's stretches on the GPU clock), and read `wait[0]` separately as host stalls. Both are already in every arm's cell JSON since the recipe lineage. `runs/overnight-20261122/table_gpu.py <OUT> --control a,b --test c,d` scores a suite this way** (one-stream cells a round, multi-stream cells as total forward and wall, then `pass_ms` / `absorb_ms` from the profile lines; it falls back to the host metric on images with no clock). The note further down that the clock-mark profiler is broken is about per-phase marks, which this review did not check; the stretch-level `wait` / `tail` values are sane (lifetime `tail` + `wait` = `forward_ms` − 0.02 ms).
2. **Per-round time at four streams is confounded by round count.** The same 1,280 tokens take 149–157 rounds between control arms at 128K×4, and each extra round costs about 30 ms of total forward. Compare `rounds × tail` or `wall_s`.
3. **Probe tok/s is stepped.** `suite_probe.py` divides tokens by the server's `decode_s`, printed to 0.1 s: 0.6% steps on a 16 s row, 1% on a 10 s row. That is why 128.0 repeats across eight arms. The client-timed probe (k269 onward) avoids it.
4. **Forward metrics cannot see drafter-pass patches** (0260, 0265, 0267, 0268). The profile line's `pass_ms` can, at 0.01 ms.
5. **Real one-off events remain.** Host stalls of 70–100 ms show as `wait` (k269 ctl1 4K×1 992 µs a round, r271a 32K×1 776, nccl mc4a 4K×1 703). Whole cells run 0.5–0.9 ms a round slow on the GPU clock with no code change (kall all3a 4K×1, k272 ctl2 4K×1, k272 128K×1 in ctl1 / ctl2 / tr16a, sweep1 gargmax 128K×1). Cause not found. A per-round median would remove both; only means are saved.
6. **Evidence gaps.** `arms.sh` saves banner counts, never the text, so no first-engagement shape is on record. The banner counts in `k260-gate` and `placebound` are 0259's banner (the default pattern), present in controls too. Every arm runs with `TF_DS_ROUND_PROFILE=1`.

### Re-scored results (GPU clock at 4K / 32K / 128K × 1, test vs control mean; `pass_ms` test vs control)

| Patch | Forward, GPU clock | Pass or other direct metric | Reading |
|---|---|---|---|
| 0264 | +0.88 / +0.17 / −0.19% | wall over six cells +0.25% | **neutral, not −0.41%** |
| 0265 | +0.06 / −0.10 / −0.23% | pass 4.15 / 4.11 vs 4.17 / 4.14 | −0.025 ms a pass |
| 0266 | −0.03 / +0.01 / −0.27% | pass 4.17 / 4.14 | nothing |
| 0268 | −0.38 / −0.01 / −0.39% | pass 4.15 / 4.12 | −0.02 ms a pass |
| 0265+0266+0268 | +0.10 / +0.05% (4K cell hit a slow episode) | pass 4.13 / 4.14 vs 4.16 / 4.14 | −0.015 ms a pass |
| 0267 | +0.69 / +0.70 / +1.06% (k267); +0.73 / +0.87 / +0.56% (kall all4) | pass 6.33 / 6.34 vs 4.20 / 4.20 | see "Unexplained" |
| 0269 | −0.20 / −0.13 / +0.19% | absorb 0.25 / 0.25 vs 0.26 / 0.26; the A/A arms also read 0.25 | nothing |
| 0270 | −0.15 / +0.38 / +0.45% | `prefill_ms` 724.0 / 725.5 vs 725.3 / 725.9 | nothing |
| 0271 (A/A) | −0.33 / +0.41 / +0.06% | n/a | the resolution of this table |
| `GREEDY_ARGMAX` | −0.01 / +0.11 / +0.91 ms a round | n/a | the recorded "+1.2%" is one 128K cell |

Not covered: 0260 (no profile lines saved in `k260-gate`), 0261 (no on/off run on sampled traffic exists), and the four-stream cells (round-count variation).

### Corrections to earlier records

- **0264 was adopted on a confounded number.** Its "−2.15% at 128K×4" compares on arms at 156 / 156 rounds with off arms at 148 / 151. Total forward was +2.1% and wall +1.2% (15.92 / 15.91 vs 15.63 / 15.81 s) with the switch on. It is exact and neutral, so it stays under the adopt-neutral rule; do not cite a gain. [Arms](runs/overnight-20261122/placebound/)
- **k265's −1.8% at 4K×1 never existed**, so kall did not fail to reproduce it.
- **0259 is real** by wall in all six cells, about 1%.
- **0266's launch saving** applies at 3 layers a round; report-0266 says 40.
- **0270's "76 → 16 launches a chunk"** describes a cold 127-token chunk. A 2,048-row chunk skips the pad in both arms.
- **0267's launch count** omits the cross-rank gather in every Markov step (`tri_markov.zig:162–165`).

### Unexplained

- **0267, the forward got slower too.** The greedy loss is +2.75 to +3.66 ms a round on every probe row. About 3 ms of it is outside the forward, in the drafter pass (lifetime `pass_ms` +2.13 ms is diluted by sampled passes, where the arm does not engage). The other 0.2–0.4 ms is in the forward that follows: its GPU-clock time is above both controls in all 12 one-stream arm-cells of k267 and kall, against a 0.28% identical-code median. So a change inside the pass moved the next forward by about 0.25 ms. Cause unknown; cache state at the pass-to-forward boundary is the hypothesis (it is the only evidence for candidates 3 and 4 below). The penalty does not track drafts verified (prose 1.48 drafts a round, +3.4 ms; go 5.79, +3.2 ms), so it is per pass. [Probe rows](runs/overnight-20261122/k267/)
- **0278, the gain is about 1,400 times the work removed.** The whole patch is the `tap()` write stride plus dropping three `copyRows` and reading `ch.taps` in `absorb`, all on the decoder chunk, once a prompt (encoder-only chunks return at layer 20, `model.zig:1626–1636`; taps are layers 37–39). That is about 44 µs. Measured: TTFT at 32K 15.489–15.504 s against 15.531–15.586 s in the eight control-equivalent runs (−63.5 ms, both test arms above all eight), and at 96K −102 ms and −434 ms (the two test arms differ by 332 ms). Something may be real, but it is not the stated mechanism. Worth a prefill phase profile with the switch on and off before quoting "+0.5%". Engine commit `a8c95b7` in `~/Workspace/tfref-kstack4` · [Arms](runs/k278/)

### Amendments to the audit document

57 of its 62 line-level source claims check out against the worktrees and none is wrong.
- **Ranking.** Its three "first hardware window" items have ceilings near 0.1% (0261 early exit, an estimate: no selector timing exists), 0.01–0.05% (0269's two 384-byte uploads) and nothing at serving level (0198, 2.6–3.3 µs a call on 128-row prompt chunks).
- **0261.** The fallback draw kernel (`dsv41_ops.cu:1570–1602`) runs the same eight passes over 129,280 columns on rank 0, so an early exit belongs in both arms. The top-k exchange is an NCCL all-gather (`round.zig:2191`) while the full-logit exchange takes the RDMA ring up to 17 rows (`prompt.zig:153`), so fewer bytes need not mean faster. **Worth doing from the audit's design: shipped 0261 against the fallback on sampled requests, scored on the GPU clock. It needs no build and has never been done.**
- **0269.** The absorb synchronize runs every round in the served order (`TF_DS_ABSORB_AFTER_PICK` is off), not only under the profiler. The three-view proposal needs a 16-byte-aligned `SLOT_OF` pointer; a view starting at an odd row is not, and the archived kit manifest has no unaligned variant.

### The drafter pass (4.11–4.18 ms at one stream; breakdown measured 2026-11-23, attribution 2026-11-24 — see "Drafter degradation attribution" below)

- **Structure** (`draft.zig:570–714`, `tri_markov.zig:120–165`): three or four small uploads, one graph replay, a synchronize and two device-to-host copies. The graph holds three EXL3 stages (attention with grouped linears, a cross-rank gather, MoE at top 3 of 128 plus the shared expert, a second gather), then the **target's own vocabulary head on every block row** (a second full head read a cycle), then `steps` Markov iterations of `_bias`, `_score`, `_finish`, a cross-rank gather and `_pick`, then the confidence head. That is 11–12 collectives a pass.
- **No L2 prefetch in the pass** (`draft.zig:10`: left out of the port). In the forward the prefetch is worth 5.3% (`PREFETCH=0`: 4.338 s vs 4.12 s; attention producer 240 → 198 µs a layer, MoE 559 → 537 µs).
- **Pass time by block width** (rowcost `FIX_K` arms, exactly 1.00 stream a pass): 3.51 ms at K=0 (five block rows), 3.75–3.90 ms at K=1–6 (six rows once `KEPT=2` widens). November 15 read 3.43 ms at a smaller step mix; like-for-like at 5 rows / 5 steps / greedy / 1 stream it is **3.43 → 3.51 ms, +0.05–0.08 ms**, inside the ±0.05 ms same-work repeat band (`runs/drafter-degradation-20261124/REPORT.md` §0). Do not read 3.43 → 4.21 as a regression.
- **Markov bias cache: 256 rows, hard-coded** (`tri_markov.zig:23`, the kit's `TF_DS_MARKOV_CACHE` default; list in `markov_tokens.zig`). Miss rate 51.7% (k267) and 53.3% (kall ctl1: 23,386 hits, 26,648 misses, 7,173 passes, so 7.0 lookups and 3.7 misses a pass). A miss step recomputes the row against the Markov head, 64,640 × 256 fp16 = 33.1 MB a rank. The cached set changes a step's time, never a draft (`markov_tokens.zig:4`).
- **Offline hit rates on the frozen corpus** (`runs/overnight-20261122/corpus.json`, natural text, not model output): the served list hits 49.2%, matching the served counter. Held-out frequency lists hit 61% (512), 72% (1,024), 81% (2,048), 88% (4,096). The served list plus a per-reply LRU of 256 hits 66–73%. A cached row is 126 KB a rank.

### Candidates (all exact by construction; 1–3 staged, none on hardware)

1. **Break down the pass — BUILT as 0280, never executed.** A phase timer for the pass under `TF_DS_ROUND_PROFILE=2 GRAPHS=0`, plus a steps histogram and miss steps a pass. The 2026-11-23 trace review already gives a stage breakdown from saved ring traces (`trace/out/pass_breakdown.txt`); 0280 is the in-engine cross-check and runs first in `runs/k280/passprof.sh`.
2. **Enlarge the Markov bias cache — BUILT as 0281 (`TF_DS_MARKOV_CACHE`), armed in `runs/k280/run.sh`.** A miss step costs 133–153 µs and a pass has 3.2 of them (measured from the ring traces, see "Trace review results"), so the saving is 0.20–0.33 ms a pass at the offline hit rates, 0.16–0.26 ms at 0281's own held-out rates. 0267 tried to make the miss cheaper; this makes misses rarer and needs no kernel. Gate: `pass_ms` falls ≥0.15 ms against both controls with drafts unchanged.
3. **L2 prefetch inside the pass — BUILT as 0282 (`TF_DS_DRAFT_PREFETCH`), armed in `runs/k280/run.sh`.** Ceiling ≤0.18 ms a pass by analogy with the forward's 5.3%; the `dpf2` hand-off arm must be scored on forward + pass.
4. **Sweep the forward's prefetch constants** (3 µs delay, 230 GB/s pace, 16 MiB budget: the kit's values, hard-coded at `ops.zig:410–420` and `round.zig:1787`, never swept here; B2 traced the rank imbalance to this prefetch).
5. **Time 0261 on and off on sampled requests** (no build).

### Round 0280–0283 (STAGED 2026-11-23 07:35 UTC; nothing launched, nothing run on a GPU)

Four timing-only commits on `kernel-stack-0280` (worktree `~/Workspace/tfref-kstack5`, base production `2bb052f`), no CUDA source changed, tree clean. **Image `tensorfold-dsv41:bench-fused-next-1421187` is built on both ranks** (production's image plus the two cross-built executables; the base's 20 fatbins embedded byte for byte). The review session confirmed the image on the head and production still up on `2bb052f`, and re-ran `zig build test` and `zig build` at the tip: both exit 0. The worker's copy rests on the agent's checksum record (`runs/k280/build/proof.json`). **The new executables have never been executed.** [Briefs and reports](runs/kernel-round-20261123b/) · [Suite](runs/k280/README.md)

| Patch | Commit | Switch | What it is |
|---|---|---|---|
| 0280 | `cb8a4ca` | `TF_DS_PASS_PROFILE=1` (+ counters under any `TF_DS_ROUND_PROFILE`) | the pass's phase times (`pass_phases_ms`, eager), and `pass_counts`: steps histogram (buckets 1–8), miss steps a pass, block rows, zero-initialized `markov_cache` |
| 0281 | `b0715eb` | `TF_DS_MARKOV_CACHE=<rows>` (unset 256; 256–4,096) | more cached Markov bias rows; the first 256 are today's list in today's order (pinned by a host test) |
| 0282 | `92dc42f` | `TF_DS_DRAFT_PREFETCH=1\|2` | the paced L2 prefetch inside the pass: three sites a stage; `2` adds the hand-off of the forward's first reads |
| 0283 | `1421187` | `TF_DS_L2_DELAY_NS`, `TF_DS_L2_RATE_GBPS`, `TF_DS_L2_BUDGET_MIB` (unset 3000 / 230 / 16) | the prefetch's pacing and budget, rank-local; the drafter's prefetch takes the same values |

- **0281's list is a general-text ranking, and its hit rates are lower than the review's estimate.** The served kit ships no ranking (cubins only), so the 3,840 ids after the served 256 come from ExLlamaV3's public calibration files (half code, half English), not fitted to any benchmark prompt. Held out on `corpus.json`: 54.4% at 512, 61.3% at 1,024, 67.7% at 2,048, 76.5% at 4,096 (the review's corpus-fitted lists read 61 / 72 / 81 / 88%). With the measured miss cost that is an estimated 0.16–0.18 ms a pass at 2,048 rows and 0.23–0.26 ms at 4,096.
- **0281 and memory.** 4,096 rows take 473 MiB more a rank. The head shows about 6 GiB available with production up, and memguard kills below 2.5 GiB.
- **0281's gate.** Output tokens cannot move whatever the cache holds, so the exact panel does not test it. `runs/k280/gate_drafts.py` compares per-request rounds, accepted/drafted and sha against control (it passes the saved k272 and k267 arms and fails the D1 drafter arms, as it should).
- **0282 details.** Site 1 sits behind `wq_b`, where the forward forks (`round.zig:1216`). The served six-row pass reads the cuBLAS fp32 gate copy, not `gate_w` (the kit has drafter gates at 5, 10 and 15 rows only), so site 2 prefetches whichever the pass will read. Value `2` makes the pass wait at its end for the hand-off launch (up to about 76 µs), so score it on forward + pass; the traces give no evidence for the hand-off.
- **Compare arms within the new image only.** Production's `tensorfold-native` came from a from-source build (GCC 13.3 `.comment`); the new image's executables are cross-built, so host code generation may differ. Every staged suite has its controls on the new image.
- **Staged, in the order to run them** (each restarts production and restores and validates it at the end; start the next only after `restored: validate rc=0`):
  1. Copy to the head as single files: `runs/k280/` (README, arms, scripts, `work.py`, `gate_drafts.py`, `topk-*`) and `runs/overnight-20261122/{arms.sh,table_gpu.py}`. The exact `rsync` lines are in `runs/kernel-round-20261123b/REPORT.md`.
  2. `runs/k280/passprof.sh` (about 20 min): the pass breakdown under `TF_DS_PASS_PROFILE=1`, and the executables' first run. It should reproduce the trace review's breakdown.
  3. `runs/k280/run.sh` (about 1 h 45 min): `ctl1, mc2048a, mc4096a, dpf1a, dpf2a, ctl2, dpf2b, dpf1b, mc4096b, mc2048b`, sampled probe and sampled workload cells.
  4. `runs/k280/l2-run.sh` (about 50 min): rate 180 / 270, delay 0 / 10000, budget 8 / 32 against two controls.
  5. `runs/k280/topk-run.sh` (about 50 min, production's own image): shipped 0261 against the fallback on sampled rounds.
  6. Gate and score: `python3 runs/k280/gate_drafts.py runs/k280/main --control ctl1,ctl2 --test ...`, then `python3 runs/overnight-20261122/table_gpu.py runs/k280/main --control ctl1,ctl2 --test mc2048a,mc2048b` (and the other pairs). `pass_ms` is the first number to read.
- **Unverified until those run:** that the executables load; any speed effect; that drafts are unchanged under 0281 and 0282 on hardware; that the kit's `_score` reads cache slots above 255 correctly; that a pass graph captures and replays with a second stream (0282); the scripts end to end (only `bash -n` and a parse were done).


### Three more agents launched 2026-11-23 06:36–06:37 UTC (all DONE)

- **Kernel pattern sweep: DONE 07:28 UTC**, see "Kernel pattern sweep results" below. [Report](runs/kernel-sweep-20261123/REPORT.md) · [Brief](runs/kernel-sweep-20261123/BRIEF.md)
- **Trace review: DONE 07:17 UTC**, see "Trace review results" below. [Results](runs/trace-review-20261123/RESULTS.md) · [Brief](runs/trace-review-20261123/BRIEF.md)
- **OpenObserve review: DONE 07:27 UTC**, see "OpenObserve review results" below. [Results](runs/observe-review-20261123/RESULTS.md) · [Coverage](runs/observe-review-20261123/COVERAGE.md) · [Brief](runs/observe-review-20261123/BRIEF.md)

### Trace review results (2026-11-23 07:17 UTC)

Read-only analysis of the saved ring traces (12 rank-pairs: `k272/tr16{a,b}`, `b2/mech`, `b2/rowsplit`). 22 scripts and their outputs are beside [the results](runs/trace-review-20261123/RESULTS.md). Absolute times are from the greedy, short-context mech arms on image `10c9866` (their steady cycle is 39.86 ms), so shares of the 42.8 ms cycle are approximate. The Markov figures were re-derived independently by the review session and agree.

- **The drafter pass has a breakdown now (measured).** A one-stream, six-step pass is about 3.9 ms of GPU time: vocabulary head 1.08 ms, three MoE stages 1.29 ms, three attention stages 0.68 ms, Markov steps 0.49 ms, twelve ring transfers 0.23–0.26 ms. Another 0.39 ms sits between the forward's last transfer and the pass (absorb, pick, uploads) and 0.10 ms between the pass and the forward.
- **A Markov bias miss costs 133–153 µs (measured).** The per-step interval is cleanly bimodal: 9.8 µs on a hit, 142–162 µs on a miss, which is one 33 MB head read at bus speed. The slow mode falls on the same ring sequence in all four arms and both ranks (906 of 906 steps), so it is the token. 3.2 misses a six-step pass, 0.44–0.49 ms, about 1.1% of a cycle. At four streams one head read serves every missing row, but a step is slow 96–100% of the time. **This prices 0281 without the idle-window microbenchmark: 0.20–0.33 ms a pass at the offline hit rates (inferred), so it deserves an arm.**
- **The slow cells are stalls at the Engram layer-1 cut (position measured, cause inferred).** In `tr16a` the 32K × 1 and 128K × 1 cells run 0.94 and 0.62 ms a round slower than in `tr16b` with no code change. All of the first and about 80% of the second sits at the Engram layer-1 gather, the first transfer of the forward's second graph stretch: rank 1 arrives late (producer 674 against 202 µs), rank 0 waits in the ring. 31 of 34 sampled forward stalls above 0.5 ms in a whole arm are at this position. The round's Engram hash, submit and reads must fit inside stretch 0, which is layer 0 alone, about 1.1 ms (`round.zig:2068, 2075, 2355`). The stalled host is rank 1's, so rank 0's `tail` shows it as GPU time. Only one of the slow cells listed above has a trace; that the others are the same is inferred. **Next step, no build:** the two cells five times each with the trace on, in three arms: control, `TF_DS_ENGRAM_SUBMIT_EARLY=1` (0161, exists, off in production), and NVMe power-state latency tolerance 0.
- **Rank skew at the gathers is random and costs about four times the B2 figure (measured).** B2's signed median was 2–5 µs; the mean absolute producer difference is 17–23 µs a hidden gather, uncorrelated between two runs of the same tokens (r = 0.02–0.07). The 71 plain gathers of a forward take 0.58 ms above the standalone floor (0.35 ms with `PREFETCH=0`, 0.50 ms with `SIDE=0`). How much is recoverable is unknown. Score the 0283 sweep with `ring_cost.py` on a 1-in-4 trace as well as round time.
- **Smaller items, 0.2–0.9% each:** the decode KV fetch sends every window row's 512 picks separately (0.39 ms a forward; a distinct-pick counter by depth is the next step); a sampled round's tail is 79–122 µs longer than a greedy round's although the greedy one ships 1.3–1.8 MB over the ring (the NCCL top-k exchange; run `runs/k280/topk-run.sh` first); keyed draft noise costs 24 µs a Markov step; 0.14–0.21 ms of the hand-offs is unattributed.
- **The forward does not start cold after the pass**, as far as these traces show: layers 1–3 equal later layers and do not move with the preceding Markov step's hit or miss. 0282's ceiling is at most 0.18 ms, by analogy only. k267's +0.25 ms cannot be tested (no trace with 0267 on). Prefetch is worth 35–42 µs at plain attention layers, 43–67 µs at KV layers, 11–24 µs at MoE, and hurts nowhere.
- **No periodicity** at 32 or 64 forwards. The 70–100 ms host stalls appear in no trace. One more verified row costs 3.15 ms (2.96 MoE + 0.19 attention).
- **Crash dumps in the saved logs:** one real segfault with a stack (`combo4/bisect-to0184` and `to0185b`, an undefined pointer in `moe`, fixed by that suite's `0184z`); none in the November 22–23 suites.
- **Not answerable from these traces:** which kernel inside an interval is slow, what rank 1's host does during a stall, graph capture against replay, pick overlap at depth, the NCCL exchange's own cost, four-stream detail. Every trace is `10c9866`; production is `2bb052f`.


### OpenObserve review results (2026-11-23 07:27 UTC)

From the lab's OpenObserve only (no ssh, no request to the engine). The live dashboards were re-fetched by the review session and match the saved definitions.

- **What is stored.** 200 `tensorfold:*` series from the head only (`gx10-eb0e`), 28 host `system_*`, 12 `dcgm_fi_dev_*` and 2,643 ConnectX series from both Sparks (the worker is `gx10-235f`), two log streams. **The scrape interval is 30 s**, not the 60 s in `decisions.md`. The engine's `done` lines do not reach OpenObserve; per-request records were rebuilt from scrape intervals in which exactly one request finished. 811 engine restarts since November 10.
- **Live traffic is thin in the current epoch.** After removing suites, deploy checks and harness lifetimes: 15,856 unattributed requests on older images, 1,198 from November 18 22:21, 2,738 on the fused image from November 21 07:38, **none** across the four epochs from November 22 06:01 to November 23 05:16 (suites ran throughout), and 399 on `2bb052f` in its first 1.9 hours. "Unattributed" is presumed client traffic; nothing in the data names a client.
- **Verdicts, live against benchmarked:**

| Benchmarked | Live | Verdict |
|---|---|---|
| One-stream round, short prompt: 42.6 ms at 5.5 drafted rows | 38.8 ms, +3.9% against the benchmark's line at the same drafted rows (n=1) | matches |
| The same at depth (the probe has no deep rows) | +4 to +10% now (n=4); on the fused image +7.5% at 8–64K, +9.5% at 64–200K, +12.3% past 200K (n=280) | **deviates +8–12%**, likely context depth |
| One-stream tok/s: 105–135 code-like, 59–60 prose | 74.6–110.1 now (n=5); median 78 on the fused image (n=280) | **about 40% under the code-like rows**, mostly tokens a round: 3.1 live against 4.2–6.1 |
| Four streams: 220 tok/s (sampled Tetris burst), 116–119 (`dec4` cell) | about 123 tok/s, 84 ms a round | matches `dec4`; 44% under the Tetris burst |
| Eight streams: 136–138.5 tok/s, about 87 ms a round (`dec8` cell) | 91.7 ms a round (+5%) | matches on the round |
| Cold prefill 2,105–2,118 tok/s at 32K | 2,104 on 30,206 tokens; 2,051 on 15,434 | matches at 32K |
| Resumed 8K turn about 1.1 s to first token | 1.28 s (n=90), tails prefill at about 1,620 tok/s | matches; resumed turns prefill 22% under the cold headline |
| Pair health | `pair_failed`, `service_refused`, preemptions all 0 | matches |

- **So the round time and prefill hold up; the headline tok/s does not describe the live mix.** The probe's code-like prompts accept more drafts a round than real requests do. A probe row built from the live mix (3.1 tokens a round, deeper contexts) is the missing benchmark.
- **Seen live, covered by no benchmark:**
  - **Queueing behind eight streams.** On November 23 05:31–06:22 UTC, 178 requests with one identical 3,718-token prompt arrived together: up to 8 running and 35 waiting, mean time to first token 422 s, and 113 of 153 ended by client disconnect (21 of the 22 that can be timed ended at 600 s). No file in the repository records it; it may be an unlogged load test.
  - **Prefill and decode interfere.** While cold passes of 166K–192K tokens ran, other streams' rounds stretched from about 92 ms to 155–312 ms and the passes ran at about 1,680 tok/s instead of 2,040–2,100.
- **Correction:** "no suite has gone past 4 streams" (open item 3 above) is wrong; the November 21–22 `run_suite.py` suites have a `dec8` cell.
- **Dashboards, updated in place** (every original panel unchanged, new tabs appended): Tensorfold 89 → 161 panels, 6 → 14 tabs, all 200 engine series (149 before), with a "Benchmark vs live" tab; DGX Sparks 21 → 119 panels, 1 → 6 tabs, every host and DCGM series and the 321 device-level ConnectX counters (the 2,322 per-queue ones through a name box). Queries were checked through the API; how the panels render in a browser was not seen. Undo: `cd runs/observe-review-20261123 && . scripts/o2env.sh && python3 -I scripts/dash_api.py restore dashboards-before/7522360476127199234.json` (Tensorfold) and `.../7522324382660689920.json` (DGX Sparks).
- **Missing from the store:** `tensorfold:requests_total{key,status}` is scraped but not stored, and it is the only series that would tell clients apart; rank 1 has no `/metrics` endpoint, and the worker's collector has failed to scrape a `tensorfold` target every 30 s since November 11; the `done` line and the round profile (`forward_ms`, `pass_ms`, `rows_a_round`) are log-only. `COVERAGE.md` gives the cheapest route for each.
- **Do not commit `dashboards-before/` or `dashboards-after/` as they are:** the exports carry the owner's login email. No credential is in any file (checked by scanning for it). The agent reported that the password appeared once in its own tool output, so it is in this machine's session transcript.

### Kernel pattern sweep results (2026-11-23 07:28 UTC)

Branch `kernel-sweep-0290` in `~/Workspace/tfref-ksweep`, base `2bb052f`, three commits, tree clean. The review session re-ran `zig build test` and `zig build` at the tip `c5c257f`: both exit 0. **Nothing was timed and nothing went through nvcc or Apple's compiler.** Each patch adds a new kernel symbol looked up only when its switch is on, so the off path runs on today's fatbin; **a switch set against today's fatbin fails at load, so these need an nvcc build for `sm_121` before any arm.**

| Patch | Commit | Switch | What it does | Harness (old source against new) |
|---|---|---|---|---|
| 0290 | `6571e56` | `TF_DS_DRAW_CUT=1` | 0261's `tf_ds_topk_select_kernel` and the fallback `tf_ds_draw_rows_kernel` stop their radix select at a complete bin (0254's pattern) | 1,310 cases, 0 failures; 2.4–2.5 of 8 digit passes on Gaussian-like rows, 6.4 on heavily tied rows |
| 0291 | `c58b41e` | `TF_DS_QD_PAIR_SHFL=1` | the pair query's half-warp maximum by four width-16 shuffles instead of a masked reduction | 498 cases, all 65,536 bf16 encodings, 0 failures. **Removes no work**; adopt only on a standalone win |
| 0292 | `c5c257f` | `TF_DS_BLOCK_MAX8=1` | the pool's block maxima in two aligned 16-byte loads (0213's body in the grid kernel) | 890 cases, 0 failures |

- **Staged, not landed:** `runs/kernel-sweep-20261123/unverified/0001-glm-index-select-cut.patch` (`GLM_INDEX_CUT=1`, the Metal `glm_index_select` stops at a complete bin). No local check exists for Metal; its evidence is a C++ emulation of the old and patched text, 573 cases, 0 failures.
- **Survey:** 68 sites across every kernel file. dsv41 CUDA 25 (4 landed, 3 do not transfer, 5 transfer and are not applied), other CUDA 11, Metal 32 across GLM, Nemotron, Flash Next, Qwen 3.5, Kimi, `ops/` and `core/` (1 staged, the rest listed). [Survey](runs/kernel-sweep-20261123/SURVEY.md) · [Patterns](runs/kernel-sweep-20261123/PATTERNS.md)
- **The catalogue dropped three of the brief's entries on the evidence:** "write in the consumer's layout" has no attributable gain (0278's is unexplained, 0260 and 0269 are neutral); the launch-grid bound was re-scored neutral (0264); the warp integer maximum holds only for whole-warp groups.
- **Rejected with reasons:** `tf_ds_topk_i64_kernel` (writes the k-th key last and feeds a kit cubin); the tile exit in the 0279 pick (the newest block is pinned to +inf, so the k-th pick is always in the last occupied tile, which closes the audit's 0279 idea); the unrotated-warp guard on `tf_ds_fused_rope_qd_kernel` (already measured neutral); `torch_ops/topk.cu` (its output order depends on the exact k-th key).
- **Largest thing found and not applied:** the gate-weight widen at `prompt.zig:1710` moves 11.8 MB a layer a bulk prompt chunk for a constant; caching it costs about 315 MB a rank. Also owed: the 8-wide bf16 widen in the same prefill stage, `tf_ds_block_argmax_v`, Nemotron's `sample.cu` `top_keys`.
- **Each landed patch still owes:** an nvcc build, register and stack counts, GPU parity of the new symbol, the exact panel; then 0290 selector-only timing and a histogram of the digit it stops at on real logits (it rides the `runs/k280/topk-run.sh` pair), 0291 and 0292 standalone timings. Stacking with 0280–0283 is untested; a textual conflict in `ops.zig` is possible.


## Drafter degradation attribution (2026-11-24)

Question put to it: live decode is ~40% under the code-like benchmark rows and the pass reads 3.43 (Nov 15) → 3.9 (traces) → 4.16–4.21 ms (served). **Verdict: neither is a regression. Claim 1 is workload mix; claim 2 is a policy widening plus suite composition. No hardware window was spent and none is needed.** [Report](runs/drafter-degradation-20261124/REPORT.md) — read-only, every figure re-read from a run record.

- **The four pass numbers are two spans.** 3.43, 3.51–3.90 and 4.11–4.26 are all `pass_ms` (host timer around `m.pass`: uploads + graph replay + synchronize + two D2H). The trace's ~3.9 ms is GPU tiles that *include* the 0.39 ms absorb/pick/uploads hand-off and *exclude* the ~0.10 ms tail. Comparable only after normalizing block rows, Markov steps, streams a pass and sampled keying.
- **Equal-work delta: +0.05–0.08 ms a pass** (5 rows / 5 steps / greedy / 1 stream: Nov 15 3.43 → `fk0` 3.51), i.e. +0.12–0.19% of the 42.8 ms cycle, −0.15 to −0.25 tok/s at 128. Below the ±0.05 ms same-work repeat band and far under the 0.28–0.64% pair-gate floor. `CONF_PACK` (production) is an unpriced offsetting saving against it.
- **The apparent +0.78 ms decomposes:** +0.24–0.39 ms deliberate policy widening (0192 `DRAFT_BLOCK=6` + `KEPT=2`: 5 → 6 block rows and Markov steps, plus the kit→cuBLAS gate at 6 rows, `draft.zig:769–775`; the `fk1`–`fk6` gradient is the widened share rising with the pin) and +0.26–0.36 ms suite composition (sampled keyed noise +0.15 ms a pass, streams a pass 1.36–1.43). The widening is net-positive where it engages (+2.5–3.1% tok/s on Tetris/code, `runs/draft-block-20261118/`).
- **Acceptance is content, not a defect.** At one build and config the probe spans **1.8 (prose, 0.51–0.54 acceptance) to 6.1 (go, 0.875) tokens a round** (`k265/ctl1.probe.json`); live's 3.1 at 64% acceptance sits inside that band. Split of the 3.1 → 4.2 gap: **80–100% prompt-dependent, 0–20% policy-tunable (`DRAFT_BLOCK=7`, Tetris only), 0% regression.** Depth explains 8–12% of the tok/s shortfall separately (round time +7.5/+9.5/+12.3% by band).
- **Disproven as causes:** D1 drafter weights (same weights both sides; rounds higher on 16/16 rows in both arms), triage/confidence gating (unarmed; `CONF_PACK` byte-identical; `CONF_LEADER_ONLY` output-invariant), the Markov cache (changes step time, never a draft, `markov_tokens.zig:4`), no-L2-prefetch-in-the-pass as a *regression* (`draft.zig:10` is identical at `8ff4797` and `2bb052f`), 0267's +0.25 ms carryover (not in the `2bb052f` lineage, not armed), and any pass-code change since Nov 15 (diff = block rows, the 6-row gate, `confPack`; `tri_markov.zig` byte-identical).
- **What it cannot answer:** the content split of the acceptance gap (no per-request series stored; `requests_total{key,status}` is scraped but not kept), the equal-work delta below ~0.3 ms, which input the widening gate reads (`r.follow.len` vs verified rows), the sampled/multi-stream split of the suite mean (lifetime `pass_ms`, no per-cell field), and whether a Markov hit from a 441–473 MiB cache stays as cheap as from today's 31.6 MiB.
- **Carried forward:** the one worth-buying window is the staged `runs/k280/run.sh` (0281 + 0282, ~1h45m, no new build). Confirm H1 if `pass_ms` falls ≥0.15 ms with drafts unchanged and the miss rate ~52% → ~32% (2,048) / ~24% (4,096); confirm H2 at ≥0.08 ms; score `dpf2` on forward + pass. An optional `TF_DS_DRAFT_BLOCK=5` arm would give Nov 15's equal-work baseline directly. [Priced experiment](runs/drafter-degradation-20261124/REPORT.md) §3 · [todo](todo.md)

## Earlier status bullets (November 22–23)

- **Superseded 21:25 UTC by 10c9866 (see below). Earlier production: `tensorfold-dsv41:bench-fused-next-27baa93` from 10:44 UTC**, with `TF_DS_KV_FETCH_REUSE=1 TF_DS_CONF_PACK=1 TF_DS_TOPK_MERGE=1 TF_DS_PLACE_BOUND_ALL=1` plus the earlier switches. **It is byte-for-byte what `zig/patches/` builds** (`recipe-0264`; the export chain is verified). Validated 8/8.
- **Undo, one step at a time:** `cp ~/dsv41-env-backups/.env.bak-20261122-pre-0264 .env && ./start.sh restart` goes back to recipe-0261 (`086e8cb`). Earlier steps: `-pre-recipe` (`69335b8`, the fused-next lineage + 0259), `-pre-kvreuse` (`84806bb`), `-pre-fnext` (before today).
- **Today's adopted gains, all exact 39/39 (plus sampled-identical for 0261):** 0259 rounds −1.7%; 0264 rounds −0.41% (−2.15% at 128K × 4); 0260/0261 neutral (launch and collective savings below resolution); the recipe-lineage move neutral. [0264](runs/overnight-20261122/placebound/RESULTS.md) · [lineage](runs/overnight-20261122/lineage/RESULTS.md) · [0259](runs/kvreuse-ab-20261122/RESULTS.md)
- **Single-switch screen on production (10:45–11:47 UTC, all exact 39/39):** `TF_DS_ENGRAM_SCALES=map` **slows decode rounds 5–8.5%** from 32K up (reject; it was in the Nov 17 "best" bundle). `TF_DS_GREEDY_ARGMAX=1` is exact but **1.2% slower** (reject). PDL, `DEVICE_ACCEPT`, `COMPRESS_SIDE`, `ENGRAM_STRICT` and `PREFILL_GATHER_OVERLAP` are within ±0.5% noise in decode and prefill. Control spread is ±0.33% mean, ±1% a cell. `DRAFT_BLOCK_KEPT=2` was retested and adopted (next bullet). [Sweep](runs/overnight-20261122/sweep1/RESULTS.md)
- **`TF_DS_DRAFT_BLOCK_KEPT=2` adopted (ABBA 11:50–12:24 UTC, exact 39/39 in all six arms):** decode tok/s +0.32% mean on the narrow probe (json +2.5%, yaml +1.0%, tetris sampled +1.25%; tetris greedy −0.9%, prose −0.8%); k3 +0.25%. Round counts repeat exactly within each pair, so the effect is real: widening sooner gives fewer, slightly longer rounds. Deploy `runs/deploy-kept2-20261122` (env-only; undo `.env.bak-20261122-pre-kept2`). [Kept](runs/overnight-20261122/kept/RESULTS.md)
- **Decode ring probe, corrected interpretation.** Across 904 rounds, rank 0 averaged about 4 µs staging, 22 µs from doorbell publication to observing the receive flag (76% of the measured interval), and 3 µs copying out. The aggregate mixes ring operations. It leaves the shares of late peer arrival, CPU proxy service, RDMA transfer and GPU flag observation unresolved. [Raw profile](runs/overnight-20261122/ringprobe/probe1.profile.txt) · [Scheduling investigation](runs/overnight-20261122/ringprobe/SCHEDULING.md)
  - **Start with the gather after MoE.** The earlier 32K × 4 profile measured 1.85/1.57 ms per round in this phase on ranks 0/1. In an exclusive two-board window, measure the graph-replay ring at the served hidden width of 5,120 (140 KiB per rank at seven rows). Match per-rank traces by round, layer, collective and sequence, using same-clock intervals. If next-layer weight prefetch delays publication, test moving its fork just after publication with the same bytes and join. Check the next gather for displaced wait and compare probe-off, graphs-on round latency and sampled output tok/s at one and four streams. This remains a proposed hardware experiment. [Both-rank phases](runs/overnight-20261122/phases/RESULTS.md) · [Experiment design](runs/overnight-20261122/ringprobe/SCHEDULING.md)
- **M1, price rows correctly (DONE Nov 22 14:22–15:04 UTC: measured `TF_DS_ROUND_MS` −0.24%, not adopted; see the scheduler ABBA bullet below).** Original plan, kept for reference (`docs/benchmark-gaps-and-moe-review.md` §4.4): The draft allocator prices a round with the static `draft.round_ms` table (30.0 ms at 1 row … 52.1 at 6; `draft.zig:840`). That table predates today's faster build, and the Nov review found its shape wrong (intercept too high, slope past five rows too low). `rowcost/` measures forward ms by rows under `TF_DS_FIX_K=0..6` at 1–4 streams. A fitted `TF_DS_ROUND_MS` then gets an ABBA. It is lossless by construction: drafts only choose how many rows a round verifies. (`TF_DS_ALLOCATE_LEARNED=1` is the in-process variant; it lost 2.5% in combo2, so do not retry it.) A wait loop must not `pgrep -f` a pattern that appears in its own command line; one deadlocked that way for 75 min. Wait on a log line instead (`chain1.sh`).
- **Kernel round 0265–0268 (pi agents, `mimo-v2.6-pro` via `xiaomi-token-plan-sgp`, started 13:49 UTC), from todo.md "New kernels to test against KEPT=2":** one agent a patch, each in its own worktree `~/Workspace/tfref-k<NNNN>` (branch `kernel-<NNNN>` off `recipe-0264` `27baa93`). 0265 `TF_DS_DRAFT_HEAD_SCORED` (the drafter head projects only scored rows), 0266 `TF_DS_COMPRESS2_PAIR_ONCE` (0197 computes each distinct ratio-2 pair once a window), 0267 `TF_DS_MARKOV_FUSED` (greedy Markov bias miss + tile scoring, plus a miss counter under the profile), 0268 `TF_DS_DRAFT_QKV_NORM` (the drafter's joint q/KV norm). The 0198 E8M0-from-bits, exact-reciprocal and warp-integer-max items are already in recipe-0264 (`tf_ds_scale_byte_bits`, `tf_ds_recip_bits`; the masked `__reduce_max_sync` was measured 0.10 µs slower and is kept only for the whole-warp case). Briefs and reports: `runs/kernel-round-20261122b/`. Each one is gated on hardware separately, with `KEPT=2` in both arms.
  - **k265 result (17:00 UTC, `runs/overnight-20261122/k265/RESULTS.md`):** stacked image `5ec7aba` (0265+0266+0268) with every switch off is bit-identical to production. Each switch alone: exact ×2 and sampled-identical ×2. The sampled probe is flat (it's quantized and can't resolve <0.5%). The workload's round time shows 0265 −1.75% and 0268 −1.89% at 4K×1 (both arms below both controls), plus 0268 −1.55% at 32K×4. 0266 has no measurable effect. Only 0265 logs a banner. 0266/0268 engagement is inferred from code (0266's pair kernels are mandatory `try` lookups at load; 0268's gate `wq_a.n<=2048` holds at 1280); the rank-0 profile can't show either.
  - **Lesson:** a switch with no banner and a null result can't be told apart from a switch that never fired. Future briefs should require a one-time `std.log.info` banner on first engagement (0265's pattern, `draft.zig:709`).
  - **Joint deployment gate.** The six-arm `kall` suite on `0f89e5a` started at 17:39 UTC. At the 18:08 UTC check on November 22, the second three-switch arm (`all3b`) had started at 18:07 and both all-four arms were pending. The tabulator waits for the suite to finish and syncs the remote artifacts. Compare both three-switch repeats with both controls for exactness, sampled identity and speed before adopting 0265 + 0268 or including 0266. The combined arm includes 0266, so it cannot isolate that patch’s contribution. [Arms and run artifacts](runs/overnight-20261122/kall/) · [Individual results](runs/overnight-20261122/k265/RESULTS.md)
  - **Standalone diagnostics deferred.** The 0266/0267 microbenchmark queue and its watcher were cancelled before either produced GPU timings. The prepared harnesses compare pair-once with a repeated compressor baseline and sweep Markov cache misses against the original bias-plus-score sequence. Resume in a separately scheduled idle Spark window. [Follow-up](todo.md#new-kernels-to-test-against-kept2-2026-11-22) · [Harnesses](runs/negative-kernels-20261122/)
- **kall (18:38 UTC, `runs/overnight-20261122/kall/RESULTS.md`): 0265+0266+0268 together are exact ×2 but give no gain, so they are held and production stays on `27baa93`.** Probe −0.08%, rounds +0.13% mean. 4K×1 is +0.16%, so k265's single-switch −1.75% and −1.89% didn't reproduce (likely noise). 4-stream rounds are +0.7–1.1%. With 0267 added (all4), rounds are +1.25%. The patches stay off by default in `kernel-stack-0265`. Lesson: a ≈1.8% single-cell win from two arms each is not enough evidence; require a replicate before staging a deploy.
- **k267 (17:39 UTC, `runs/overnight-20261122/k267/RESULTS.md`): 0267 `TF_DS_MARKOV_FUSED` is rejected.** It's exact and sampled-identical and the banner fires, but every greedy probe row is 6–10% slower (prose −10.2%) and the fused kernel is slower than the two launches it replaces. Sampled traffic is unaffected (the arm doesn't engage). The bias-cache miss rate is 51.7% (rank 0, about 39K lookups an arm). The `markov_cache` counters start from Zig's `undefined` fill (0xAAAA…); subtract `0xAAAAAAAAAAAAAAAA` until they're zero-initialized.
- **k269 (20:13 UTC, `runs/overnight-20261122/k269/RESULTS.md`): 0269, 0270 and 0271 are each exact ×2, none is measurably faster, so all are held and production is unchanged.**
  - 0269 and 0270 engage (banners); probe deltas are −0.13% / −0.16%.
  - 0271 never engaged (no exact-resume prepend in this workload) and needs a multi-turn replay.
  - **Noise floor (key finding):** the r271 arms run code identical to control, so they're an A/A test. They show **±2–3% per round-time cell** (+3.02% at 32K×1). The "both test arms below both controls" rule passes by chance in ~1 of 6 cells; that likely explains k265's unreproduced −1.8%.
  - **Better round-time signal:** use the rank-0 lifetime `forward_ms` (±0.25% across 8 arms) or a replicate suite.
  - **New probe metrics:** chunk-gap p95 has a ~±0.5% arm-to-arm spread. Streamed x1 runs 129.6–129.9 tok/s at p95 45.5 ms; streamed x4 runs ≈215 tok/s at p95 71 ms.
- **Round B2 done (`runs/sched-20261122/b2/RESULTS.md`): root cause of rank 1's wait.**
  - **Measurement artifact:** most of k272's "rank 0 late" came from `arms.sh`'s `TF_DS_ROUND_PROFILE=1`, which only rank 0 reads; rank 1 needs `RANK1_PROFILE`. **Trace and per-rank work must set `TF_DS_ROUND_PROFILE=0`.** Arm-vs-arm comparisons in past suites stay valid.
  - **Real residual (profile off):** rank 1 late by ≈2–5 µs a gather at 1 stream, ≈0.15 ms a round (~0.4%); balanced at 4 streams.
  - **Cause:** the paced L2 prefetch, which saves rank 0 ~4 µs a gather more than rank 1. With `PREFETCH=0` the ranks are equal to 0.1%, but prefetch is a large net win (4.12 vs 4.34 s), so it stays on.
  - **Head box:** 1.2–1.4% slower on small-row bf16 GEMV microbench, not clocks/config/fragmentation. It does **not** show in the engine's kernels.
  - **nsys:** CUDA injection doesn't take effect in this engine (only OS-runtime calls were captured).
  - **Mech arms (1 stream, 8×384 tokens, trace N=4, 39,294 records a rank each):** fx1L/fx1Lb (production) max 4.117/4.124 s, gap p50 attn −1.5/−2.6 µs, moe −5.2 µs; `PREFETCH=0` 4.338 s (+5.3% slower), gap +1.0/+1.6 µs; `SIDE=0` 4.167 s (+1.1% slower), gap +0.6/−0.4 µs. Producer µs r0/r1: attn 198/201 with prefetch vs 240/240 without; moe 537/542 vs 559/559.
  - **Row/concurrency split (profile off, N=16):** gap = producer r0 − r1 (positive = rank 0 late). k0 (no drafts, 1 row): rank 0 late ~9–11 µs p50 at x1, ~2 µs at x4. Fixed k6 and production (6–7 rows): rank 1 late ~3–10 µs p50 at x1, ≈0 at x4. So the sign flips with row count; at production row counts it is rank 1.
  - **Possible next step:** a per-rank prefetch budget knob (≤0.4% at 1 stream). Not pursued: below the ±0.25% forward_ms resolution once halved, and prefetch has no rank-specific code.
- **Cache replay (00:30 UTC, `runs/cache-20261122/RESULTS.md`): 0276 and 0271 are both inert in production; both held with no hardware arm.**
  - **0276:** no state went stale (0 `misses_stale`, pool 2.1M positions, ~480K tokens cached). Every kept state also gets a spill-tier point, which 0276 skips by design.
  - **0271:** it needs `TF_DS_EXACT_RESUME`, which is lossy (decisions §0121).
  - **Multi-turn finding:** resumes land on the 2,048-row grid, so each later turn recomputes up to 2,047 rows. That's ≈1.1 s TTFT per ~8K-token turn at the head of the queue. Prompt passes run one at a time (≈1.9K tok/s), so at C=4 turns queue to 4–6.7 s TTFT.
  - `replay.py` (`repeat` / `turns` modes, `/metrics` deltas) is the reusable workload.
- **nccl suite (01:05–02:18 UTC, `runs/overnight-20261122/nccl/`), all 6 arms exact 39/39 + sampled IDENTICAL:**
  - **`NCCL_MAX_CTAS=4` ADOPTED (02:22 UTC, `runs/deploy-mc4-20261123/`).** Cold prefill +0.83% at 32K (2,101/2,102 vs 2,084/2,084 tok/s) and +0.85% at 96K (2,045/2,050 vs 2,031/2,029). Both arms beat both controls, whose spread is ≤0.1%. Decode is neutral (probe −0.00%, chunk-gap p95 −0.01%). Env proof: `nccl/mc4a.nccl-env.txt` and the deploy's `armed.txt`. Deploy checks: exact IDENTICAL vs `deploy-10c9866/exact.json`, sampled IDENTICAL, validate rc=0.
  - **`TF_DS_SOLO_OVERHEAD=1` held.** It engages (fewer rounds, 7,452 vs 7,499–7,554, 6.41 rows a round vs 6.31–6.35), but the probe is −0.01% and chunk-gap p95 −0.18%.
  - `TCP_NODELAY` is dropped from the candidate list: `link.zig` already sets it, plus `SO_BUSY_POLL`.
  - `TF_DS_NO_COPIES` was measured before (`to-benchmark.md`: mildly negative).
- **Kernel round 0277–0279 (pi agents, started 02:49 UTC, base production `10c9866`, `runs/kernel-round-20261123/`).**
  - 0277 `TF_DS_DRAFT_RING_BATCH`: the drafter's three per-stage KV norm/RoPE/ring launches in one, dropping the unread `dr.ao`.
  - 0278 `TF_DS_TAPS_PACKED`: prompt taps written straight into the drafter's row-major layout (no completion copies).
  - 0279 `TF_DS_IDX_PICK_WIDE`: the indexer's small-row selection past 16,384 blocks, up to 65,536 (decode beyond a 262K-token bucket, which still falls back to 3 launches).
  - **todo.md is stale here:** top-k exchange in sampled verification is 0261 `TOPK_MERGE` (already in production), and 0195's small-row form and `max_nbk` 16,384 are done.
- **NCCL screen (03:26–04:17 UTC, `runs/nccl-screen-20261123/RESULTS.md`): no winner; `MAX_CTAS=4` stays.** `PROTO=LL` −19% prefill, `MAX_CTAS=2` −2.3%, `IB_QPS_PER_CONNECTION=2` −1.2%, `BUFFSIZE=8M` −0.4%. `MAX_CTAS=8`, `MIN_CTAS=4`, `PROTO=Simple`, `NCHANNELS_PER_NET_PEER` 1/4 and `P2P_NET_CHUNKSIZE=512K` are all within ±0.2% (controls ±0.1%).
- **Kernel round 0277–0279 results:**
  - 0277: not built. `kvNormRope` is the kit cubin `_kv_norm_rope`, and ~0.02% of a round wasn't worth it.
  - 0278 `TAPS_PACKED`: committed `971c69d`.
  - 0279 `IDX_PICK_WIDE`: committed `f6e126b`. It engages at contexts over 262K: 300K → nbk 32,768, 600K → 65,536.
  - Stack `kernel-stack-0278` = `2bb052f` (`tfref-kstack4`). Its image is built; `zig build test` passed on rerun (one spill-tier timing flake).
- **k278 ADOPTED (05:20 UTC, `runs/k278/RESULTS.md`): production is now 2bb052f with 0278 `TAPS_PACKED` and 0279 `IDX_PICK_WIDE` on.**
  - All arms exact 39/39 and sampled identical; the 300K/600K decode shas equal control.
  - 0278: cold prefill +0.4% at 32K, +0.6% at 96K, with every cell above both controls.
  - 0279: neutral at depth.
  - **Spill restore works:** a restarted server serves a 300K prompt in 1.5 s TTFT and 600K in 2.7 s, from disk (cold: 152 s / 332 s).
- **Drafter D1 REJECTED (03:26 UTC, `runs/d1-20261123/RESULTS.md`).** Exact ×2, but it accepts fewer tokens a round on every probe row: decode −2.42% mean, streamed x1 −5.15%. It was fit against a different base quant (LibertAIDAI), not our Mia EAQ + ablit. Both weight files are deleted; the `DATA_SUFFIX` hook stays.
- **PRODUCTION CHANGED 21:25 UTC: image `tensorfold-dsv41:bench-fused-next-10c9866`** (`kernel-stack-0272` = `27baa93` + 0272 + 0273 + 0274 + 0275). This adopts the 0273 failed-capture join fix and the 0274 tool-call first-token fix. 0272 (`RING_TRACE`) and 0275 (`COPY_MIN`) are in the image but unset.
  - Deploy checks: exact 39/39 and sampled IDENTICAL vs `sched2/ctl1`, `tool_choice: required` + thinking returns a tool call, validate rc=0.
  - Backup: `~/dsv41-env-backups/.env.bak-20261122-pre-10c9866`.
  - **New exact reference** for later suites: `runs/deploy-10c9866-20261122/exact.json` (identical to `sched2/ctl1`).
- **k272 / Round B (`runs/overnight-20261122/k272/RESULTS.md`).**
  - 0275 `COPY_MIN=3` is neutral (probe +0.14%), so it's held.
  - 0272 trace overhead at N=16: forward +0.2%, x4 chunk-gap p95 −2.4%.
  - **Trace finding:** the proxy is not the cost (0.7 µs doorbell→post, 32 ns completion→flag).
  - **Rank 0 is the late producer at 1–7 rows** (p50 waits: r1 18–21 µs vs r0 7–8 µs). **Corrected cost (B2):** imbalance costs about half the producer gap per gather (each rank proceeds when its peer's data lands), so ≈0.25 ms a round (~0.7%) at 1–7 rows. At 8+ rows it reverses (rank 1 late, ≈0.23 ms). The earlier "≈1 ms / 2.5%" was wrong. KV exchanges run the other way.
  - Ruled out: clocks (both 2,380 MHz, no throttle), config (rank 1's flags match), other GPU processes.
  - **B2 (done, see the B2 bullet above):** nsys could not see CUDA work; the per-rank trace with the profile off on both ranks settled it instead.
- **k272 suite (staged 20:30 UTC):** image `10c9866` is `kernel-stack-0272` (`~/Workspace/tfref-kstack3`) = `27baa93` + 0272 + 0273 + 0274 + 0275. The picks were clean and `zig build test` passes; the syntax error set equals the base's.
  - **Arms:** ctl1, cm3a, tr16a, ctl2, tr16b, cm3b. `cm3` is `TF_DS_COPY_MIN=3`; `tr16` is `TF_DS_RING_TRACE=16`.
  - **Bug fixes in the controls:** 0273 and 0274 are unconditional, so the ctl arms carry them; exact vs production proves they keep bits.
  - **Trace files:** `arms.sh` now copies each rank's `/tmp/ring_trace.<r>.jsonl` out of the container when an arm sets `TF_DS_RING_TRACE`. The chain runs `h0272/summarize.py` on both arms (Round B's data).
  - 0276 (cache replace-stale): done, but inert in production (see the cache replay bullet).
- **Outside-repo reviews (19:00 UTC).** Nine forks reviewed: zachxu Coder-1x, Senti100, HelloTianxu, joseyarashio, Blackwellboy/BLACKFOLD, bertholomus, Yushell 2.7 bpw, ursuciprian, Elevatormusic. Clones are in the session scratchpad; the source patches were copied to `runs/kernel-round-20261122d/src/`.
  - **No faster decode:** none beats our sampled 1-stream or 4-stream decode.
  - **Production checks:** the Engram token map's sha matches the kit's (`de40bcdd…`).
  - **Bug confirmed in our tree:** a failed stretch capture leaves `side_out` / `pf_out` set (`round.zig:1905`), so every later capture fails until a restart.
  - **Kernel round 0273–0276 (pi agents, started 19:03 UTC):**
    - 0273 the capture-join fix (joseyarashio 0062);
    - 0274 the call-gate first token (joseyarashio 0061);
    - 0275 `TF_DS_COPY_MIN` (Yushell 0006);
    - 0276 `TF_DS_CACHE_REPLACE_STALE` (upstream PR #134 keep side).
  - **Later candidates:**
    - Drafter D1 overlay (ursuciprian): **done, rejected −2.4%** (`runs/d1-20261123/`);
    - `top_k` -1/0/>56 with concurrent streams: **open**;
    - `TCP_NODELAY` (already set in `link.zig`) and `NCCL_MAX_CTAS=4`: **adopted**, +0.8% prefill;
    - 4,096-row prompt passes (bertholomus v0.7.1: +3–10% prefill; a port): **open, operator decision** (chunk shape moves bits, decisions §0121);
    - `TF_DS_NO_COPIES` (measured earlier, mildly negative) and `SOLO_OVERHEAD`: **held**, neutral.
  - **Calibration:** the kit_bench word-list prompts read ~11% high against natural text, so outside prefill numbers (2,543–2,583 tok/s) aren't comparable with ours (1,984–2,066 natural).
- **Scheduling round A results (18:42 UTC, `runs/sched-20261122/RESULTS-A.md`):**
  - **Ring cost (standalone bench, width 5120, production settings):** 9.4 µs + 1.16 µs a row (1 row 9.9 µs, 7 rows 17.3 µs, 28 rows 41.0 µs). Repeats agree within ±0.4 µs, and byte checks are clean.
  - **Back-to-back price:** 7.22 rows × 80 gathers ≈ 1.4 ms a round.
  - **Served probe vs bench:** the served rank-0 probe shows ≈29 µs a gather, so about 11 µs a gather (≈0.9 ms a round) is an upper bound on excess above protocol.
  - **Proxy:** pinned to CPU 15 on both ranks, with 0 involuntary switches, 0 migrations and ≤0.3 ms run-queue wait across x1/x4 bursts. Under 4 streams it spins at 100%.
  - **Decision:** the CPU-reservation candidate is dropped. Excess wait is producer skew or per-sequence proxy/NIC service, which the 0272 trace must split.
- **Scheduling round (`runs/sched-20261122/README.md`, from `ringprobe/SCHEDULING.md`), started 18:15 UTC.** A, no engine code: (A1) standalone `tf-dsv41-rdma-bench` at width 5120 with production HCAs, GID 3 and PDL, and only the proxy pinned to CPU 15; (A2) read-only proxy scheduling counters while serving; (A3) `suite_probe.py` chunk-gap p50/p95 + `--stream-bursts`, and `probetable.py --metric` with a rounds-match column. B: 0272 `TF_DS_RING_TRACE` trace patch (pi agent). C: scheduling candidates, only after the trace.
  - Facts found while planning:
    - The bench needs `rdma_gather.cu` as a fatbin file (the engine embeds it). Compile it in the image with `nvcc -fatbin`, torch flags, `-O3` and `sm_121`.
    - The bench's default GID is 5; production uses 3.
    - The bench never pins its proxy, so pin the `tf-rdma-proxy` TID alone.
    - Decode hidden gathers use the plain `Ring.gather` (compare `us_gather`).
    - Rank 1's ring probes are never printed.
    - The live proxy shows `Cpus_allowed_list: 15` with only 6 involuntary switches lifetime (one sample).
- **Kernel round 0269–0271 (pi agents, started 17:08 UTC; `runs/kernel-round-20261122c/`, worktrees `~/Workspace/tfref-k02{69,70,71}` off `27baa93`):** the remaining open dsv41 CUDA items of todo.md. 0269 `TF_DS_DRAFT_TAPS_DIRECT` (the drafter reads contiguous verify-round taps in place; decode, every round), 0270 `TF_DS_PROMPT_PICKS_ONCE` (prompt attention pads its picks once per selection; prefill), 0271 `TF_DS_RESUME_ASSEMBLE` (one kernel assembles exact-resume rows and positions). Each must log an engagement banner. `run-agent.sh <n>` runs the agent, then resumes it (`-c`) until a commit or a premise-failed report. Image-row Engram pass-through was skipped (text-only serving); router-weight widening was skipped (routing is ≤0.9% of the prefill window).
  - **Status at 18:55 UTC:** all three committed with reports (0269 `5d1b4cd`, 0270 `6926a5e`, 0271 `8078e03`).
    - 0269 engages on every verify round at the served config: all windows draft, so the tap rows are contiguous. Its absorb is eager, outside any captured graph.
    - 0270 cuts 60 launches per prefill chunk per rank (76 → 16). Prefill is not graph-captured.
    - 0271 turns 4 copy launches (6 when the prefix wraps) plus a pageable upload into one kernel.
  - **Stack:** `kernel-stack-0269` (`~/Workspace/tfref-kstack2`) is `5c69584` = `27baa93` + 0269 + 0270 + 0271. The conflicts were additive (imports/refAll/switch reads, kept both). `zig build test` is 834/834 on rerun (the first run hit the timing-sensitive link-watch test under the agents' load).
  - **CUDA syntax stubs:** the checker reports 9 errors on the production base `27baa93` itself, the same set as the stack (stub false positives, e.g. `'hi'` in `dsv41_ops.cu`), so "syntax ok" can't be required verbatim on this tree. Compare the error set against the base instead.
  - **k269 suite:** ctl1, t269a, p270a, r271a, ctl2, r271b, p270b, t269b on `5c69584`, with `--narrow --sampled --stream-bursts` (the first suite with chunk-gap p50/p95). Launched by the chain after the image build.
- **Scheduler ABBA on `KEPT=2` (14:22–15:04 UTC, all arms exact 39/39 and sampled-identical):** a measured `TF_DS_ROUND_MS` (from `rowcost`) is **−0.24% mean and −2.65% at four sampled streams**. Rejected as fitted: past 7 rows the table was extrapolated, and pricing wide rounds dearer makes the allocator draft less. `TF_DS_DRAFT_BLOCK=7` is **+0.61% mean**, with greedy code +7.2%, ts +2.6% and go +1.6%, but sampled Tetris is −2.1% solo and at x2. **Closed (sched3, 15:05–15:33 UTC): on sampled traffic it is −0.10% mean and −1.9%/−1.0% at x2/x4, so it is rejected and production stays at block 6.** Greedy-only probes overstate draft-width wins. Gate every scheduler change on `suite_probe.py --narrow --sampled` (all classes at T=1.0, two seeds) as well. Under `TF_DS_FIX_K`, concurrent streams do not share rounds, so the pin measures solo rows only. [sched2](runs/overnight-20261122/sched2/RESULTS.md) · [sched3](runs/overnight-20261122/sched3/RESULTS.md) · [rowcost](runs/overnight-20261122/rowcost/RESULTS.md)
- **Engine source of record:** branch `recipe-0264` (was `recipe-0261`) in `~/Workspace/tfref-fnext-rebase` (worktree of `tfref-fused-next`). Build bench images from it or its descendants with `runs/fused-next-20261121/bench/build.sh <rev>`. Patches go on top and get exported with `scripts/export-fnext-series.sh --branch <rev>`. **The current tip is `recipe-0264`.**
- **Candidate 1 is done: bit-identical, decode rounds 1.7% shorter.**
  - Sizing found the sharded-KV fetch and exchange at 3.6–4.2% of a round, projecting 3.0–3.5%. [Sizing](runs/kvreuse-size-20261122/RESULTS.md)
  - The ABBA gate on one image: 39/39 identical in every arm, reuse engaged on both ranks, rounds −0.5 to −2.5% (mean −1.7%). [A/B](runs/kvreuse-ab-20261122/RESULTS.md)
  - About half the projection was realized. The likely reason is cross-rank wait that moves to the next collective. The next lever there is rank balance, not the exchange.
  - Source: branch `kv-fetch-reuse` `69335b8` in `~/Workspace/tfref-kvreuse`. **Not yet exported to `zig/patches/`.**
- **Earlier today:** `84806bb` was deployed 05:58–06:01 UTC (image swap; [record](runs/deploy-fnext-20261122/)).
- **Candidate 3 closed (07:24–07:31 UTC): the pooled prompt indexer covers only each prompt's ≤128 bounded-replay rows.** A diagnostic image logged the pool at the candidate source on cold 24K–164K prompts. It fired on chunk pairs summing to exactly 128 rows (104 + 24, 91 + 37, 93 + 35). Bulk prefill rows take the keyed path, which has no pool. The saving would be about 15–20 ms of an 80.7 s 160K prefill (0.02%). The earlier "~10% of long prefill" estimate wrongly assumed every row was pooled. [Pool union](runs/pool-union-20261122/RESULTS.md) · [kernel pricing and the exact compact route](runs/prefill-pool-score-20261122/RESULTS.md)
  - Kept as findings: the kit's `_index_score_rows` is linear in width (29 µs per 1,024 columns per 128 rows). Compacted keys through it are bit-exact. `_index_score_cand` is 60% 1–12 ULP off it. Adjacent rows' pools overlap little (a 128-row union is 57–100% of the width).
  - Cold prefill on production: 1,984 / 2,066 / 2,040 / 2,031 tok/s at 24.6K / 49K / 98K / 164K.
- **Overnight 2026-11-22 (operator: "drive overnight", restarts and config changes authorized).** Tools are in `runs/overnight-20261122/`:
  - `arms.sh` runs any `name|image|ENV…` arm list. Each arm gets the exact panel `--like` a reference plus a compare, the round-time workload, and banner/error counts on both ranks. Production is restored and validated at the end.
  - `table.py` prints the per-cell round-time table with deltas against the control arms.
  - `phases.sh` gives both ranks' eager phase timers plus the prefill profiler. Rank 1 prints only at exit: `docker stop` rank 0, read rank 1's log, then `./stop.sh` (start.sh's `RANK1_PROFILE` note).
- **0260–0263 had not landed when the operator reported them (08:00 UTC).** Only the briefs existed (`runs/kernel-round-20261122/BRIEF-*.md`). `~/Workspace/tfref-k260` (branch `kernel-0260-0263`, base `8d838cc` = the rebased recipe tree) was clean, with no commits or reports, and no authoring session was running.
  - A background writer agent is implementing them in brief order: 0260 `TF_DS_CONF_PACK`, 0261 `TF_DS_TOPK_MERGE`, 0263 `TF_DS_KV_RING_BATCH`, then 0262 `TF_DS_POOL_SCORE` last, because pool-union showed its scope is the ≤128 replay rows (about 0.02%).
  - **The recipe tree is now hardware-gated (08:02–08:26 UTC):** `recipe-0259` `a5a5822` (image `bench-fused-next-a5a5822`, full source build) is **exact 39/39** against production in both arms. It is neutral overall (+0.12%) but +3.2% at 128K × 4 in both arms (unattributed; the rebased tree carries the recipe's 0182–0211 forms). 0260+ A/Bs use the recipe image as control. [Recipe gate](runs/overnight-20261122/recipe-gate/RESULTS.md)
- **Item 1 closed: the ranks are balanced** (08:33–08:42 UTC, both ranks' eager phase timers). At 32K × 4 the two ranks agree within 1–2% on every phase. What remains in decode's collectives is latency: about 80 hidden all-gathers a round at about 40 µs (≈ 5.5%), and the reused KV exchange at about 69 µs × 8. The next non-kit decode lever is collective latency (a ring for the hidden all-gathers like the KV path's), not balance. [Phases](runs/overnight-20261122/phases/RESULTS.md)
- **Item 2 priced: the keyed prefill indexer is 2.7% of a chunk at 32K and 7.5% at 128K**, growing linearly with depth. A prefill chunk is about 47% experts (gate/up 281–302 ms, down 144–157 ms of ≈ 890–975), 11% hidden all-gathers, 10–14% dense projections, 5% engram.
- **0260–0263 writer results (08:43 UTC).** Branch `kernel-0260-0263` in `~/Workspace/tfref-k260`; reports in `runs/kernel-round-20261122/report-*.md`.
  - **0260 `TF_DS_CONF_PACK` `e06735c`:** committed. 11 → 1 launches; identical by construction (exact widen only); CPU harness 152 cases × 2 passes.
  - **0261 `TF_DS_TOPK_MERGE` `33091fb`:** committed. Takes effect only with `TF_DS_T27_DEVICE_DRAW=1` on sampled rows; cross-rank check in `pairSwitches` bit 3; CPU harness 110 rounds / 325 picks, 0 failures.
  - **0263:** stopped. The kit's FP8 step was unknown. The spec is now in `runs/kernel-round-20261122/kit-kv-norm-rope.md` (fetched from the public kit source), but it is parked at about 0.02% of a round.
  - **0262:** stopped by its brief's stop condition. `_index_score_rows` shares one key buffer across rows. Moot at about 0.02%.
  - Both committed patches were cherry-picked onto `recipe-0259` → branch `k260-on-0259` `864d32f` → image `bench-fused-next-864d32f`. Its six-arm gate (base / conf / merge / both ×2 / base, exact `--like` reference + a sampled-identity check with varying top_k + round times) is in `runs/overnight-20261122/k260-gate`.
- The operator has authorized restarts and config changes for this testing (2026-11-22).
- **Production runs the `fused-next` binary, while `zig/patches/` holds the rebased equivalent, which has never run on hardware.** Building and gating an image from the recipe (a full source build on the head) is still open. It does not block the image swap. Do not `just sync-head` or restart with changed `zig/patches` until that build has been run.

**State at 04:40 UTC, after the 0254–0258 pair gate:**

- **Production is back on exactly the configuration it ran before the window**: both ranks on `tensorfold-dsv41:bench-prune-six-20261120` with the same `.env` switches (`TF_DS_IDX_FUSED=1`, `TF_DS_FUSED_COMPRESS2_NORM=1`, `TF_DS_FUSED_ROPE_STORE=1`, `TF_DS_FUSED_ROUTE=1`, `TF_DS_DRAFT_BLOCK=6`, `TF_DS_ENGRAM_LEAD=1`, `TENSORFOLD_TOKEN_CACHE_MB=32`, `IMAGE=`/`PREPARE=0`). `/health` 200, chat answers, spill tier cold-namespaced by the suite blocks only. Nothing from `fused-next` is deployed.
- **The 0254–0258 gate passed 5/5 blocks and is speed-neutral.** Exact 39/39 on every block, probe 11/11 hashes with identical rounds/accepted, `fixed_replay` inputs and token hash equal, `dec8_staggered` 3/3 waves equal; next-vs-base `-1.35%`/`+1.71%` pure decode against a `-2.23%` base-base control. So `84806bb` is adoptable on correctness and buys nothing in serving (its standalone wins stay −39…−60% `0254` and −15…−23% `0255`). [Results](runs/suite-fused-next2-20261121/RESULTS.md) · [Suite](runs/suite-fused-next2-20261121/README.md)
- Two suite-scripting bugs were fixed and committed: `preflight.sh`'s `cd` (one level too high) and its rsync target; `dec8_staggered`'s script was missing on the head, so s01's cell was re-recorded and merged. The guarded `0215` capture is still uncaptured (a stopped production reads as "no server", not idle).

- **Adopted and rebased (2026-11-22, operator's pick).** `84806bb` (`0254–0258`) is the series to build from: 19 patches (`0213–0217`, `0230–0231`, `0240–0242`, `0250–0258`) now sit in `zig/patches/`, regenerated from `~/Workspace/tfref-fnext-rebase` (`c3b3859` + the cherry-picked round) by `just export-fnext`; `0254`/`0255` replace the hand-ported pair. Two conflicts were resolved by hand (0217's `Ops` literal + dynamic-shared opt-in, 0256's dispatch + `0182`'s `TF_DS_TOPK_FAST` precedence). Verified: chain reproduces the rebased tip, `zig build test` passes, and a `TF_REF + patches` checkout type-checks (`zig build install native -Dfatbins=<dummy dir>`). **Not deployed**: no image has been built from the recipe yet, and no deploy gate run.

### Lessons from the 0254–0258 gate

- Identify the baseline and price the work at the shape serving actually uses. Both images already enabled fusion. At the traced 1,024-score width, 0255 saved 0.206 µs at seven rows. The dec8 matched-step timing swung further between two base runs than between builds. Use the per-wave breakdown and same-image controls before attributing a throughput dip to a kernel. [Timing analysis](runs/suite-fused-next2-20261121/RESULTS.md#where-the-time-goes)
- Choose fusion by shape and buffer reuse. On the candidate image at 128 query rows, the split rotation/FP4 chain saved a paired median 2.56–3.30 µs on a single invocation across two runs. Fusion won with eight distinct buffers in one graph. Measure the workload's actual reuse before changing dispatch. For the deep indexer, distributing wide reads across more GPU blocks can repay a second launch. Keep the 6,144-block threshold until a valid crossover sweep. For prefill, profile the full-row scoring ahead of the pool mask. [128-row query experiment](runs/suite-fused-next2-20261121/query128/RESULTS.md) · [Shape comparisons and follow-ups](runs/suite-fused-next2-20261121/RESULTS.md#where-the-time-goes)
- A long prompt can finish in one token. Verify multiple active decode rounds in each intended bucket, then weight kernel savings by measured launch counts and GPU time on each rank. The deduplicated shape trace establishes dispatch coverage only. [Replay coverage](runs/suite-fused-next2-20261121/RESULTS.md#trace-s05-128-lines-per-rank-both-ranks-equal-in-shape-coverage)

### Kernel, instruction and hardware findings register (fused rounds 0193–0258, as of 2026-11-22)

This register collects the findings for future patch decisions. GB10 is `sm_121`. Every timing is standalone through CUDA graphs unless it says otherwise. Builds use `nvcc -O3 -arch=sm_121 --fmad=false --ftz=false`, never fast math.

**Where decode time goes: the ceiling for repo-owned kernels.** The data is the solo decode stage windows on the pre-fused image (Nov 18, stage-instrumented, so shares are approximate). [Stage windows](runs/overnight-opt/followthrough02/stage-windows.csv)

| Stage | Head share | Worker share | Owner | Note |
| --- | ---: | ---: | --- | --- |
| `moe` | 45.2% | 40.0% | kit expert cubins | 202/228 GB/s, 80%/90% of the contiguous-read reference. Do not touch (rebuilt kit = new binary). |
| `a_q` | 6.0% | 4.7% | mixed | 88/129 GB/s |
| `attn` | 4.4% | 3.9% | kit attention core | 158/182 GB/s |
| `gather_a` + `gather_m` | 6.6% | 13.4% | NCCL all-gathers | Worker figure is mostly waiting on the head |
| `sh_rows` + `sh_fetch` | 3.7% | 9.8% | **repo (sharded KV fetch)** | Largest repo-owned decode stage. See candidate 1 below. |
| `a_in`, `a_woa`, `a_core`, `m_rt` | 3.2–3.7% each | 1.9–2.4% each | mixed | |
| `head`, `engram` | 2.5%, 2.4% | 2.6%, 2.5% | kit / repo | `head` at 185 GB/s, 73% of reference |
| `a_idx` (all indexer work, where 0195/0213–0217/0254/0255 live) | 2.3% | 1.9% | **repo** | Bound on what any indexer kernel can still win in decode |
| `mix_attn` + `mix_ffn` | 2.9% | 2.2% | repo | |

By estimate, the fused rounds so far worked inside roughly 8% of the decode round (a_idx, parts of a_in/a_q, the compressor and the route). This is the likely reason (an estimate, not a measurement) that the fused pair gates since November 20 have landed within about ±1% of control. The exception is `suite-fused-20261121` at probe +0.76%/+0.91%; the November 19 route was a real 18–19% regression. A 2 µs kernel saving at seven rows across 40 layers projects to about 0.08 ms of a 42–87 ms round. Graph-on decode reads 146–162 GB/s per rank. [Bandwidth](runs/overnight-opt/RESULTS.md)

**Per-kernel timings at served shapes (standalone GB10)**

| Kernel / path | Before → after | Shape | Source |
| --- | --- | --- | --- |
| 0197 decode compressor + RMSNorm | 3.48 → 1.75 µs; 3.99 → 2.00; 6.74 → 2.01 | 1/7/32 × 512 | [audit](runs/fused-kernel-audit-20261120/RESULTS.md#final-0197-decode-compressor) |
| 0195 fused indexer selection | 44.77 → 9.80 µs | 4,096 blocks, random scores | same |
| 0199 six-slot prune (counters on) | 9.782 → 2.547 µs; counters off 1.819 vs 2.133 prototype | 7 rows | [0199](runs/prune-six-20261120/RESULTS.md) |
| 0199 stats on minus off | 0.728 µs/layer (0.069% of a 42 ms round) | 7 rows | keep counters on |
| 0254 score top-k | 237 → 101 µs random; tied 139 µs | 128 × 32,768, k=512 | [kernel review](runs/suite-fused-next-20261121/kernel-review/RESULTS.md) |
| 0255 candidate mask | 1.257 → 1.051 µs | 7 rows, width 1,024 (128 eight-score blocks) | [timing](runs/suite-fused-next-20261121/kernel-review/index-timing.json) |
| 0215 small-row deep indexer | 12.14 → 10.53 µs (1 row), 12.44 → 11.80 (7 rows) at 8,192 cands; **loses** 8.46 → 8.67 at 7 rows/4,096 | | [wp1](runs/fused-next-20261121/wp1/RESULTS.md) |
| 0198 query rotation + FP4, 128 rows | fused 5.80 vs split 5.02 µs. Split wins one invocation by 2.56–3.30 µs (paired median, two runs). Fused wins with 8 distinct buffers in one graph. Fused also wins at 48 and 1,024 rows. | | [query128](runs/suite-fused-next2-20261121/query128/RESULTS.md) |
| 0198 query baseline | 1.681 µs | 7 rows × 32 heads × 128 | [gb10-repeat](runs/fused-kernel-audit-20261120/gb10-repeat.log) |
| 0200–0212 follow-up | prompt compressor 28–67% faster, rope-store 14–45% faster; decode compressor 13–17% **slower** (0207 `isfinite`, ~0.03% of a round); 0206 key fold 316.6 vs 38.5 µs (R=128, width 32,768) | | [followup](runs/fused-followup-20261121/RESULTS.md) |
| Original fused route 0193/0194 | 18–19% slower single-stream, 11% slower at 4 streams. One block per row recomputes the whole 384×5,120 gate. | | [Nov 19](runs/fused-bench-20261119/RESULTS.md) |

**Instruction-level (SASS) findings.** Per-kernel counts for the `84806bb` sources are in [opcounts.txt](runs/fnext-sass-20261122/opcounts.txt).

- **Duplicate `expf` in the compressor: closed, no patch needed (checked 2026-11-22).** The source calls `expf` four times on two operands. NVCC already merges them: every `MUFU.EX2` appears in adjacent pairs, followed by the two `MUFU.RCP` of the divides. That holds in `compress2x_norm128/512` (4 = 2 × 2 bodies split at the entry `BRA.U !UP0`), `compress2x` (2), and the generic `compress2_norm_kernel` (68, its inlined `ITEMS` specialisations, still paired). This retires `todo.md`'s "cache the two compressor exponentials" item.
- **RoPE partner shuffles:** 0242 already skips the exchange on wholly unrotated warps in the latent/key chains. `tf_ds_fused_rope_qd_kernel` (one element a thread, used below decode's largest window) still shuffles unconditionally. At HD=128/RD=64 that is 2 of 4 warps per head, one `SHFL` per element. Worth folding into any future edit of that kernel, but not a patch on its own.
- Barrier counts: 0202 cut `fused_prompt_idx_key` from 6 to 4 and `fused_idx_key` from 5 to 3. 0209 cut LDS from 26 to 4. `tf_ds_idx_fused_kernel` carries 15 `BAR.SYNC`, 40 `SHFL` and 1 `ATOMS` (the radix histogram).
- Shared atomics: `sm_121` already coalesces intra-warp uniform shared `atomicAdd`. On ties, 0254's extra cost is about 82–86% from 2 extra row-scan passes (+32.1 MiB, 6 barriers) and 14–18% from cross-warp single-bank contention. Warp aggregation with `__match_any_sync` would address only the second part. Cutoff-straddling −∞ rows are under 0.03% of served wall time. [0254 ties](runs/pre-bench-20261121/topk-and-patches/RESULTS.md)
- Generated FP4 conversion uses seven dependent comparisons per element. It is a plausible contributor to the 128-row fused-query gap, with an unmeasured share. [wp3](runs/fused-next-20261121/wp3/RESULTS.md)
- Bit-exactness rules that held: the 0197 decode RMSNorm had to spell out the kit's approximate PTX instructions (SASS `MUFU.SQRT` then `MUFU.RCP` with range handling) instead of `sqrtf` + `__fdiv_rn`. Nine bf16 bytes differed until that change. The prompt norm keeps torch's `rsqrt` (`MUFU.RSQ`). Triton contracts `re = fma(xe, cs, -(xo*sn))`, and the odd element differs between torch and the kit (0252 fixed the pair query's odd element). 0206's key XORs the top bit and the kit's `_score_keys` does not. The kit `_score_keys` faults when `width % 4 != 0`. All kernels keep `LOCAL=0 STACK=0`.

**Hardware and measurement findings**

- Pair-gate resolution: the base-to-base control spread is about 0.6% (dec8 matched-step up to 2.2%). The dec8 dip traced to random HTTP admission order under `TF_DS_ROUND_ROWS=16` plus the `min(t_first)` prefill exclusion, not a kernel. Run conditions: clocks 2,392/2,385 MHz, 84–87 °C max, lowest head free memory 4.9 GiB. [dec8 audit](runs/pre-bench-20261121/dec8-audit/RESULTS.md)
- Harness: legacy stream-0 `cuMemcpy*_v2` is asynchronous for pageable buffers of 64 KiB or less and races `CU_STREAM_NON_BLOCKING` kernels (B2, 2 of 220 false mismatches). `bench.py` now uses stream-ordered copies and emits `timing_valid` only while `live.connections == 0`. [B2](runs/pre-bench-20261121/b2-investigation/RESULTS.md)
- **Match programmatic dependent launch (PDL) per kernel.** Native `TF_DS_PDL=1` was exact and speed-neutral in s05. The served native path uses PDL off, while the pinned Triton kit selects PDL-enabled variants for the Markov bias and score. Reproduce each path’s launch attribute, argument alignment and constexprs. Extract the fatbin from the benchmark image’s lanes executable, record its hash and the kit cubin hashes, and check the parameter counts before timing. [Served query example](runs/suite-fused-next2-20261121/query128/RESULTS.md) · [Markov harness](runs/negative-kernels-20261122/markov.py)
- **Graphs hide only 1–2.5% of a decode round** on this build (eager vs graphs-on `forward_ms`, 2026-11-22), so eager stage shares carry over to served rounds. [kvreuse-size](runs/kvreuse-size-20261122/RESULTS.md)
- **The served clock-mark profiler is broken.** With `TF_DS_CLOCK_MARKS` and graphs on, `round_clock_us.rounds` reads `0xAAAA…` (undefined fill: the served `Model`'s `clk_*` field defaults never apply) and every phase reads 0 µs. Use `TF_DS_ROUND_PROFILE=2 GRAPHS=0` (eager phase timer, about 0–3% overhead) for stage attribution.
- **Decode rows cost almost only in the MoE.** From one stream to four (about 6 to about 20 rows a round), `moe` goes 15.5–17.8 → 38–40 ms. `head` moves 1.12 → 1.27 ms, `a_q` 2.53 → 2.66, `attn` 1.8 → 2.4, `sh_rows` 0.85 → 2.1. Every row-proportional idea outside the experts is worth tenths of a percent; the expert GEMMs (kit, do-not-touch) and their grouping are where row cost lives.
- Production decode stages (84806bb, eager timer): `moe` 15.5–17.8 ms (1 stream) and 38–40 ms (4 streams) of 31.5–62 ms rounds, `attn` 1.8–2.4 ms, `a_idx` 0.8–2.8 ms (grows with depth), `sh_fetch` 0.34–0.49 ms, `sh_rows` 0.85–2.14 ms.
- Dense prefill projections reach 35% / 67% of the same-shape FP16 reference on head / worker. The prompt gate/up stage is 17.9% of the head's chat-prime window. Standalone EXL3 unpack is 1.0%, prompt routing 0.90%.
- 0217 needs the opt-in dynamic shared memory limit to cover 131,072 pool columns, requested at module load.

### GB10 findings to reuse for the next kernel change

- **Price decode weight reads against the measured access pattern.** Four-stream routed MoE reached 93–96% of the pair’s 231–236 GB/s scattered-read reference. At fixed bytes, reaching that reference would save about 2.4–3.7% of the measured eager decode window. For larger gains, measure expert bytes per emitted token on real shared rounds. For bandwidth attribution, the calibrated counter was `lts__t_sectors_op_read_lookup_miss_realtime.sum` at 32 bytes per sector. It counts read misses beyond L2, some of which the system cache can satisfy. Collect counters separately from timing. [Hardware calibration and measured ceilings](runs/overnight-opt/RESULTS.md) · [Expert byte accounting](runs/w4/RESULTS.md)
- **Check which instructions a fusion replaces.** On a Markov miss, 0267 saves 252.5 KiB of FP16 staging traffic per rank and one launch per greedy step, but replaces the kit’s tensor-core dot with scalar FP32 accumulation on CUDA cores. The served greedy probes lost 6–10%. The contribution from scalar arithmetic, register pressure or spills remains unmeasured. The deferred harness records registers and local bytes per thread and times bias, score and their chain across hit/miss counts. Preserve the FP16 rounding point before adding the logit when trying another implementation. [Kernel analysis](runs/kernel-round-20261122b/report-0267.md) · [Served results](runs/overnight-20261122/k267/RESULTS.md)
- **Account for parallel blocks and active lanes.** At width 512 the original compressor took 1.75/2.00/2.01 µs at 1/7/32 rows. Patch 0266 reduces seven rows to four distinct pair computations while keeping one block per row. The served result was within control drift. For instruction substitutions, check the actual lane mask in SASS. On the FP4 latent path, ptxas emitted six static `REDUX` instructions for the half-warp version of `__reduce_max_sync`; it added about 0.10 µs over the four shuffle stages. The whole-warp path uses a different reduction. [0266 geometry](runs/kernel-round-20261122b/report-0266.md) · [Served crossover](runs/overnight-20261122/k265/RESULTS.md) · [Reduction timings and SASS](runs/fused-next-20261121/wp3/RESULTS.md)
- **Preserve the serving working set in a graph replay.** At 128 query rows, the split kit won by 2.56–3.30 µs for one invocation. Eight distinct buffer pairs flipped the result in favor of fusion by 0.98–1.14 µs per invocation. Eight uses of one buffer favored the kit again. Record buffer reuse and invocation count, place GPU timing events inside the graph, and restore any input the kit mutates before the timed region. [Paired timings and in-place RoPE control](runs/suite-fused-next2-20261121/query128/RESULTS.md)
- **Check dispatch and capture boundaries before changing row layouts.** For 0265 the served EXL3 vocabulary head selects its reduction template from the layer shape, allowing scored rows to be packed. The plain cuBLAS path can round differently when the GEMM row count changes. For 0268 the joint Q/KV norm resolves from the kit’s constexprs and argument specialization; preserve the drafter’s `slot = -1` ring-write guard when comparing kernels. For 0269, drafter absorb is enqueued eagerly even with serving graphs enabled. Contiguous tap windows can therefore be selected per round on the host; a gap between participating windows falls back to packing. The 0269 source audit estimates 1–4 copies avoided per rank, with hardware timing still open in that report. [0265 dispatch audit](runs/kernel-round-20261122b/report-0265.md) · [0268 kit contract](runs/kernel-round-20261122b/report-0268.md) · [0269 capture and layout audit](runs/kernel-round-20261122c/report-0269.md)

### The next set of patches, ranked by serving price (2026-11-22)

No engine work past `0258` exists in any `tfref-*` checkout, so this is the next round. It follows the gate's lesson: price at the serving shape before building, and use `84806bb` as the control. Each item needs a standalone byte-parity check, then the 39-case exact suite, long-reply hashes and a graphs-on pair A/B.

1. **(DONE 2026-11-22: 0259, −1.7% decode rounds, in production.) Reuse fetched KV and renumbered picks between decode layers (sharded fetch mode).**
   - The served checkpoint has eight stretches of layers that share a KV source and pick set. 30 of 38 fetches are repeats, 29.53 MiB of logical payload per round at 7 rows / 512 picks.
   - It targets the largest repo-owned decode stage (`sh_rows` + `sh_fetch`: 3.7% head, 9.8% worker). `sh_rows` runs at 0.1–0.7 GB/s, so it is the exchange's wait time, not bytes. It is marked right after `fetchDecode`'s ring send (`round.zig:1420`). Price it by exchanges saved (30 of 38 per round), not by MiB.
   - Structure confirmed in the checkpoint config (2026-11-22): `kv_source_layer_ids [2, 8, 14, 20]` and `index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]`. So layers 2–39 form eight stretches (2–7, 8–13, 14–19, 20–23, 24–27, 28–31, 32–35, 36–39), each sharing one KV source and one pick set, and `fetchDecode` (`round.zig:1380`) runs once per compressed layer. Next, with no restart: size `sh_fetch`/`sh_rows` with graphs on, using the `84806bb` stage trace on the idle head.
   - Invalidate on a new round, a KV update or a pick update, and keep both ranks' ring schedules identical. (`round.zig:1329–1374`, `config.zig:287–290`)
2. **Sampled verification: exchange rank-local top-k instead of full logits.** Clients run at T=1.0, so this is on every real decode round. Each rank's collective input drops from 258,560 to 448 bytes per row (`top_k` ≤ 56). The host check matched 96 cases. Benchmark selection + exchange + draw with graphs on. (`lanes.zig:1449–1523`, `round.zig:2009–2027`)
3. **(CLOSED 2026-11-22: only the ≤128 replay rows take the pooled path, so it is worth about 0.02% of prefill; see runs/pool-union-20261122.) Prefill: score only retained blocks in later prompt indexers (layers 24/28/32/36).**
   - At 128 × 65,536 the current path materializes 33.5 MB of scores, of which at most a quarter of the columns per row survive. That makes 0254's top-k a prefill cost.
   - At 2,048 candidate blocks a row needs 16,384 or fewer scored columns.
   - Profile full-row scoring before building. (`prompt.zig:1435–1450`)
4. **(CLOSED 2026-11-22, by measurement: the vocab head is weight-read bound. The target head, which uses the same EXL3 weights, costs 1.12–1.16 ms a round at ~6 rows and 1.25–1.28 ms at ~20 rows, about 9 µs a row, so projecting 12 rows instead of 20 saves ~0.07 ms a pass, 0.1–0.2%. Source: runs/kvreuse-size-20261122/p2g0.json `ph_head`.) Drafter: project vocabulary logits only for scored steps.** At 4 streams / 5 block rows / 3 steps that is 12 rows instead of 20, saving 1.97 MiB of FP32 output per rank per pass. First measure how often the policy asks for fewer steps than the block width.
5. **Prompt expert activation in the gate/up GEMM epilogue.** This stage is 17.9% of chat-prime prefill, with 126 MiB of FP32 intermediates per rank per layer. It touches the expert path, so time a standalone epilogue first.
6. **Small decode items, only if bundled with a larger edit:**
   - 0197 at 128 threads × 4 channels (D=512): removes the `row[512]` exchange and a barrier. The kernel is 2.0 µs.
   - `rope_qd`'s unrotated-warp shuffle skip.
   - Repeated ratio-2 pair computed once per window (4 distinct pairs in 7 rows).
   - Tighter 0199 expert grid (49 → 43 places). The capped test moved decode only 0.0–0.5%.

**Owed measurements on the adopted series (idle head, no restart):**
- The guarded `0215` crossover capture (`small_nbk` stays 6,144 until `timing_valid: true` medians move it).
- A `fixed_replay` prompt that decodes about 200 tokens, so the 8,192/16,384-block decode buckets are actually traced with multiple rounds.

**Previous state (2026-11-21, stopped before Spark engine restart):**
- **Live production (untouched, actively serving):** Both ranks run `tensorfold-dsv41:bench-prune-six-20261120` (`~/Workspace/tfref-fusedbench` @ `d66e663`) with `TF_DS_IDX_FUSED=1`, `TF_DS_FUSED_COMPRESS2_NORM=1`, `TF_DS_FUSED_ROPE_STORE=1`, `TF_DS_FUSED_ROUTE=1`, `TF_DS_DRAFT_BLOCK=6`, `TF_DS_ENGRAM_LEAD=1`, `TENSORFOLD_TOKEN_CACHE_MB=32`, and `IMAGE=` / `PREPARE=0`. **Undo:** `cp ~/dsv41-env-backups/.env.bak-20261121-pre-fused .env && ./start.sh restart` on the head (and remove the seven lines from `.env`). [Suite results](runs/suite-fused-20261121/RESULTS.md) · [Deploy record](runs/deploy-fused-20261121/)
- **`fused-next` engine branch (`~/Workspace/tfref-fused-next` @ `84806bb`, `0213–0258`):**
  - `0213–0253` (`d5ecba0`) passed the 5-block paired Spark suite (`39/39` exact in all blocks, all comparable replies identical, speed & `TF_DS_PDL=1` within +-1% resolution; production restored). [Suite results](runs/suite-fused-next-20261121/RESULTS.md)
  - `0254` (`topk_indices_cut`, `-39%` to `-60%` standalone) and `0255` (`cand_fast8`, `-15%` to `-23%` standalone) passed `194/194` standalone parity cases. [Kernel review](runs/suite-fused-next-20261121/kernel-review/RESULTS.md)
  - `0256–0258` (`84806bb`) made `fused_trace.zig` thread-safe, unified call-site arm dispatch with `0254`/`0255` trace coverage (`TF_DS_TOPK_FAST` precedence preserved), extracted `planKvSource2` (`0231` host dispatch, `630/632` Zig tests + 4-chunk GB10 byte parity `0` diffs), and fixed S9/S12. [0231 dispatch](runs/pre-bench-20261121/p0231-dispatch/RESULTS.md)
  - Closed without restart: Review Blocker B2 (`bench.py` stream-0 vs `CU_STREAM_NON_BLOCKING` DMA race fixed; kernels `0`-error under `compute-sanitizer`), `dec8` `-0.59%/-0.69%` dip (random HTTP admission order under `TF_DS_ROUND_ROWS=16` + `min(t_first)` prefill exclusion; active decode neutral/faster), `0254` tied-score cost (`<0.03%` served share, no kernel change), and `bench.py` 4-caller `floor`/`newnf` label + `timing_valid` guards (`9/9` unit tests, `100` historical rows audited). [B2](runs/pre-bench-20261121/b2-investigation/RESULTS.md) · [dec8](runs/pre-bench-20261121/dec8-audit/RESULTS.md) · [Harness audit](runs/pre-bench-20261121/harness-audit/RESULTS.md) · [0254 & patch lineage](runs/pre-bench-20261121/topk-and-patches/RESULTS.md)

**Next actions (in order):**
0. **(2026-11-22) Run `runs/deploy-fnext-20261122/deploy.sh` on the head to put `84806bb` in production** (see the current state above), then start the ranked next set with candidate 1's no-restart stage trace.
1. **Adoption DONE (keep `84806bb`); the recipe-image gate is still owed (the image swap above does not need it).** Build an image from the recipe (`./start.sh restart` prepares when `zig/patches` changed; expect a full source build on the head) and run the series' gates: 39-case exact panel, `tools/validate.py --compare`, the hardware smoke, then a serving A/B against production. Until that passes, production stays on `bench-prune-six-20261120`. The measurement window ran 2026-11-22 02:34–04:31 UTC this way: `bash runs/suite-fused-next2-20261121/preflight.sh` from the workstation (its two path bugs are now fixed). It builds/ships `bench-fused-next-84806bb` (no restart during build), syncs only the suite folder (never `just sync-head`), and on the head: waits for 30 external-idle minutes, snapshots `production.rank{0,1}.env`, captures the guarded `0215` crossover timings into `runs/pre-bench-20261121/crossover-0215/` (keep `idx_fused.small_nbk = 6,144` unless `timing_valid: true` medians move it), then runs ABBA `s01–s04` (`d5ecba0` vs `84806bb`) + `s05` (`TF_DS_FUSED_TRACE=1` both ranks) with the `fixed_replay` (unsalted 49,967/99,904/200,000-token hashes + prefill tok/s + decode ms/round) and `dec8_staggered` cells. Gates: `python3 -I runs/suite-fused-next2-20261121/gate.py` and `table2.py`. [Suite README](runs/suite-fused-next2-20261121/README.md)
2. **DONE: the rebase/export is in `zig/patches/` (`just export-fnext` regenerates it).** Keep `~/Workspace/tfref-fnext-rebase` as the export source; the engine's own `fused-next` branch (84806bb) remains the reference the pair gate measured. [Lineage map](runs/pre-bench-20261121/topk-and-patches/RESULTS.md)
3. **After the pair gate:** the owed full soak/300K round for `TF_DS_ENGRAM_LEAD=1` ([evaluation order](to-benchmark.md#current-evaluation-order)); update `runs/overnight-opt` / `runs/review-round` baseline checks to accept `TENSORFOLD_TOKEN_CACHE_MB=32` before any full campaign.

**0199 six-slot decode prune, November 20.** The opt-in `TF_DS_FUSED_ROUTE=1` path now uses one thread per row for the applied, uncapped 0.85/3/kept rule with six routed picks and one shared slot. It works with triage off. On GB10 at seven mixed synthetic rows, median kernel time with counters enabled fell from 9.782 to 2.547 µs. With counters disabled, 0199 took 1.819 µs versus 2.133 µs for the earlier six-slot prototype. The kernel skips sorting when the shares are already ordered, tests only the three ranks eligible to drop, and updates one distinct-expert bitmap per pick. GPU comparisons matched pick IDs, FP32 weight bits and all six counters at six window sizes from 1 to 64 rows. Both service binaries built. Serving throughput and long-reply parity for 0199 are still open. The earlier two-rank test of disabling counters on the generic kernel was within control variation, so keep counters enabled. [0199 results and build](runs/prune-six-20261120/RESULTS.md) · [Generic-kernel stats A/B](runs/prune-stats-20261120/RESULTS.md)

**Before the 0199 serving A/B.** Use the same 0199-capable image on both ranks and in both arms. Keep triage and PDL off, graphs and prune stats on, and vary only `TF_DS_FUSED_ROUTE=0/1`. This isolates the six-slot kernel from 0194’s dead-row launch fusion. Compare the 39 exact cases and long greedy reply hashes, then bracket the throughput measurements with controls and compare matched round counts and draft acceptance. At seven rows the kernel saving projects to 0.29 ms in a 42 ms round if all 40 layers realize it. The existing probe derives `ms_per_round` from `decode_s` rounded to 0.1 s, a 0.52 ms step at 194 rounds. Use the server’s directly reported `tok_per_s` for the greedy comparison, or record higher-precision decode duration for round timing. [Probe calculation](runs/draft-block-20261118/probe.py) · [Recorded control](runs/prune-stats-20261120/control1.timing.json)

The standalone harness also passed a correctness-only graph replay on GB10. It restored the input rows between two launches of the same graph and retained a shared stats buffer across both. Pick IDs and weight bits matched the generic eager reference, and all six cumulative counters matched the sums of the reference windows, including across a 32-bit carry. This covered five datasets and six row counts with PDL off. [Graph replay check](runs/prune-six-20261120/graph-verify.log)

**Final 0197 decode compressor, November 20.** Set `TF_DS_FUSED_COMPRESS2_NORM=1` on both ranks to fuse 0127’s ring writes, partner lookup and pair compression with the pinned RMSNorm in one launch at widths 128 and 512. This switch takes precedence over `TF_DS_FUSED_COMPRESS2` at those widths. Each thread handles one channel and reuses the current row’s KV and score from registers. The kernel preserves the bf16 rounding and reduction order from the mismatch repair.

| Decode shape | 0127 + RMSNorm | First repaired 0197 | Final 0197 | Speedup over 0127 + RMSNorm |
| --- | ---: | ---: | ---: | ---: |
| 1 × 512 | 3.48 µs | 3.54 µs | 1.75 µs | 1.98× |
| 7 × 512 | 3.99 µs | 3.62 µs | 2.00 µs | 1.99× |
| 32 × 512 | 6.74 µs | 3.61 µs | 2.01 µs | 3.36× |

The final kernel was fastest at every measured shape in both standalone GB10 runs. Both runs matched every checked byte in the normalized output, stride padding, raw rings and invalid flag. The original mismatch cases passed at both widths, and the 0195 and 0198 regression checks passed. Production kept running throughout. The two-rank 1,024-token Tetris hash, serving throughput and hardware performance ceiling remain unverified. [Full comparison and reproduction](runs/fused-kernel-audit-20261120/RESULTS.md#final-0197-decode-compressor)

**0195/0197/0198 revised and tested on a GB10, November 20.** The old 0197 decode RMSNorm disagreed with the pinned kit on nine bf16 output bytes across 4,096 rows of width 512. Changing only its square root and reciprocal to the kit's instructions removed all nine differences. A failing row also matched after replay through the repaired decode compressor. In standalone CUDA graphs at seven rows, the revised compressor took 3.67 µs versus 11.11 µs unfused and 4.13 µs with 0127. The 0195 indexer took 9.80 versus 44.77 µs at 4,096 blocks with random scores. All five revised 0198 paths matched their references and ran faster at their measured shapes. The switches remain opt-in. The revised two-rank Tetris hash and serving throughput have yet to be measured. [Review, regression probe and timings](runs/fused-kernel-audit-20261120/RESULTS.md)

**0193/0194 rework, before 0199.** The rework uses the kit gate, router, and parallel expert prep. With both pruning and a dead-row mask active, `TF_DS_FUSED_ROUTE=1` combines them in one launch. The earlier 0196 drafter fusion was removed from the recipe. The November 19 results below describe the original kernels.

**Measured on the pair, November 20 (18:38–19:09 UTC): the rework is bit-identical and speed-neutral; nothing fused is faster.** One image (`bench-fused-rework-20261120`: production source + reworked 0193/0194 + 0195/0197/0198, no 0196), one switch per arm. With `TF_DS_DRAFT_TRIAGE=conf:0.2`, adding `TF_DS_FUSED_ROUTE=1` gave 39/39 identical exact cases, unchanged long-reply hashes and the same speed (the merged launch runs on every layer while triage is on). It passes because the original fusion was dropped, not repaired, and it does nothing under production's switches (no triage). **0197 alone causes the 1024-token greedy Tetris reply change** (`053b9e`, 195 rounds); 0195 and 0198 change no hashes, but nothing shows whether their fused paths ran. Every arm is within ~0.5% of the two controls (off vs off2 differ ~0.6%, the resolution floor); triage alone is ~0.5% below both. This bracket left prune stats enabled and PDL off. Production restored and validated; nothing deployed. [Results](runs/fused-bench-20261120/RESULTS.md)

DeepSeek-V4.1-Flash on two DGX Sparks, TensorFold EXL3 TP2. Use the [best runtime configuration and challenger history](best-runtime.md), [deployment recipe](README.md), [measurement definitions](docs/measurements.md), [CUDA kernel microbenchmark & opcode process](docs/cuda-kernel-microbenchmark-process.md), [GB10 hardware performance guide](docs/gb10-hardware-performance-guide.md), [benchmark matrix](to-benchmark.md) and [decision record](decisions.md).

**Quality at deep context, November 20: not the configuration.** A reported "intelligence degradation" in a live agent session at ~533K tokens (69.6% of a 768K client window) was investigated. `.env` and the image are unchanged since the November 19 lead-reader deploy. A planted-recall probe (eight vault codes at 2–98% of a natural-text prompt, then their order and sum) gave identical token hashes with `TF_DS_ENGRAM_LEAD` on and off at 4K, 64K, 256K and 530K, and a cache-resumed 530K equalled the cold one. Greedy, production failed only at 530K and the repo baseline (all tuning keys off, ABLIT kept) at 256K and 530K, both by reasoning to the token cap. At the clients' sampling (T=1.0, top_p 0.95, top_k 20, five explicit seeds, 8,192-token cap) production recalled all eight codes and the sum at 530K on 5 of 5 seeds but swapped mid-context entries on every seed; 256K was 4 of 5 (one dropped the last entry); 64K was 3 of 5, the other two reasoning to the cap. A reproduced 64K runaway had the right answer within ~2,000 characters, then "verified the order" by transcribing the corpus until the cap: over-verification, not a short loop, so upstream PR #138's `LOOP_GUARD` probably would not catch it. Production was restored and validated after each restart; nothing was changed. **If this comes up again:** check the prompt depth and whether the bad replies ended on `max_tokens` (the done lines) before suspecting switches; judge quality at T=1.0 with explicit per-request seeds (without one the seed is keyed on the prompt, so repeats are identical), never greedy (upstream MiaAI-Lab#54: this model loops at T=0 on items that pass at T=1.0); suggest a thinking budget or lower `reasoning_effort` for runaways. Rerun with `runs/depth-quality-20261120/sampled.sh` on the head (waits for 120 s idle, no restart). **Not measured:** the repo baseline at sampled settings (the one open comparison; it isolates the lossy T13 prune) and more than five seeds per depth. [Results](runs/depth-quality-20261120/RESULTS.md) · [Upstream review](runs/depth-quality-20261120/RESULTS.md#caveat-from-upstream-greedy-decoding-loops-on-this-model)

**Original fused routing patches (November 18–19, not deployed).** `zig/patches/0193-dsv41-tf-ds-fused-route.patch` and `zig/patches/0194-dsv41-tf-ds-fused-route-prep.patch` (stacked on 0193, both on series tip `432d800`, `TF_DS_FUSED_ROUTE=1`, **off by default**) fold a decode window's gate chunk-sums, top-six routing, the T13 prune's rule, 0177's dead-row override, the experts' rotation (`rot_in_kernel`'s FWHT128 arithmetic) and the expert grouping (`ids`/`ucount`/`members`) into **one** launch per layer; only the two kit `grouped_decode` launches remain. The prompt path keeps cuBLAS + a fused route (kc=0). An offline CPU harness executed the exact extracted kernel against the upstream `decode_prep_kernel` and `_route_parts` as bitwise references (3,367 fuzz checks + the prep comparison, 0 failures); `zig build test` shows no regressions (only the two pre-existing HEAD failures, `prompt_cache.test.0178` and `round.hotPick`, both untouched). Two independent reviews found and fixed: a PDL-wait race (`pdl_enter` is now the kernel's first statement), a launcher scratch-allocation line that would not compile, unchecked pointer/alignment trust boundaries, and a `shared_id == ne` rejection that would have refused every routing call on the served model (fixed; host test added). **Bit-identity with the compiled kit on a GB10 is unproven** — Triton's intra-thread reduction order for `[BN,256]` and libdevice `expf`/`logf`/`sqrtf` parity are the open items. Production, images, and `just sync-head` are untouched; `decisions.md`/`todo.md` were not edited. The harness lives in `/home/colum/Workspace/p0194/fused_run.cpp` (scratch). [Archived 0193/0194](runs/fused-bench-20261119/stacked-series/0001-0193-0194-recipe-patches-fixed-versions.patch)

**Measured on the pair, November 19 (20:00–20:34 UTC): no fused patch is faster; none is adoptable.** This supersedes the "unproven"/"gated" wording above. The fixed 0193/0194 (re-exported after the first image, built 06:39 UTC, failed the gate 36/39) still differ on 31 of 39 exact cases, and the fused route is 18–19% slower single-stream and 11% slower at four streams (likely because the kernel runs one block a row and each block recomputes the whole 384×5120 gate; not profiled). 0195+0197+0198, ported onto the production tree and run as one bundle, pass the 39 cases and are speed-neutral, but change a 1024-token greedy reply near token 984 (which patch is not isolated). 0196 ran only with the route on and added nothing measurable. Production restored and validated on both ranks. Item 1's route half failed, so by its own rule items 3–4 are not justified as listed; item 1's PDL half and item 2 were not run. [Results](runs/fused-bench-20261119/RESULTS.md)

**Original benchmark plan.**
1. **Spark gate before any timing:** build an image with 0193+0194, run the 39-case exact suite with `TF_DS_FUSED_ROUTE=1`; then with `TF_DS_PDL=1` (PDL is unset in production today). If bits diverge, stop and rework — nothing else in this list is justified.
2. **PDL everywhere (0126, one arm with the gap profiler).** The largest unsized lever; every kernel in `dsv41_ops.cu` already takes `pdl`; graphs hide host launches but not device-side gaps. Estimated 1–3%; keep or drop on the measured gap script, per the review's plan.
3. **Graphs-on timing A/B:** `m_rt` / stage-window pass with the fused launch vs unfused — the eager profile overstates what a captured graph hides. Adopt or drop 0193/0194.
4. **Then the remaining fusible candidates, in order:** (a) **indexer selection chain** — built and exported as **0195** (`TF_DS_IDX_FUSED`, off by default): one launch for the candidate pool's `blockMax → topk_indices → poolPick` (`round.zig`/`prompt.zig` candidate-source arms); `applyPool`/`candFast` left unfused (already single launches). (b) **drafter routing** — built and exported as **0196** (`TF_DS_FUSED_ROUTE` reused, no new switch): the drafter's gate + routing through the same fused launch, no prune/dead. (c) **ratio-2 compressor chain** — built and exported as **0197** (`TF_DS_FUSED_COMPRESS2_NORM`, off by default): the chain plus its trailing RMSNorm in one launch, decode and prompt; it supersedes 0127 when set. (d) **norm-RoPE glue** — built and exported as **0198** (`TF_DS_FUSED_ROPE_STORE`, off by default): the decode indexer key/latent/query and prompt key/latent rope-store chains in one launch each; Engram's `add2Bf16`+`_engram_gate` deliberately left unfused (three `tl.sum` reductions for two launches a pass). All four live on engine branch `fused-route` (b1033b4 + b31f3a6 + 25eeab5 + 20001bc + 44e725a), were CPU-emulation verified and independently reviewed, and are gated on the Spark exact suite in that order. GB10 bit-identity unproven; C/D's honest risk is `_rmsnorm`'s reconstructed `tl.sum` order (D's harness caught and fixed an E4M3 saturation transcription error, now host-tested). C saves ~3 launches a round and D ~20 (0.05–0.2% of a round) — only worth keeping if (2) shows real device-side launch gaps. [Scope](docs/fused-c-d-scoping.md) · [Report](../p0197/report.md)
5. **Do not touch:** the kit cubins (experts GEMMs, attention core, head, Engram lookups) and the all-gathers — not for lack of source (it is public; [audit](docs/kernel-kit-provenance.md)), but because a rebuilt/replaced kit is a new binary whose bits only the Spark exact suite can adjudicate, and NCCL is a linked binary. A kit rebuild, if ever wanted, goes build → 39-case exact suite → adopt or drop. The RDMA ring and the repo-owned host paths are ours. Cap 12,1 remains unjustified. `TF_DS_DRAFT_BLOCK=6` image stays production.

**Final pinned-image result.** For mixed short decode and persistent chat, use the deployed recipe with keyed drafts and the 32 MiB tokenizer cache. At 3K × 4 the cache alone stayed within 0.2% of deployed throughput. For cold-prefill-heavy traffic, the screened eight-switch bundle with unkeyed drafts delivered 2,070–2,072 input tok/s at 32K (+5.1–5.5%) and 2,021 at 100K (+3.9–4.3%). Four-stream warm output fell 5.9–7.6% at 32K and 5.5–11.6% at 100K. All 24 full-panel outputs matched both deployed controls. [Full crossover](runs/overnight-opt/RESULTS.md#full-32k-and-100k-confirmation)


At 08:57 Pacific both ranks were back on the original configuration with zero connected or waiting clients. The restored service matched all eight saved replies. [Final health](runs/overnight-opt/followthrough02/final-health.json) · [Validation](runs/overnight-opt/followthrough02/config-crossover/production.validate.log)

**Production since November 18, 17:19 UTC.** Both ranks run the deployed recipe plus `TENSORFOLD_TOKEN_CACHE_MB=32`. The image, keyed drafts and every other switch are unchanged, and the other seven candidate switches stay unset. The setting lives in the head's `.env` and in this checkout's `.env`, which `just sync-head` pushes; the head's previous file is `~/dsv41-env-backups/.env.bak-20261118-pre-token-cache`, outside the checkout. `start.sh` exited 1 on the usual first-request smoke failure and one retry answered. After the restart all eight saved replies and all eight concurrent-to-solo comparisons matched, and each rank's spill tier kept its 61 points. A three-turn probe on a 34.6K-token history spent 29 ms, 15 ms and 1 ms in server encoding. [Restart](runs/deploy-token-cache-20261118/restart.log) · [Validation](runs/deploy-token-cache-20261118/production.validate.log) · [Cache probe](runs/deploy-token-cache-20261118/cache-live.jsonl)

Any `./start.sh restart` now includes the cache. For a pre-cache control arm export `TENSORFOLD_TOKEN_CACHE_MB=0`, which overrides `.env`. The campaign drivers in `runs/overnight-opt` and `runs/review-round` refuse to start while any candidate switch is present in production, whatever its value. Update their baseline check to accept the cache before the next campaign, and leave the `.env` line in place.

The [morning benchmark report](runs/overnight-opt/MORNING.md) summarizes the workload-specific configurations and hardware ceilings. On the pinned serving image, the eight warm switches changed four-stream output by **−3.5% at 3K** and **−1.2% at 300K** against two deployed controls. All 20 warm outputs matched. Four-stream 300K median TTFT improved 4.2% and 11.2% across repeats. Use the deployed recipe for short four-way throughput and the separately measured token cache for immutable long-chat latency. [Direct baseline comparison](runs/overnight-opt/RESULTS.md#same-image-production-crossover-at-3k-and-300k)

On the repaired experimental image, snapshot trimming held 84 ordinary histories in 299 MiB. At the same 512 MiB cap the control evicted 12 points and required eight disk restores. All 96 responses matched. A switch to the trimmed layout starts a fresh compatible spill namespace. [Pressure trial](runs/overnight-opt/RESULTS.md#trimmed-ordinary-snapshots)

**Production since November 18, 22:21 UTC.** Both ranks run `tensorfold-dsv41:exl3-tp2-ea522d353626` with `TF_DS_DRAFT_BLOCK=6` and the token cache. The image is the pinned series plus the draft-block patch, `zig/patches/0192` on the head and in this checkout's longer series. Its two executables were cross-built on the workstation and layered over the pinned image; the recipe's setup accepts the image by its `tf.patches` label, so a plain `./start.sh restart` starts it. The switch lets a stream's drafter pass offer six drafts after a round that kept four rows, at one or two streams. On Baseten's Tetris prompt tokens per round rose from 4.76 to 5.13 against their 5.6, with throughput up 2.5% there and 2–3% on the code and JSON prompts. Prose and four or more streams are unchanged. After the restart all eight saved replies and all eight concurrent-to-solo comparisons matched, and the 15-row probe reproduced the gated arm row for row. A 111K-token prompt with a copied continuation ran 108 of 112 rounds at seven tokens alone and gave the same output hash as four copies run together on the five-draft path. The new image starts a fresh spill namespace; the old one is still on disk (2.3 GB on the head) and unread. [Results, patch and adoption](runs/draft-block-20261118/RESULTS.md) · [Validation](runs/draft-block-20261118/deploy/validate.log)

To turn the switch off, remove the `TF_DS_DRAFT_BLOCK` line from the head's `.env` and this checkout's, then `./start.sh restart`; the image can stay. Switch-off was not run on the pair with this exact binary: the three earlier builds matched control with it off, and this build reproduced the earlier refined arm on 15 of 15 rows. To go back to the pinned image as well, also remove `zig/patches/0192-dsv41-tf-ds-draft-block.patch` on the head. `exl3-tp2-234f9080b326` is still on both ranks.

**Production since November 19, 02:22 UTC.** `.env` on the head and in this checkout also sets `TF_DS_ENGRAM_LEAD=1`, turned on at colum's instruction as a provisional default; the image and every other switch are unchanged. Each Engram layer's prompt rows are read a chunk ahead. In the November 19 narrow run it raised cold prefill on natural text 4.9%, 5.0% and 4.5% at 8K, 32K and 100K, with decode at one and four streams and every reply unchanged. After the restart both ranks showed the lead banner with no row cache, all eight saved replies and all eight concurrent-to-solo comparisons matched, and four cold 32K natural-text prefills on production's own cache ran at 2,061–2,069 tok/s. **Full testing is owed at the next full benchmark round:** the narrow run was one block per arm with no soak, no 300K prompt, no exact panel and no long-lived deep-session run, and the lead reader was on during the November 13 slowdown (with the 256 MiB row cache). [Narrow run](runs/suite-20261119/RESULTS.md) · [Deploy record](runs/deploy-lead-20261119/deploy.log) · [What is owed](to-benchmark.md#current-evaluation-order)

To turn it off, delete the `TF_DS_ENGRAM_LEAD` line from the head's `.env` and this checkout's, then `./start.sh restart`; the `.env` from before the change is `~/dsv41-env-backups/.env.bak-20261119-pre-lead` on the head. Turn it off if cold prefill falls as the process ages (on November 13, with the row cache also on, the slowdown set in after roughly 400K uncached tokens) or if either rank's `MemAvailable` trends down. Do not add `TF_DS_ENGRAM_DEDUPE` while it is on: that pair broke the ranks on November 12.

**Draft ladder, November 19: built, measured, not adopted.** `TF_DS_DRAFT_LADDER` gates a seven-draft block's last draft by each stream's own record. On 21 code-like prompts it and plain block 7 are both 0.4% faster than production's six (±0.5% across prompts), and six is 3.2% faster than five. Production is unchanged. The bench image `tensorfold-dsv41:bench-draft-ladder-0bd63fda950f` is on both ranks and the source is branch `draft-ladder` in `~/Workspace/tfref-draftblock`; the patch is not in `zig/patches`. [Results](runs/draft-ladder-20261119/RESULTS.md)

Do not run `just sync-head` as things stand. It pushes this checkout's 134-patch series over the head's 118 (which ends at 0192). The recipe hash would change, and the next restart would stop the server and build from source on the head, a build that has not been run. The head's copies of `continue.md`, `best-runtime.md` and `decisions.md` are older than these; copy single files instead. The campaign drivers in `runs/overnight-opt` and `runs/review-round` expect that pinned tag and unset candidate switches, so they refuse on the image as well as the token cache.

## Serving configuration and benchmark images

The November 17 depth campaign recorded both ranks on `tensorfold-dsv41:exl3-tp2-234f9080b326`. It includes patches through 0171 and the build fixup 0171z. Production uses applied T13 pruning, L2 prefetch, KV sharding with carry and twins, spill, replay floor 1, prefill slice 5, graphs with cap 128, eight streams and a 2,101,248-row pool. The eight switches in the candidate below were unset on both ranks at that time. Since November 18, 17:19 UTC production also sets `TENSORFOLD_TOKEN_CACHE_MB=32`; the other seven remain unset. Since 22:21 UTC it runs image `exl3-tp2-ea522d353626` with `TF_DS_DRAFT_BLOCK=6`. Compare the command, environment and image before any new run. [Rank 0](runs/depth/end.rank0.id) · [Rank 1](runs/depth/end.rank1.id)

The `best` candidate adds `TF_DS_ENGRAM_LEAD=1 TF_DS_ENGRAM_SCALES=map TF_DS_ENGRAM_STRICT=1 TENSORFOLD_TOKEN_CACHE_MB=32 TF_DS_DEVICE_ACCEPT=1 TF_DS_COMPRESS_SIDE=1 TF_DS_PDL=1 TF_DS_FUSED_COMPRESS2=1`. Three rotated passes found +6.9%, +5.3% and +4.1% diverse uncached prefill at 8K, 32K and 100K, and about +0.5–1% short decode. The soak sustained about +4% prefill. The shallow-only attribution put the four GPU switches at −6.8%/+0.8% four-user output relative to adding the tokenizer cache and three Engram settings alone. The tokenizer cache alone stayed within 0.2% of deployed four-user output. [Pinned attribution](runs/overnight-opt/RESULTS.md#pinned-configuration-attribution-and-keyed-drafts) The first smoke succeeded on 21/21 starts with device accept and failed on 29/29 without it. The scales mapping adds roughly six seconds to the first request compared with I/O, including with map-warm. [Combination results](runs/combo/RESULTS.md) · [Cold-start and soak results](runs/combo2/RESULTS.md)

The earlier odds screen used a different image with bench fixups 0184z (initialize 0169's `rd.hot`) and 0186z (forward declaration). 0190 crashed at default flags and was excluded. 0185/0186 produced garbage on smoke, 0188 hung at 8K, and 0184 alone or with 0182 failed the exact record. Top-k screened neutral. Adding gather overlap cost ~0.8% at 32K. T15 had no applied expert cap in these schedules. Reuse the pinned serving image for runtime comparisons. The local exported series and the head's deployable patch tree differ. [Odds and phase D](runs/combo4/RESULTS.md) · [Second-pass manifest](runs/bench-prep2/SERIES.md)

The historical November 13 traffic window had about 650 requests, prompt p50 ~129K/p90 ~257K, and ~98.4% cached at the median. Decode accounted for about 86–89% of summed request time, which counts overlapping requests separately. This trace predates 0136. A fresh analysis after the two startup smokes found 652 completed requests with usable timing fields. An estimated 121 arrivals fell within the first five seconds after another lead’s long prefill, using a 0.2 s boundary margin. Five disjoint November 18 windows totaling 2 h 20 m 48 s contained zero completions, so current traffic weights are still unknown. [Coverage](runs/overnight-opt/kv/quiet-summary.json) [Overlap and selection](runs/engine-followup/RESULTS.md#traffic-coverage) [Production window](runs/overnight/break-1600Z.rank0.log) · [Traffic coverage](runs/depth/RESULTS.md#item-1-production-traffic-on-a-healthy-config)


Leave 0121 exact resume disabled even on its supported configuration. The ordinary grid reproduced all four cold appends, while exact resume changed both greedy outputs following 8,201-token histories. The repaired full-end identical-resend result applies to 0170. [Supported-mode comparison](runs/overnight-opt/RESULTS.md#supported-0121-resume-against-independent-cold-appends)

## November 17 review round

On the pinned image, the sampled production → best → production trial reproduced a warm TTFT improvement of ~5% at 4K and 100K. An identical-input replay matched all 14 warm token hashes across both depths. At 100K paired TTFT fell 4.9%. Sampled decode differences were within ~2% at either depth. At 4K, 257-token bursts ran one lane at a time, with the last request waiting 14.7–15.0 s. At 258 tokens all four ran together, the last waited 3.8 s and the batch finished ~35% sooner. After prefill emits the first token, these budgets fall either side of the scheduler’s 256-tokens-left finish rule. [Threshold evidence](runs/review-round/RESULTS.md#one-token-boundary)

On a 250K three-message conversation, enabling only `TENSORFOLD_TOKEN_CACHE_MB=32` reduced warm TTFT by 206–229 ms across 100/500/2,000-token appends. The nine matched trials used the same cached grid position and generated the same tokens in both controls and the candidate. This is the clearest configuration gain for immutable long chat histories. [250K trial](runs/review-round/RESULTS.md#token-cache-on-a-250k-conversation)

The round ended at 00:15 UTC on November 18. Both original serving containers were restored, all eight saved validation replies matched, and the benchmark switches are back at their production values. [Final validation](runs/review-round/t15-r8-fixed/restore.validate.log)

## November 18 runtime configuration

Production is the deployed recipe plus the 32 MiB tokenizer cache; the eight-switch bundle is a workload-specific option. Keep current T13 pruning with no expert cap, L2 prefetch on both ranks, and the default fetch-mode KV decode. Clean 300K four-stream physical rounds took 79.68 ms with fetch and 80.34 ms with owner. The isolated 32 MiB tokenizer cache saved 445–508 ms median TTFT on 500K appends. Cap 12,1 raised eight-way sampled task throughput from 74.7 to 88.3 tok/s, but exceeded the held-out NLL margin and changed all eight greedy answers between solo and concurrent service. Disabling prefetch increased matched physical-round time. [Runtime configuration](best-runtime.md) · [Controlled results](runs/engine-followup/RESULTS.md)

Two independent benchmark builds improved different queue cases. On retained 300K sessions the 500 ms held-prefix deadline cut maximum wait from about 25 s to 6.32 s in the balanced control/candidate/return sequence and to 6.37 s with asymmetric occupancy and reversed order. The armed 500 ms policy matched all 39 exact references. Adaptive finish grace preserved the completion time of 80-token leads and cut the slowest follower’s TTFT by ~6.1 s at 4K and 100K. With 256-token leads it delayed the first user's completion by about seven seconds. Neither experimental switch has been deployed. [KV placement](runs/engine-followup/RESULTS.md#shorter-held-prefix-patience) · [Adaptive admission](runs/engine-followup/RESULTS.md#adaptive-admission)

At the end of the earlier campaign on November 18 at 06:18 UTC both rank signatures matched the original production configuration and all eight saved replies matched. At 06:30 UTC the serving pair was healthy, with zero connections or waiters. [Final validation](runs/engine-followup/owner-timing02/production.validate.log) · [Health](runs/engine-followup/final-health.json)

## What remains worth measuring

**Screen, November 18–19.** A short screen on the production image left one item queued and one parked. **Queued:** `TF_DS_ENGRAM_LEAD=1` alone raised cold prefill 7.5%, 11.6% and 10.1% at 8K, 32K and 100K with identical replies and no decode cost; its full round is the first row of the [evaluation order](to-benchmark.md#current-evaluation-order). **Parked as a future idea:** expert caps `32,1` and `24,1` raised output 8% and 13% at four mixed streams and 5% and 15% at eight Tetris streams with single-stream likelihood inside the margin, but only 2 and 1 of 8 concurrent replies then equalled their solo replies and quality at eight streams is unread. A per-stream cap would not bind at those sizes, and a cheaper first pass to choose experts was built (the fused-route patches above) and measured 11–19% slower and not exact on November 19 ([results](runs/fused-bench-20261119/RESULTS.md)). Pruning off did not raise draft acceptance and cost 11–13% of round time. Each arm was one fresh process. [Screen and source reading](runs/screen-20261118/RESULTS.md)

1. **Confirm the scope of the guarded 500 ms held-prefix rule.** At the real 2,048-token persistence floor, the guard preserved a 296,960-token session whose follower copy was missing. The burst plus three returns took 24.671 s guarded, 29.163 s with ten-second patience and 179.827 s with an unconditional 500 ms. The reverse-order 1,024-token comparison saved 144.924 s end to end against plain 500 ms. With all histories backed the guarded 384-token sequence took 5.890 s longer, with different last-fill preparation before the matched pressure state. With identical all-warm preparation, the reverse-order repeat took 29.356 s guarded and 30.144 s plain; the earlier healthy penalty did not reproduce. A delayed peer also blocked the guard for 10.010 s in host loopback. General serving retains ten-second patience. A bounded retention prototype passed simultaneous two-rank CPU store tests. Its asynchronous acquisition and failure invalidation still require serving integration. [Production-floor crossover](runs/overnight-opt/kv/RESULTS.md#proof-of-a-committed-point-on-the-pair)
2. **Keep the original index bucket.** Fixed-row tests measured indexer steps across 131,072 and 262,144 tokens. With serving graphs and speculation active, finer buckets slowed both four-stream observations at 132K. At 263K the differences depended on placement and the returning control. Retain the default bucket. Against the midpoint of both defaults, finer buckets changed 132K throughput by −6.20%/−0.75% and 263K by +5.49%/+1.57%. Control drift and unequal 263K placement preclude a general serving gain. [Serving pilot](runs/overnight-opt/RESULTS.md#finer-index-serving-pilot)
3. **Weight admission and caching against representative arrivals.** A 142-token natural stop with three staggered 300K followers reproduced the stream-latency cost of adaptive grace. Short-held fixed grace reduced maximum queue to 4.77 s while preserving the 3.48 s lead. Short-held adaptive grace stretched that lead to 10.82 s. The remaining uncertainty is how often real users face those arrival and cache conditions. [Natural completions](runs/overnight-opt/RESULTS.md#natural-completion-at-300k) The two recent intervals between benchmarks had zero completions. The older trace estimates overlap but omits the requested token budgets and per-part retained rows. [Traffic coverage](runs/engine-followup/RESULTS.md#traffic-coverage)
4. **Retain uncapped pruning.** Cap 12,1 failed the source-paired likelihood margin on two independent 20-source cohorts. With eight live streams it generated SQL/tool output faster but lost the sole executable SQL answer and completed the same two tool episodes as current. Higher-budget hard tasks at widths one and four remain exploratory for a relaxed quality target. [Width-eight results](runs/overnight-opt/prune/RESULTS.md)
5. **Keep full prompt ends workload-specific.** The original exact resend changed four of four 128-token outputs. The repaired follower reproduced all four, a 100K resend, four long score comparisons and 39 frozen cases. Of 409 external prompts at or above 100K, all lengths were unique. Across all 652 requests, possible resends represented at most 0.283% of recorded prefill time from earlier requests in this interval. The separate 0121 switch requires a configuration with spill, KV sharding and prefill slicing disabled. [Correctness and traffic](runs/overnight-opt/RESULTS.md#retained-prompt-state-and-identical-resends)

For patch numbers, build gates and remaining engineering work see the [work packages](todo.md), [first manifest](runs/bench-prep/README.md) and [second-pass manifest](runs/bench-prep2/SERIES.md). The exported recipe now includes 0190z for the integrated build and checkpoint corrections. The 0190za test patch corrects two fast fake-scheduler fixtures and an undefined optional in the off-grid test. Clean verified Debug and ReleaseSafe suites now each pass 769/769. The repaired default benchmark image passed 39 frozen cases and all eight saved replies. Optional switches retain their individual correctness gates. [Build and replay](runs/overnight-opt/build-audit.md)

### Earlier overnight pilot

The missing-snapshot pilot reproduced the scheduler gain on the eight-switch incumbent. In the paired 1,024-token bursts the 500 ms policy saved 7.01 s of batch wall and both arms evicted the same unbacked victim. A cold revisit cost about 152 s extra. The initial fixed-row indexer pilot showed a 0.117 ms step across 131,072 tokens at one verified row. A calibrated below-L2 read counter recovered a known scan on both ranks. [Overnight report and stage measurements](runs/overnight-opt/RESULTS.md)
