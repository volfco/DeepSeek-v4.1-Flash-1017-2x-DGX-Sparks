# Decisions

Short entries: what was decided, what else was considered, and why. Code references are patch numbers under
`zig/patches/`.

## 2026-11-20 deep-context quality: configuration kept; judge quality sampled, not greedy

Change nothing in production after a reported degradation in a ~533K-token session. The configuration had not changed since the last validated deploy, the lead reader is bit-identical at 4K–530K, and cache resume equals cold prefill. The greedy probe's 530K failure (and the repo baseline's at 256K) did not survive sampling: at T=1.0 production recalled all eight codes and the sum at 530K on 5 of 5 seeds. What remains is a model property, not a switch: mid-context order slips at 530K, and 2 of 5 seeds at 64K reasoned to the 8,192 cap by over-verifying. Considered and not done: rolling back to the repo baseline (greedy it was no better, and sampled it is unmeasured); adopting upstream PR #138's `LOOP_GUARD` (unmerged, untested on a pair, and aimed at short repeated passages, which the observed runaway is not). Quality checks on this model use T=1.0/top_p 0.95/top_k 20 with explicit seeds, never greedy (upstream MiaAI-Lab#54). [Results](runs/depth-quality-20261120/RESULTS.md)

## 2026-11-19 draft ladder built, not adopted; the six-draft block confirmed

Keep `TF_DS_DRAFT_BLOCK=6` and leave `TF_DS_DRAFT_LADDER` out of the recipe. The ladder offers a seven-draft block's last draft only when this pass's survival through six drafts, times the stream's running share of right seventh drafts, reaches 0.43. On 21 code-like prompts it is 0.4% faster than six (±0.5% across prompts) and equal to plain block 7 (−0.03 ± 0.14% paired); its keep rule asked for 1%. The cost it cannot avoid is the seven-row pass: with the seventh draft unused that pass is 0.9% slower than six by rounds and verified drafts, and the seventh draft earns back about that much. A width ladder that picked six or seven perfectly per prompt would have reached 0.9%. The same arms measured six 3.2% faster than five (slower on none of the 21 prompts, equal on one by rate), which is the first broad measurement of the November 18 deployment. The ladder's replies equalled production's on validation and all 25 probe replies; the 39-case exact panel was not run on it. Considered first and dropped: a per-stream ladder with trial rounds at seven, because scoring a draft from the next round's tokens needs no trials. The bench image and the `draft-ladder` branch are kept. [Results](runs/draft-ladder-20261119/RESULTS.md)

## 2026-11-19 production: Engram lead reader on, provisionally

Set `TF_DS_ENGRAM_LEAD=1` in production's `.env`, at colum's instruction, ahead of its full test. The narrow run gave +4.9%, +5.0% and +4.5% cold prefill on natural text at 8K, 32K and 100K with decode at one and four streams unchanged and every reply identical; the screen had shown +7.5–11.6% on random-word prompts. It is a `.env` line and not a recipe default, so it stays easy to remove. What it has not passed: a soak of 1.5M uncached tokens, an hour or more of process age, the long-lived deep-session run, 300K prompts, the warm tail of a deep turn and the 39-case exact panel. It was on during the November 13 slowdown, together with the 256 MiB row cache, which is off here. Those tests are owed at the next full benchmark round, with production as the lead-on arm and a lead-off control. The draft-widening thresholds 3 and 2 and the seven-draft block were measured in the same run and not adopted: no threshold beat the control gap, and block 7 gained on three code prompts and lost on two others and at two streams. [Narrow run](runs/suite-20261119/RESULTS.md) · [Deploy record](runs/deploy-lead-20261119/deploy.log)

## 2026-11-18 production: six-draft block deployed

Run `TF_DS_DRAFT_BLOCK=6` with the default threshold. Of the four variants measured it is the only one with no cell below the control spread: Tetris +2.5%, the code and JSON prompts +2–3%, two code streams +2%, prose and four or more streams unchanged, every reply identical. Block 7 gained 9.5% on one code prompt and lost 0.7% on one sampled seed. Widening every pass cost about 1% on prose and 2.5% at four streams. The image is the pinned series plus one patch, promoted from the gated cross-build so the recipe starts it unaided; the spill namespace starts cold. A lower threshold and block 7 for code-heavy use remain unmeasured options. [Results and adoption](runs/draft-block-20261118/RESULTS.md)

## 2026-11-18 production: tokenizer cache deployed

Deploy `TENSORFOLD_TOKEN_CACHE_MB=32` on both ranks and change nothing else. It is the only candidate switch that held up on the pinned image at every depth tested. It saved 206–229 ms of first-token latency on 250K appends and 445–508 ms at 500K, and stayed within 0.2% of deployed four-stream output at 3K. The three Engram switches stay off: adding them to the cache changed 3K four-stream output by −4.1%/−2.1% against deployed. Those processes were under six minutes old, and the steady-state cost is unmeasured. The four GPU switches stay off because no pinned-image comparison shows a gain from them. Unkeyed drafts lower tokens per round, and snapshot trim needs a different image. The restart at 17:19 UTC kept the image and the spill namespace, and all eight saved replies matched. [Deploy evidence](runs/deploy-token-cache-20261118/production.validate.log) · [Pinned attribution](runs/overnight-opt/RESULTS.md#pinned-configuration-attribution-and-keyed-drafts)

## 2026-11-18 follow-up benchmarks

Prefer adaptive grace for a near-finished lead with queued followers on the benchmark image. It reproduced the five-second control’s lead completion at 4K and 100K and cut the follower’s wait for first output by ~6.1 s. Fixed 500 ms completed the followers’ first tokens about half a second earlier and delayed the lead by about three seconds. Maintain the general incumbent pending representative arrivals and a service objective. The 300K queue slowdown reproduced in four peak states, including an uneven partition split. Five of twelve revisits restored the complete prefix from disk. [Follow-up campaign](runs/engine-followup/RESULTS.md)

Favor the 500 ms sharded held-prefix deadline for the tested four-request 300K pressure case. On balanced and asymmetric partitions it reduced maximum queue from about 25 s to 6.32–6.37 s, raised throughput ~18%, and preserved burst hashes, victim cache cuts and the armed 39-case exact output. Isolated builds tested the KV policy and adaptive scheduler separately. [Planner evidence](runs/engine-followup/RESULTS.md#shorter-held-prefix-patience)

Keep fetch-mode KV decode. At matched low occupancy, 300K four-stream rounds took 79.68 ms for fetch and 80.34 ms for owner. The previous owner arm following exact was a high-occupancy run with very little overlap. Its timing is excluded. Prioritize fixed-row compressor/indexer measurements at depth. [Phase attribution](runs/engine-followup/RESULTS.md#eager-phase-attribution) · [Owner/fetch replay](runs/engine-followup/RESULTS.md#fetch-versus-owner-decode)

Keep L2 prefetch on and retain the current uncapped T13 prune rule. Disabling prefetch lengthened median matched physical rounds by 5.6–6.3% for one stream and 2.2–2.5% for four. Cap 12,1 improved eight-way sampled task throughput by ~18%, but the held-out NLL increase exceeded the quality margin and all eight deterministic replies changed between solo and concurrent runs. Keep the 32 MiB tokenizer cache for immutable long chats. At 500K it saved 445–508 ms median client TTFT across three append sizes, including 436–437 ms of encoding. [Configuration record](best-runtime.md) · [Controlled measurements](runs/engine-followup/RESULTS.md)

## 2026-11-18 admission and runtime configuration

Keep the eight-switch `best` recipe as the general incumbent. The 500 ms finish grace on the benchmark image improved staggered short-reply throughput by 45% at 4K and 41% at 100K with ~2K uncached tokens. At both depths the first reply took over seven seconds longer to finish. Zero grace produced the fastest 4K bursts and the longest active-stream pauses. Track queue-oriented and batch-throughput choices separately until a service objective settles the trade-off. [Configuration record](best-runtime.md) · [Trial and controls](runs/admission-grace/RESULTS.md)

## 2026-11-17 benchmarking decisions

Keep the deployed image and settings as the control. After the cold-prefill gains, a sampled trial with a production return block and identical-input replay found ~5% lower warm TTFT for the eight-switch `best` configuration. Its sampled decode differences changed direction with depth. Exclude finer buckets from timing rankings until exact passes. The original T15 screens omitted the expert cap. The corrected eight-row trial with the cap held fixed found a 0.0–0.5% median decode-time reduction, below the timing threshold. [Handoff](continue.md) · [Benchmark matrix](to-benchmark.md)

The new controlled 250K chat trial favors the 32 MiB token cache on persistent multi-turn histories. Prioritize the 257/258-token admission discontinuity and the controlled 300K placement-pressure case for engine changes. [Review round](runs/review-round/RESULTS.md)

Defer expert hot-cache and 0196 after W4. The 221–222 GB/s estimate applies to eager MoE. Prioritize admission, warm-turn encoding and acceptance at depth. The existing off-grid and score-set runs establish output changes and regression coverage; quality bounds require independent continuations and concurrent workload evidence. [W4](runs/w4/RESULTS.md) · [Depth](runs/depth/RESULTS.md)

## 2026-11-16: the benchmark-gap work packages land as `0129`-`0145`, off by default, unrun

- Scope: major (seventeen patches: the Engram proposed rows of `to-benchmark.md`, W1/W4/W5/W10 of
  `docs/benchmark-gaps-and-moe-review.md`, and the two second-pass instruments).
- Considered: (a) land them on `series-0119` as the review's rules say, (b) land them on
  `decode-step-structure` (the current `zig/patches` tip, including `0123`-`0128`), (c) keep them in a scratch
  folder only. Chose (b), then export through `scripts/port-patches.sh` into `zig/patches/`: the docs' rule exists to
  avoid conflicting with the in-flight series, and basing on its tip achieves that directly (two patches needed a
  hand-resolved conflict with `0134`/`0141`, both taken as "keep both"). The export path is (a) + (c): the script
  verifies on a clean `TF_REF` checkout that the patches apply in filename order and reproduce the branch, which
  `git format-patch` alone does not.
- Numbering: the review said "number new patches from `0130`", but the export numbers from the last hand-written
  patch and produced `0129`-`0145`. Renaming the generated files would break the script's own bookkeeping, so the
  files keep the numbers the script gave them and every doc carries the mapping (plan number → landed number).
- Not landed: `TF_DS_ENGRAM_SPLIT=2` (`LEAD` already owns the second AIO ring `paio2`/`lead_ring`, and `aioGather`
  writes contiguous id slices into contiguous destinations, so striping needs per-row destinations or a bounce copy —
  not a minimal change) and the streamed-reply text scan (W10-b; the allocator question is unresolved and the change
  must stay bit-identical for stop sequences).
- Gates: host only. `zig build test` green per commit and `zig build` green after the merge (that is what compiles the
  batch driver and the server, which the host test root does not reach). Nothing has run on the pair; the W1 row-cache
  fix and `0143` (sequential class) carry their pair gates in `to-benchmark.md`.
- Consequence: `zig/patches/README.md` gained seventeen rows and the image tag changes, so the next `./start.sh`
  rebuilds; `runs/bench-prep/` (README, `run_arm.sh`, `r8.requests.json`, `env-vars.md`, `overnight-fold.md`) is the
  other agent's entry point. `0142`'s `.cu` is UNVERIFIED-BUILD — if the image build fails on `probe.cu`, drop that
  patch before rebuilding.

## 2026-11-15: pair's first request after a restart fails deterministically (pre-existing, both images)

- Scope: operational (the smoke step in `start.sh`; every matrix/ultrafast arm restart).
- Symptom: `start.sh`'s 5/5 smoke test now fails on **every** restart, deterministically: the very first
  request after a fresh process exits with `error: cuLaunchKernel: CUDA_ERROR_INVALID_VALUE`, logged as
  `ended ... reason=error (CudaFailed) prompt=8 tokens=1`, HTTP 500. The identical request retried afterwards
  passes (HTTP 200, `sha=207398c96af1`), and every arm's later requests pass — replies are unaffected.
- Ruled out: not caused by the `0123`-`0128` series — reproduced on `exl3-tp2-f3f69b951cd3` (the pre-series
  campaign image) with the same flags, same signature, 2/2 restarts. The ultrafast campaign (2026-11-15, image
  `f3f69b951cd3`) passed the same smoke until at least ~18:10Z; the failure appeared later the same day, with no
  reboot or driver change visible in the logs (CUDA Forward Compatibility 13.3 / driver 580.159.03 then and now).
- Why it matters now: `bench_matrix.sh` treats restart rc≠0 as RESTART-FAILED and aborts the arm, so **no matrix
  arm can run while this persists**. The arms reported here were driven with `just restart-head` + manual `bench.py`
  against `http://100.95.49.21:8000` instead. `exact.py`/`validate.py` are unaffected (they never asserted on the
  smoke, only on replies).
- Unknown: the mechanism. It is deterministic per fresh process and benign in effect. Needs either the pair's
  timing (reboot the pair once and retest before any bench or campaign), or an engine-side look at what the first
  round's first `cuLaunchKernel` does differently (the log names no kernel; the failure hits before any token).
  Not root-caused; not blocking; recorded so nobody re-derives it the hard way. Until it is, `start.sh`'s smoke
  gate stays red and the harness's arms cannot be used as-is.

## Decode-step structure (`0123`-`0128`): what landed from the plan's "To explore" A-E

- Scope: major (six switches + one design note; every one is off by default and bit-identical to the base; nothing
  ran on the pair).
- Considered: (a) the plan's A-E end state (device accept chained into the pass, one frame, one graph a round, side
  streams, PDL everywhere, fused kernels, Engram staging) as one implementation push — rejected: the rank-0 profile
  (`runs/ultrafast/prof/*.log`, graphs on: send 0.005/0.006, absorb 0.26/0.29, logits 0.13/0.07, sample 0.00 ms of a
  34.4/53.6 ms round; forward 89%; eager MoE 16.1/31.5 ms of 34/56; a_idx 0.57/0.61; Engram 0.47/0.57) says the
  host path is ~1% of a round and the one-frame stretch cannot pay; (b) implement only what the operator picked per
  item, each behind its own switch with its own gate; (c) drop A-E as unproven — rejected: the operator asked for the
  implementation with the profile as the arbiter of keep/drop, not as a reason not to write the patches.
- Decision: `0123` `TF_DS_ROUND_PROFILE` (the served process's profile at `/metrics`; attribution only); `0124`
  `TF_DS_DEVICE_ACCEPT` (chain windows compute their keep on the device — the foundational piece of Baseten's on-device
  accept; the one-frame verify+pass stretch stays unbuilt: the depth rule, the cuts and the T26 keys are rank-0 host
  state); `0125` `TF_DS_COMPRESS_SIDE` (the compressor chain on its own stream); `0126` `TF_DS_PDL`
  (`griddepcontrol` at every hand-written kernel's entry + the attribute on their launches + `tools/nsys_breakdown.py`
  as the gap measure; the kit's Triton kernels already launch PDL); `0127` `TF_DS_FUSED_COMPRESS2` (the ratio-2
  compressor chain in one kernel; the review-caught cross-block race in drafted verify windows closed by resolving
  partners against `rslot` in-kernel); `0128` `TF_DS_ENGRAM_BOTH` (both Engram layers' rows land at the round's first
  Engram cut; three stretches a round become two). Item E's full staging was **not** built: without GPUDirect Storage
  the libaio completion is only observable on the host, so no kernel can wait on it — option A needs a cuFile demo on
  the pair plus a measurement that the landing is on the critical path after `0128`; option B (poll a pinned host CQ
  word) is rejected on risk/reward; the analysis is in `docs/engram-staging.md`. Also checked and dropped: the
  drafter's blocks already group their stages' `wkv` into one `grouped` call at absorb time (`draft.zig:505`), so the
  `mtp_wkv` analogy is spent.
- Why: the plan's own sizing gate ("if send + logits + sample + pass is a small share of step_ms, stop here") answers
  against the one-frame stretch and item C's urgency — the profile, not optimism. Each patch is one idea behind its
  own switch, off by default, host-tested only, and owed the pair gates (`exact.py` IDENTICAL, then the smoke, then
  `validate.py --compare`) before any is called done. Export: `/home/colum/Workspace/tfref-next-handover/` (format-patch,
  applies onto `series-0119`; verified by tree hash on a scratch clone). The 0119/0121 run was live while these were
  made, so no `.patch` file entered this checkout's `zig/patches/` — image hash unchanged; the six README rows and the
  `to-benchmark.md` section ride the working tree uncommitted, like the 0119-0122 series. `AGENTS.md`'s rule that the
  hardware smoke is part of landing any series applies unchanged: these are not landed until they pass it.

## Baseten-ultrafast series 0119-0122: four switches, off by default, unmeasured

- Scope: major (four new switches; `0119`, `0120` and `0121` are bit-safe, `0122` is LOSSY by construction).
- Considered: (a) the plan's four items as one patch each, all off by default and host-tested only; (b) the two
  identity-safe items only (`0119`, `0120`), deferring resume and speculative sampling; (c) all four plus resume phase B
  (keeping the reply's decode-written rows). Chose (a): phase B is LOSSY and the plan itself gates it on phase A's
  numbers, so it stays a proposal.
- Decision: `0119` tokenization cache (`TENSORFOLD_TOKEN_CACHE_MB`, rank 0), `0120` teacher-forced scoring
  (`TF_DS_SCORE=1` plus `tools/score.py`), `0121` exact-position resume phase A (`TF_DS_EXACT_RESUME=1`, both ranks,
  in the fingerprint so a mismatch refuses at the hello; `TF_DS_DEBUG_SPLIT` for step 0), `0122` speculative sampling
  (`TF_DS_SPEC_SAMPLE=1`, both ranks, in the fingerprint, refused with `TF_DS_T26_KEYED_DRAFT`). The plan's numbers
  assumed `0118` was free; this repo already had `0117` (energy) and `0118` (LoRA), so the series starts at `0119`.
  Refusals keep the LOSSY and structural cases apart: `0121` refuses the spill tier, `KV_SHARD`, `TF_DS_PREFILL_SLICE`
  and image prompts and keeps off-grid states in memory; `0120` refuses anything but `n == 1`, `stream` false,
  `temperature` 0, no other live stream.
- Why: each switch is one idea with its own gate, so a failed idea can be dropped without touching the others, and the
  ground rules are the series' (off by default, host tests, identity gate, then a hardware smoke). What is still
  unmeasured, by design of this pass: `0119`'s `encode_s` share of TTFT at 100K/250K/500K, `0120`'s self-consistency set
  (`tools/score_sets/` is not generated), `0121`'s step 0 (`TF_DS_DEBUG_SPLIT` against `--logits-sha 1`) and its
  cold-equals-warm gate (`tools/resume_exact.py`), and `0122`'s distribution gate (`tools/samp_test.py`). Nothing ran on
  the pair.

## 0122: speculative sampling needs the drafter's full-vocabulary normaliser

- Scope: major (the shape of the `0122` patch; a sampled stream's reply changes distributionally, not bit for bit).
- Considered: (a) a greedy draft as a point-mass proposal — accept with probability p(x), residual p minus x: exact and
  needs no normaliser, but its acceptance rate is p(argmax), which is the rate today's greedy drafting already gets, so
  it cannot beat `0054`; (b) the textbook rule with q = the drafter's `softmax(logit / T)` over the full vocabulary,
  which needs `sum exp(logit / T)` a step on the device and across both ranks (Baseten's own design); (c) comparing two
  independent draws (draft against the target's own draw): exact with no normaliser, but its acceptance is `sum p q`,
  below (a) and `0054`.
- Decision: (b). Zq is accumulated in the noise kernel we own (`tf_ds_keyed_noise_kernel`, while the row is still
  clean), gathered peer-to-peer a step; q at the target's candidate set comes from a new `tf_ds_gather_cols_kernel` with
  the keyed Gumbel removed on the host. The accept and residual decisions rewrite `o.sampled[]` in `verifyFn`, so the
  core's `acceptPath` implements the rule with no core, engine or wire change; the drafts' key is the stream's seed with
  a draft tag, independent of the sampler's.
- Why: (a) would have been the cheapest patch and is exactly why it was rejected — a switch that cannot raise acceptance
  is not worth a LOSSY invariant. (b) is the only variant that can beat `0054`, which is the plan's own bar for keeping
  the item. The residual needs q over the target's candidates, which is why the gather kernel exists; `0054` stays the arm
  to beat and this switch should be dropped if it does not.

## 0114 segfault: `Model.rc` undefined on the cache-off path — one-line init fix

- Scope: major (a crash on every first request with default settings; caught before any benchmark counted).
- Considered: (a) bisect the seven patches on the pair; (b) root-cause from the crash trace first; (c) ship with
  `TF_DS_ENGRAM_ROWCACHE_MB` set (the cache-on path works) as a workaround.
- Decision: (b) — trace (`engram_io.get` ← `engram_aio.submitLocked` ← `aioGather`) + experiments (cache-on works;
  `ENGRAM_AIO=0` works; defaults crash) pinned it: `Model.open` builds from `m.* = undefined` (field defaults are
  dead there), `0114` assigned `m.rc` only in the row-cache-on branch, and `m.aio.?.rc = m.rc` laundered the
  undefined 0xAA optional into every ring (`RowCache.io` at +0x10 = the exact fault address). Fix: `m.rc = null`
  in `Model.open`'s init list (also kills wild `deinit`s at errdefer/close), folded into `0114`'s own segment.
- Why: (a) costs a rebuild per bisect step and the trace already named 0114's file; (c) hides a latent crash for
  anyone running defaults and leaves the wild-`deinit` paths. Why every prior gate missed it: no host test runs
  `Model.open` (GPU-only) and diffs cannot show `= undefined` construction — only hardware could. Standing rule
  added to `todo.md`: a hardware smoke is part of landing any patch series. The pass after the fix: all 9 gates
  `IDENTICAL`, `0114` signed + on its read-sum win (`to-benchmark.md` Results).

## Hit-rate patches 0110-0116: implementation choices

- Scope: minor (how the seven chosen findings landed; the finding choice is logged above).
- Considered: (a) land as implemented, review after; (b) independent review first, fix, then land; and per-finding:
  0112's supersede relation on unidentified ancestry (protect vs base behaviour), 0115's shared flag plumbing
  (per-mark bits vs keep-position set), 0113's stride anchor (positional on the trimmed chain vs turn depth).
- Decision: (b) — the review found one blocker (0113's reopen depths landed off the stride grid; the first publish
  after a restart deleted every extra) and two majors (0112's `adopt` never marked a restored state's family copy
  shared; 0116's miss split attributed `tail_window` before the token match). All fixed with regression tests
  (0113's fix is mutation-verified) before landing. Per-finding: unidentified ancestry falls back to base eviction
  (protection is never claimed on guesswork); 0115 keys on the shared cuts' keep positions (the deepest shared keep
  is triggered by the history mark, so per-mark bits cannot cover the `min_gap` case); 0113 anchors the stride in
  each point's turn depth with the root at 1 (a positional rule decays as the chain is pruned, and the depth grid
  must survive a store reopen).
- Why: the switches move eviction, spill manifests and metrics only — kept states stay exact copies and resume stays
  bit-exact — but policy bugs still cost the hit rate the patches exist to win (the review's blocker would have made
  `TF_SPILL_STRIDE` useless across restarts). Gates: host suite + 315 server tests per step, and a clean-room apply
  of the seven exports (tree-identical to the chain). `0110`-`0116` in `zig/patches/`; measurement rows in
  `to-benchmark.md`; none run on a pair yet.

## Hit-rate patches 0110-0116: scope, switches, and #4 as benchmark rows

- Scope: major (which review findings become engine patches and how they are gated).
- Considered: (a) top 6 findings + the metrics finding as patches (7); (b) top 6 only; (c) all 14
  implementable findings; (d) top 3 quick wins. #4 (`SPILL_MIN_TOKENS`): patch-unconditional vs always-on
  patch vs benchmark rows only. LOSSY sub-items (append-only compaction, image-row skip, canonical tool
  renderer): opt-in toggles vs skipped.
- Decision: (a) — `0110` shared-cut discovery, `0111` edit-recovery gate, `0112` shared/branch eviction
  protection, `0113` strided spill retention, `0114` Engram row cache, `0115` shared flag from the marks plan,
  `0116` cache metrics — each on a `TF_DS_*` switch. Behaviour switches off by default (the series' rule:
  unset behaves like the pinned engine); `0116` follows `0069` (on, `=0` off). #4 lands in `to-benchmark.md`
  as measurement rows only (measure before changing a default). All LOSSY sub-items skipped (stay in the
  review as proposals).
- Why: the seven carry the review's ranked impact and every one is bit-safe (kept states are exact copies;
  resume is bit-exact by construction); the metrics patch is the measuring stick for the other six. #4 changes
  on-disk behaviour for every deployment, so it gets the matrix's measure-first rule instead. Patches are
  built as one ordered chain over the deployed series (`fbbc9713` + 0001-0098 incl. the working-tree `0097`
  + 0103-0106), exported as consecutive diffs so they apply in filename order; host tests per patch plus a
  clean-tree sequential apply + `zig build test` validation. Implementation by parallel subagent workers in
  isolated worktrees (`0112`/`0115` flagged: adjacent `keepWith` hunks), independent review + validation
  subagents after integration.

## Prefix-cache hit-rate review: scope and deliverable

- Scope: major (what the review covers and what it produces).
- Considered: (a) prompt/prefix cache only; (b) prompt cache plus Engram row reads (Engram de-duplicated against
  `to-benchmark.md`); (c) every caching layer incl. CUDA-graph and page cache. Deliverable: report only vs report
  plus implementation. Workload: agents only vs agents + long chats vs benchmark matrix.
- Decision: (b), report only, optimized for agents + long chats up to the 1M window (incl. eviction/restart).
  Engram findings must not duplicate `to-benchmark.md`'s matrix; only gaps or corrections with code evidence.
- Why: the prompt cache is what the hit-rate metrics (`prompt_tokens_cached_total`, `usage.prompt_tokens_details`)
  measure, and Engram's read volume is directly driven by prompt-cache misses, so both belong in one review; report
  only keeps the running benchmark campaign untouched and leaves implementation picks to the operator. Engine code
  read at `/tmp/tf-engine` (`fbbc9713` + patches 0001-0098, commit `f72c1573`), deployed diff per `zig/patches/`.

## 0097 fix: DecodePool.loop called `work` as a method

- Scope: minor (build fix; no bits, no behavior change).
- Considered: (a) fix `0097-engram-fast.patch` in place; (b) a follow-up patch on top of the series.
- Decision: (a), `p.work(j)` -> `work(j)` in `DecodePool.loop`.
- Why: `work(j: *Job)` is a plain function of the pool's namespace (the sibling call in `decode` already calls it
  bare), so method syntax passed the pool where a Job belongs and the server build failed at `engram_io.zig:414`.
  The series is pre-pair and pre-PR and the image hash changes either way, so a fixing patch would only carry a
  broken intermediate. Found by the first full build of the 0094-0098/0103-0106 series (host tests do not build
  the server).

## Benchmark evidence tools (scripts/bench_matrix.sh, tools/obs_arm_metrics.py, scripts/dedupe_probe.sh)

- Scope: minor (measurement only; no engine behavior).
- Considered: (a) trust bench.py's `--metrics` snapshots for the per-arm engram histogram; (b) scrape
  `tensorfold:engram_lookup_seconds` per arm around the bench block; (c) take per-arm counts from the lab's
  OpenObserve store (60 s scrapes).
- Decision: (b) as the table's source, (c) as a cross-check, (a) rejected. The same-prompt dedupe probe flushes
  the prompt cache's disk state (the spill tier's `rank*/points` trees) between legs and prints `cached_tokens`.
- Why: bench.py truncates raw snapshots at 20k chars, past which the engram series sit; OpenObserve's scrape
  cadence is too coarse for ~4 min arms and cannot see a crash's last counters. The first probe attempt measured
  the prompt cache instead of the switch (114 -> 12 on warm cache vs 114 -> 114 cold): same-salt A/B is only
  valid with the disk cache flushed, which the probe now enforces and asserts.

## Benchmark campaign protocol (tools/bench_matrix.sh)

- Scope: major (what the to-benchmark.md matrix measures and how).
- Considered: (a) the doc's full protocol: medians of 3 alternating base/arm blocks and a per-switch `exact.py`
  record before each row; (b) one bench block per arm (reps 3), `tools/validate.py --compare` as the per-arm
  identity gate, base blocks bookending the phases; (c) single-rep triage only.
- Decision: (b), operator's call: "do not perform deep testing, just enough to get a signal on if the toggle has a
  net positive or negative effect". Sampled arms are scored on accept/tokens_per_round and greedy identity (the
  doc's "quality set" is undefined in this repo), and TRIM is skipped (it punches ~101 GB per Spark irreversibly).
- Why: (a) triples wall-clock (every switch needs a container restart) and its per-switch exact records add little
  to a sign-a-toggle +/- read; (b) keeps one variable per row and one drift bookend per phase. Deferred work is in
  `todo.md`; the cross-image identity gate is one `exact.py` record pair (old image `runs/exact-oldimg.json` vs the
  new image `--like`).

## 0094 Engram trim: punch the unread rows out of each rank's table

- Scope: minor (disk footprint; no speed, no bits).
- Considered: (a) a per-rank repacked table with rebased hash offsets (halves the file but needs a tool, a loader change
  and a new fingerprint); (b) punch holes in place (FALLOC_FL_PUNCH_HOLE) for the row ranges a rank never reads; (c) do
  nothing.
- Decision: (b), opt-in with `TF_DS_ENGRAM_TRIM=1`, and only when the file has one link.
- Why: each rank reads only its own 12 of 24 hash buckets, so ~101 GB a Spark is dead weight, and (b) frees it without
  touching the layout, the offsets, the sizes or the ranks' fingerprint. (a) buys the same bytes for a table tool and a
  loader rebase; (b) needs neither. The nlink guard keeps a hardlinked `HEAD_ENGRAM_DIR` from emptying the HF cache copy
  that a later `VERIFY_SHA` checks.

## 0095 Engram scales from a mapping, not a second I/O

- Scope: minor (read count; no bits).
- Considered: (a) keep the scale tensor resident in RAM (3.07 GB a rank); (b) map it read-only and copy rows from the page
  cache; (c) interleave weight and scale rows in a repacked table (one span a row); (d) do nothing.
- Decision: (b), opt-in with `TF_DS_ENGRAM_SCALES=map`.
- Why: (a) is 3.07 GB of anonymous memory the head cannot spare (its worst point is ~3.1 GiB above the memguard floor);
  (b) is clean page cache, which the guard counts as available and the kernel reclaims, and its worst case is the current
  4 KiB read. (c) buys the same I/O count but needs a 203 GB repack and a new identity; it can be added later on top of
  (b) without changing the switch.

## 0096 Engram dedupe within a prompt chunk

- Scope: minor (read count; no bits).
- Considered: (a) dedupe a chunk's row ids in the read-ahead and lead readers; (b) a decoded-row LRU across chunks and
  requests; (c) do nothing.
- Decision: (a), opt-in with `TF_DS_ENGRAM_DEDUPE=1`; the round path and the main thread's fallback keep the old path.
- Why: repeated n-grams are common in prose and code, and the same id is the same bytes, so dedupe is exact and cheap.
  (b) would also catch repeats across chunks but needs eviction and locking, and a shared scratch with a round's reads
  would race; the prompt cache already skips cached prefixes, which is where most cross-chunk repeats live.

## 0097 Engram host-side economies

- Scope: minor (CPU and hash start latency; no bits).
- Considered: (a) leave the host path alone; (b) reciprocal modulo, a persistent decode pool, the fallback's hash reuse
  and a ring-slot knob, behind one switch; (c) do only the modulo.
- Decision: (b), opt-in with `TF_DS_ENGRAM_FAST=1` (and `TF_DS_ENGRAM_AIO_SLOTS`).
- Why: each is expected under 1 % of a chunk, so the value is in being able to measure them; the reciprocal is tested
  against `@mod` for every prime and 5,000 values each, and every item is bit-identical, so `FAST=0` is the old engine.

## 0098 Strict Engram table validation

- Scope: minor (refusals; no bits on a good checkpoint).
- Considered: (a) make the dtype and scale-row checks unconditional; (b) refuse behind a switch and warn without it; (c)
  warn only.
- Decision: (b), `TF_DS_ENGRAM_STRICT=1`, plus an ungated bounds check on the token map.
- Why: today's checkpoint passes both checks, so an unconditional refusal would only be exercised by a future checkpoint;
  behind a switch it can be A/B'd against the warning path. The token-map bound is a no-op for every id the map was made
  for, so it needs no switch.

## 0121 exact resume: LOSSY, not bit-identical — step-0 gate found the verdict (2026-11-16)

- Scope: major (retention of the 0119-0122 series' 0121; bits).
- Considered: (a) W8 phase 1 — resume off-grid only when the warm chunk's and replay's row counts equal the cold
  pass's (the review's 3.2 candidate, kernel-class thresholds at 64/128 rows); (b) W8 phase 2 — inert-row padding to
  cold cell M (experts -1 picks, zero rows, gate GEMM at cold M, replay split at grid); (c) LOSSY opt-in with a
  quantified drift; (d) drop 0121.
- Decision: (c). 0121 phase A stays `TF_DS_EXACT_RESUME=1`, opt-in, never default; grid resume (2048) stays production.
- Why: step 0 (`docs/benchmark-gaps-and-moe-review.md` §6 W8, the plan's split-identity experiment, run on the pair
  against `--prompt-cache-gib 0` with spill off): turn 6's reply `96d0fd7dcd8e` under `TF_DS_DEBUG_SPLIT=8228` equals
  the warm-resume sha from the campaign run (float-NLL arms A-F, `runs/matrix/step0-cold-shape/`); bits move for EVERY
  off-grid cut (`DEBUG_SPLIT` 8228/9300/9380/9440/9480), in both directions, at every second-part size tested
  (16-2012). The cause is not the 64/128 kernel classes but any chunk-shape change (accumulation order under
  different batch shapes). Phase 1 as approved at the time is vacuous (an off-grid keep whose shapes equal the cold
  pass's is grid resume; no re-gate can bite). Phase 2 cannot restore equality either: the restored prefix's bits are
  whatever the parent pass's tail shape produced (turn 5's M=36 tail pass vs turn 6's M=89 cell), baked into the kept
  state no kernel emulation inside the later turn can rewrite. The full split-identity ladder: control arms bit-stable
  (t3k identical in all arms), reward sha on splits; `--logits-sha 1` prints nothing on this build (no output on either
  rank, TF_LOG=debug or not); the teacher-forced score was the working instrument (t10k 2.6198 -> 2.4719/2.5886/2.5609/
  2.5496/2.4852; t70k 2.7811 -> 2.8846/2.7893/2.8104/2.8358/2.7581). Grid resume remains the only bit-stable resume.
  Drift on agent traffic measured by the campaign: turns 6-8 of 8 replied differently from cold, deterministic 3/3
  passes; TTFT 1.31 -> 0.26 s on the affected turn. Withdrawn the claim in continue.md/todo.md that phase A was
  bit-identical (or «grid resume matches cold» as a reason exact-resume could be safe). Artifacts:
  runs/matrix/step0-cold-shape/.

## 0160-0181 the second-pass series: scope, gates and delivery (2026-11-16)

- Scope: major (a new patch series on top of 0129-0145; no bits change where marked identical).
- Considered: (a) only the plan's "Where to start" items; (b) every repo-side second-pass item;
  (c) (b) plus the LOSSY opt-ins and the indexer buckets' repo half. For M3/M4/M5/M7 (whose "step 0 that can kill it"
  gates are all unrun): mechanisms + kit contracts, M3+M4 only with M5/M7 as notes, or notes for all four. Test gate:
  no tests / host tests per commit / host + CUDA type-check. Delivery: full landing in zig/patches/0160+, patches +
  README only, or a scratch handover.
- Decision: (c) for scope; mechanisms + kit contracts for M3/M4/M5/M7 (each off by default, M3/M7 sequential-class);
  host tests (`zig build` + `zig build test`) per commit, no pair runs; full landing (the series' patches in
  zig/patches/, README rows, runs/bench-prep2/env-vars.md, to-benchmark rows, doc updates) — the twelve built ones
  landed as `0160`-`0171`, the rest of the plan's numbers are the odds (see the 0182 entry below). Numbering starts
  at 0160 (0146-0159 left free), and `scripts/port-patches.sh` gained `--first` / `--keep-generated` so the new
  series exports beside 0129-0145 without disturbing them.
- Why: the operator asked for the 2nd-pass ideas as patches ready for a live-benchmark engine, explicitly including
  M3/M4/M5/M7 and the zstd/Bloom checks; (c) is the widest set that is still patchable in this repo (kit-only items —
  larger draft block, replay key scoring, old/new key split, the buckets' kit half — cannot be patched here at all).
  The step-0 gates stay documented as gates-to-run, not gates-passed: the mechanisms are opt-in so the benchmark
  engine can run the killing experiments (probe study, W4 pure-read probe, tail_window counter) as arms. zstd ships
  as a check script (`runs/zstd-recompress/`, `just zstd-check`), Bloom as an off-by-default filter front plus its
  lock-wait check counter. Host tests per commit match how 0129-0145 were gated; running the pair is the operator's
  and the benchmark engine's step, not ours.

## 0160-0171 the second-pass series: landing and numbering (2026-11-16)

- Scope: minor (delivery mechanics of the series above).
- Considered: (a) export the lanes' commits with the plan's item numbers (gaps at 0163-0166 etc.); (b) reorder the
  M-commits to numeric and export contiguous-with-monotonic-map; (c) export contiguous in commit order with a
  plan→landed map.
- Decision: (c). `scripts/port-patches.sh --first 0160 --keep-generated` appended 12 patches (`0160`-`0171`) beside
  the kept `0129`-`0145`; `runs/bench-prep2/env-vars.md` carries the map (the `0129`-`0145` batch's own precedent).
- Why: (a) leaves filenames out of application order's numbering story and needs script surgery; (b)'s reorder
  re-applies diffs against foreign context (three conflict regions for no user-visible gain). The export verifies
  TF_REF + every patch in filename order equals the branch; the map is documentation, not machinery.

## 0167 (M3) the dead-row sentinel goes on the row's INPUT, not its pick (2026-11-16)

- Scope: major (the M3 mechanism's correctness rule).
- Considered: (a) sentinel on the dead row's sampled pick (the scout's suggestion: "a sentinel no draft carries,
  truncating acceptPath naturally"); (b) sentinel on the dead row's input token (the child match); (c) a mask
  parameter through acceptPath/conclude.
- Decision: (b). The dead row's `o.drafts` entry takes the sentinel, so acceptPath's child match breaks BEFORE the
  row; the pick sentinel stays as a second guard.
- Why: `conclude` commits the accepted drafts plus the DEEPEST PATH ROW's pick as the drawn token. With (a) a
  wrong "dead" call puts the sentinel row IN the path and would emit the sentinel as the drawn token. With (b) the
  path stops before it: a right call is identical to the unmarked run, a wrong one keeps one token of speculation
  less and the deferred token comes from the next round's recomputation of the same position (same keyed draws) -
  the reply's tokens are invariant. Pinned by the accept-truncation test in `lanes.zig`.

## The unbuilt odds renumber from 0182; the named set was already exported (2026-11-16)

- Scope: minor (numbering and docs; no code, no landed patch touched).
- Considered: (a) keep the plan numbering and let the odds take 0172-0181; (b) number every unbuilt item from 0182,
  the plan's 0172-0175 block first; (c) leave the unbuilt items unnumbered until they are built.
- Decision: (b). `0182`-`0185` = top-k, prompt-gather overlap, indexer buckets, bf16 partials (plan 0172-0175);
  `0186`-`0189` = greedy argmax, expert grid, discarded head, Markov trim (plan 0163-0166); `0190`/`0191` = Snap
  decoder-ring trim and restore lookahead (plan 0168/0169, also unbuilt); `0192`-`0196` = the follow-up build items
  (M7 batching, token-align splice, load staging + `uploadAsync`, reply per-round arena, M5's decoded-cache kit
  contract). Measurement-only items (M3's probe study, the row-cache lock check, the zstd check) keep no number.
  `just export-second-pass` now runs `--base ce4ea54 --first 0182`.
- Why: 0176-0181 looked like free slots but are the named set's *plan* numbers, and that set had already landed
  inside `0166`-`0171` (`runs/bench-prep2/env-vars.md`'s map; verified by applying `zig/patches/*` to TF_REF
  `fbbc971` in a scratch worktree: tree `43cb8bc` = `second-pass-int` `ce4ea54`). A future `0176-…patch` would be
  read as the M3/M4/M5/M7 items again, and plan 0172-0175 collide the same way with SERIES.md's rows. Leaving
  0172-0181 empty makes every number >=0182 a patch that does not exist yet. (a) keeps the collision; (c) leaves
  the manifest, todo and benchmark rows citing numbers with no code behind them.

## 0182 the fast top-k: a sampled threshold, a shared-memory compaction, and the row select kept as the fallback (2026-11-16)

- Scope: minor (one new kernel + one switch; the existing kernel untouched, both paths exact).
- Considered: (a) Baseten's own shape, one read: sample a threshold, compact the keys above it straight into `out`,
  rely on the count landing near k; (b) three reads: the sample, one compacting pass into shared memory, then the exact
  k-th among the candidates, with the eight-digit select over the row as the fallback; (c) a two-level value-range
  histogram over the row (deterministic bins, no sampling); (d) leave both top-k kernels and close the finding.
- Decision: (b), for `topk_indices` only (`topk_i64` untouched: it is not the measured indexer path). Candidate cap
  4096 a row in shared memory, the tail's `cols` buffer reusing it once the selection is done; the fallback is the
  existing algorithm, so the result is the exact top-k under the same tie rule whichever path runs.
- Why: (a) cannot promise identical bits - a sample that misreads a row (all scores equal, a sorted trend) would need
  some fallback anyway, and its buffer check costs the same. (c) is distribution-blind in a way that matters: the keys
  are fp32 bits, so bit-prefix bins degenerate on a narrow score range (8 bins for a whole octave), and a value-range
  histogram needs the row's min/max first - a third read with no better guarantee than (b). (d) leaves the one place
  the plan measured (`a_idx` at layers 24-36) alone. The three reads are what fits the 48 KB static shared limit with a
  1024-thread block and keep room for the tail's 2048-entry sort; a bigger cap would not change the answer, only the
  fallback rate. The algorithm was checked host-side against a brute force (51 rows, `runs/bench-prep2/topk-fast/`)
  because nvcc is not on this host, so the CUDA remains UNVERIFIED-BUILD until the image build.


## 0183 the gather overlap: a side stream with the post deferred at Engram layers, not a double-buffered gm (2026-11-16)

- Scope: minor (one stream + two events + the post's placement; off by default, bits identical).
- Considered: (a) double-buffered gm with the post deferred a layer (the scout's alternative); (b) a side stream with the
  post joined where it stands, so the gather overlaps only the host's own work; (c) a side stream and the Engram layer's
  post moved after that block's row read and upload; (d) nothing - record that prefill gathers have no slack.
- Decision: (c). The gather runs on its own stream, and `engramApply` keeps its post first unless a gather is in flight;
  then the post waits for the rows to be read and uploaded.
- Why: the layer loop's only consumer of gm is the *first* step of the next layer (the pending post in engramApply, tap,
  attnMixes or head), so at the 38 non-Engram layers there is nothing to defer behind - (a) would hold two gathers and
  still have no work to overlap them with, and (b) buys nothing outside the same two layers. At the two Engram layers the
  host reads and uploads the rows before any kernel needs h, so (c) is the one real window (≈70 ms of host read a layer
  at 2,048 rows against a measured 8.31 ms an exchange), and hc_post writes `h` while the read and upload write `eb`:
  independent, so the reorder is bit-identical. (d) was rejected because the window, though narrow, costs 30 lines and is
  measurable; the switch says so and the row will close it if the chunk wall is flat.

## 0184 finer indexer buckets: granularity 1024, ceiling the power-of-two bucket, the fused select given up (2026-11-16)

- Scope: minor (one bucket function + one switch; off by default, bits identical).
- Considered: (a) exact per-position sizing (nb = deepest/ratio, no rounding); (b) a multiple of 1024; (c) a multiple of
  the candidate pool's block; (d) keep powers of two and close the finding.
- Decision: (b), with the bucket clamped at `bucketFor`'s own power of two and `rd.bucket` taken from it.
- Why: (a) makes every new position a new bucket, i.e. a new captured graph a round and a new capture on the hot path -
  the plan's explicit cost. (c) is finer than the indexer's own 64-key tiles need (1024 is already a whole number of them
  and of the pool's blocks, which is what `candFast` and `blockMax` need). The clamp at the power of two is what keeps
  every buffer, the graph cache's sizing and the candidate pool's bounds valid, and 1024 keeps `idx_all`'s nb where it was.
  The fused `_topk_sel` is lost for a non-power-of-two `nb` - two launches instead of one - which is the price the plan
  names and the reason the row pairs this with `0182`'s fast select. (d) leaves the one measured step (`a_idx` at ~65,000
  against ~66,000 tokens) unexplored for a change of one function.

## 0185 bf16 partials: the GEMM writes bf16 rather than converting after it, and only the chunk path takes it (2026-11-16)

- Scope: minor (one switch, one gather helper, one widen; LOSSY, off by default).
- Considered: (a) keep `pa` fp32 and convert it to bf16 before the gather (one extra kernel a layer); (b) have the wo_b
  GEMM write bf16 directly (the same RNE rounding, no convert); (c) apply it to the decode rounds too, where the gather
  is the RDMA ring; (d) apply it to the 2D split as well.
- Decision: (b), for the prompt chunk only, with the widen at `ffnMixes` (where the kit reads `ga`).
- Why: (a) pays a launch for a rounding the GEMM already does when asked for bf16 output - the whole point of a LOSSY
  switch is to buy payload with bits, not to buy it with a launch. (c) is unreachable without changing the ring's
  transport (fp32 only) and the decode gather is 4 rows wide - the plan's finding is the prompt's 41.9 MB a rank. (d) is
  not ours: the 2D split's partials are the pair's own columns written by `prompt2d`, and its quarters post on the main
  stream. `pb`/`gb` are allocated only when the switch arms, so the off path's memory and launches are untouched, and the
  recorder lane builds its own Engine and never arms the flag - its fixtures stay exact whatever the environment says.

## 0186 the argmax exchange: what the reduce owes the pick kernel, and where the round decides (2026-11-16)

- Scope: medium (two kernels, a per-round flag, a graph-key bit; off by default, bits identical).
- Considered: (a) each rank's argmax of its half, exchanged and reduced with strict `>` in rank order, the pair's index
  offset by the rank's width, rank 0's row[0] rule carried as a sentinel; (b) the same but reduced only on rank 0 and
  sent back; (c) exchange (value, index) but keep writing `logits` for the non-pick readers; (d) leave the head's gather
  and close the finding with the measured `head` phase.
- Decision: (a), armed per round by lanes.zig before the forward (`Round.map`), with the switch `TF_DS_GREEDY_ARGMAX`
  and the rule in `sampling.argmaxMapWorks`; the reduce writes the picks where `argmaxRows` wrote them, so the device
  accept chain and the host read are untouched, and the round's graph key gains the choice.
- Why: (b) adds a second collective and a host hop for nothing - every rank needs the pick anyway (rank 1 verifies its
  own window), and an all-gather of R * 8 bytes is the cheapest symmetric answer. (c) keeps the very payload the item
  removes. (d) leaves the plan's measured ceiling unexplored. The rule that had to be got exactly right is the *index
  space*: `argmaxRows` answers in the whole row's columns (rank 0's half first), so a rank's local column must be offset
  by `rank * n` - the host simulation (`runs/bench-prep2/argmax-map/`, 58 cases) caught the first version missing that,
  which is why the patch's ASCII claim of "identical" rests on that run rather than on the reading. The NaN rule follows
  the same logic: only rank 0's `row[0]` is the whole row's first value, so the pair carries it as an index sentinel and
  the reduce maps it to pick 0.

## 0164/0187 the expert grid bound: already built for decode, closed for prefill (2026-11-16)

- Scope: none - a closed finding, no patch, the number `0187` stays unused.
- Considered: (a) build `TF_DS_MOE_GRID_BOUND` as planned; (b) reuse `TF_DS_T15_PLACE_BOUND` and extend it to prompt
  chunks; (c) close the finding with what already exists.
- Decision: (c) for prefill, and (a) is void because the decode half exists: `expert_prune.placeBound` under an applied
  `TF_DS_T13_EXPERT_PRUNE` cap already sizes the decode window's grids and work lists (`round.zig`'s `moe`, landed in
  `zig/patches/0053`).
- Why: (b) has no bound to pass. The pruner runs on decode windows only, so a prompt chunk has no applied cap, and the
  only other candidate - the distinct experts the picks produce - lives on the device while the kit's `rot_in` and
  `gateup_epilogue` derive `(row, slot)` from `blockIdx.x`; a grid smaller than `rows * slots` needs a place list,
  which is the kit's half (the plan says so: "the empty-place cost is kit-side"). Applying the cap to prefill to create
  a bound would be a lossy behaviour change of its own (dropped expert slots on prompt rows), not a grid tweak, and it
  belongs to whoever measures the cap on prefill. The [November 17 retest](runs/review-round/RESULTS.md#applied-cap-t15) subsequently measured T15 at eight fixed decode rows with an applied cap.

## 0166/0189 the Markov chain stop: a test, because the cheap variant already ships (2026-11-16)

- Scope: none but a host test; the number `0189` carries the evidence, not a mechanism.
- Considered: (a) add `TF_DS_MARKOV_TRIM` as a toggle around a granted+slack cap; (b) build Baseten's in-graph confidence
  stop; (c) pin the shipped behaviour with a test and close the item.
- Decision: (c). (a) would wrap a behaviour that is already the served default - `Lanes.facts` sets
  `draft_probabilities = drafting`, so `depth.Rule.budgets` takes its node-probability branch, which asks
  `@min(head depth, granted + draft_slack)` a stream, and the pass's `steps` is the largest of those. (b) needs
  conditional nodes agreed by both ranks (the plan's own caveat) and is not a repo-only change.
- Why: a switch that toggles the shipped path is either a no-op or a behaviour change nobody asked to measure. The test
  is the durable part: it fails if the cap (or the "never under 1" floor) ever regresses, which is what the plan's claim
  needs in order to stay true. It also corrects the odds list: 0166 was never unbuilt.

## 0190 the Snap trim: the layout carries the trim, so no format version bump (2026-11-16)

- Scope: medium (a layer-shape flag, the snap's ring range, a refusal; off by default, bits identical).
- Considered: (a) trim `snapCopy` only, leaving the layout as it is; (b) carry the trim in the layout (`LayerShape.ring`)
  and let planes, sizes and digest follow; (c) ship both (b) and a `format.version` bump, as the scout suggested.
- Decision: (b), plus the refusal without `TF_DS_REPLAY_FLOOR=1`.
- Why: (a) desynchronises `snapBytes` from the layout the spill tier packs against (`SpillLayoutMismatch`), and leaves a
  stored point of the other layout indistinguishable. (c) would invalidate every store on every deployment for a switch
  that ships off: the layout digest is already part of the rank fingerprint and of the store's compat digest
  (`spill_tier` -> `format.compatDigest(storeCommon(rank), ...)`), so an untrimmed store is foreign to a trimmed build and
  the pair refuses a mixed start - which is exactly what a version bump would buy, without touching the deployments that
  never enable it. The refusal stands because `TF_DS_REPLAY_FLOOR=0` reads ring rows below the floor, where a trimmed
  Snap holds nothing; the deployment's default is `REPLAY_FLOOR=1`.
## 0193/0194/audit the kit's sources exist — the gate moves to byte-identity (2026-11-19)

- Scope: major (changes the premise behind the "do not touch the kit" rule; no code change).
- Considered: (a) leave `continue.md`'s "no source, no safe replication" as-is; (b) keep the do-not-touch rule but
  correct its reason and record the provenance audit; (c) start replicating kit cubins now, since source exists.
- Decision: (b). Full mapping in [docs/kernel-kit-provenance.md](docs/kernel-kit-provenance.md). The kit
  (`bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10` @ `abe07c7`, `KIT_REF`) is public and manifest-pinned; its Triton
  AOT cubins' source is `bertholomus/TensorFold` @ `deepseek-v41-tp2` (`kernels.py`); its EXL3 cubins come from that
  branch's `cuda/exl3/` (built in the nvcr container); PyTorch's sm_120 cubin and NCCL are binaries; the RDMA
  all-gathers and all host-side Engram/row code are repo-owned. `../TensorFold` (ashhart fork) holds a parallel native
  implementation, not the served build's source.
- Why: "no source" was false and would have kept future agents from even considering in-repo equivalents (the
  0193/0194 fused route already replicates `_rowmm_parts`/`_route_parts` arithmetic on CPU). What actually blocks
  kit replacement is byte-identity: a rebuild differs (nvcc/Triton build-to-build), and only the exact suite on the
  Spark pair adjudicates bits. (c) is premature: every replication candidate still owes the same gate as 0193/0194,
  and the deployed recipe stays pinned. The do-not-touch rule is retained, restated as "a new kit artifact owes the
  Spark exact gate", not "source does not exist".
## 0195/0196 the indexer pool fusion and the drafter route: built, reviewed, gated on the Spark suite (2026-11-19)

- Scope: major (two new off-by-default candidates in the series; not deployed; no image changes).
- Considered: (a) leave `continue.md` item 4 as not-started; (b) build (a)+(b) as off-by-default patches behind
  switches, verify on CPU, export, independent review, carry gated on the Spark exact suite; (c) also build
  (c)/(d); (d) adopt immediately in an image.
- Decision: (b). A = **0195** (`TF_DS_IDX_FUSED`, off by default, refusal on bad values): one launch for the candidate
  pool's `blockMax -> topk_indices -> poolPick` (shared `tf_ds_topk_select_row` extracted verbatim; `applyPool` and
  `candFast` left unfused, already single launches). B = **0196** (`TF_DS_FUSED_ROUTE` reused, no new switch): the
  drafter's gate + routing through the same fused launch, `do_prune=0, dead=0`; `two = e.two != null` mirrors the
  main model's 2D refusal (flagged, two-token change if preferred otherwise). Engine branch `fused-route`
  (b1033b4 + b31f3a6 + 25eeab5); patches in `zig/patches/0195-dsv41-tf-ds-idx-fused.patch` and
  `0196-dsv41-tf-ds-fused-route-draft.patch`; `second-pass-int` untouched; nothing pushed or deployed.
- Why: (c) is gated on the PDL gap profiler and would double the surface ahead of evidence; (d) puts unproven bits in
  production. Verification: `zig build test` at base and after both commits shows exactly the two known pre-existing
  failures (`prompt_cache.test.0178`, `round.hotPick`); the g++ emulation harness of the exact kernel source shows
  1070 rows / 0 mismatches vs the three unfused kernels and an independent sort-by-key reference; the patches
  apply clean (0195 onto b1033b4, 0196 onto b1033b4+0195) and the patched tree is file-identical to the branch tip;
  the independent reviewer found no blockers and could not execute builds (source review). GB10 bit-identity is
  unproven everywhere — 0195's selection is integer-exact but shares the gate; 0196 inherits 0193/0194's open
  Triton/libdevice question. Adoption order: 0193/0194 exact suite first, then an A-only arm, then B; none ahead of
  the gate; both ship behind their unset switches until their own arms pass.
## 0197/0198 the compressor-norm and rope-store fusions: built, one real defect caught and fixed (2026-11-19)

- Scope: major (two new off-by-default candidates; not deployed; no image changes).
- Considered: (a) leave (c)/(d) of continue.md item 4 unstarted; (b) build both as off-by-default patches with the same
  gate discipline as 0193-0196 (switch parse that refuses bad values, fuseOk bounds, exact unfused fallback, CPU
  emulation of the exact kernel source against the launches replaced and an independent reference); (c) fold Engram's
  `add2Bf16`+`_engram_gate` too; (d) adopt ahead of the 0193/0194 gate.
- Decision: (b), and Engram parked (c). C = **0197** `TF_DS_FUSED_COMPRESS2_NORM` (ratio-2 compressor chain + trailing
  RMSNorm in one launch, decode and prompt; supersedes 0127 when set). D = **0198** `TF_DS_FUSED_ROPE_STORE` (decode
  indexer key/latent/query and prompt key/latent rope-store chains in one launch each). Engine `fused-route`:
  `20001bc`, `44e725a`. Patches exported and verified to reproduce the tree from 25eeab5 (and 0197 builds and passes the
  suite alone: 788/790, the two known failures).
- Why: the launch counts are small (C ~3, D ~20 a round, 0.05-0.2%) so both are gated on the PDL gap profiler; (c)
  buys two launches a pass for a `tl.sum`+`tl.exp` reduction-order risk, a bad trade at this size. The per-call-site
  bit rule was the deciding constraint: the decode path launches Triton kernels and the prompt path torch-parity Exact
  kernels, so each fused kernel replicates its own call site's kernel rather than a shared one.
- Verification: C's CPU harness 8100 rows / 0 fused-vs-unfused failures; D's 28,161 rows / 0, plus 923k grid checks and
  independent references. **D's harness caught a real defect**: the E4M3 scale encoder had transcribed the Triton
  saturation arm as `fb >= (1087 << 23)` (thought dead) instead of `fb >= (1087 << 20)`, so every f >= 496 encoded as
  0x80 (-0.0) where the kit saturates to 0x7E (448) — 22,161 of 923,076 blocks. Fused and unfused agreed with each
  other because both used the same helper; only the independent contract check caught it. Fixed in the kernel and the
  Zig reference, host test now pins amax 2879/2880/1e4. The independent review then found two more, both fixed:
  `tf_ds_torch_rmsnorm_row`'s shared reduction array was 128 lanes where rmsLanes is up to 512 (now 512), and the launch
  decisions ignored `fuseOk`, so an unsupported shape hard-errored instead of falling back to the unfused path (all
  call sites now gate on `fuseOk`, the 0195/0196 discipline). `zig build test` 622/624 with only the two known failures.
- Open: GB10 bit-identity unproven (as 0193-0196); the honest risk is `_rmsnorm`'s reconstructed `tl.sum` order and the
  Triton fma contraction in `_rope_heads`. Adoption order stays 0193/0194 -> A -> B -> C -> D, switches unset until
  their own arm passes.
## 0200-0212 the 0195-0199 follow-up review, thirteen items built ahead of the gate (2026-11-21)

- Scope: major (thirteen patches changing four off-by-default fused paths; no new switch but `TF_DS_CHECK_INVALID`; not
  deployed, no image change).
- Considered: which of `todo.md`'s nineteen 0195+ items to build before the benchmark: (a) the whole list including the
  five that their own text says to inspect SASS or re-sweep first; (b) only the four "operational overhead" items
  (1, 2, 4, 18); (c) everything decidable from source without a profiler — items 1-4, 8-11, 13, 15 and 17-19; (d) wait
  for the GB10 measurements before any of it. Also: one patch an item, one patch a family, or one squash (chosen: one an
  item, so a losing item can be dropped without editing a shared patch), and how to handle item 15's dead `ch.invalid`
  flag — a host reader, an always-on read, or an in-kernel trap (chosen: an opt-in switch with a logged error, no
  production cost, no synchronisation when off).
- Decision: (c). Patches `0200`-`0212`, one an item, each riding the switch its parent patch already has: 0200 dead
  fused-path scratch stores (item 1), 0201 the prompt query on 0198's kernel (2), 0202 the `L == 32` norm barrier (4),
  0203 `TF_DS_CHECK_INVALID` plus the host slot-distinctness check (15), 0204 the recorded and asserted warp-alignment
  invariant (17), 0205 `keyFuseOk` (18), 0206 the 0195 key/tile-maxima fold that drops `_score_keys` (3), 0207 the
  single-exponential finite pair softmax (8), 0208 the shared high-word cache (9), 0209 the packed-column key store with
  rotation folded in (10), 0210 the norm storing its held pair results (11), 0211 the prune prefix early-out (13), 0212
  the flattened/wider prompt packing (19). Items 5-7 (explicitly profile-gated), 12, 14 and 16 (SASS or a GB10 re-sweep
  as their gate) stay unbuilt in `todo.md`.
- Why: (c) is the largest set whose correctness can be argued and tested from the source plus host tests; (a) spends the
  benchmark's measurement budget on changes already prescribed to be measured first, and (d) leaves the free deletions
  and the two silent-invariant guards undone for no gain. 0200 alone deletes 229 KiB of stores at 0195's seven-row
  shape; 0203/0204 turn two "true because nothing checks it" assumptions into checked ones. Two items are arithmetic
  arguments rather than refactors and are flagged as such (0207: `expf(0) == 1` exactly with the full path's operand
  order; 0211: a nondecreasing rounded prefix), so their parity must be checked against the unfused path.
- Verification: a clean `TF_REF` (fbbc971) + `zig/patches` 0001-0212 applies with every `git apply --check` green (146
  patches) and reproduces the work tree byte for byte (`diff -r`); `zig build test` passes every suite (73/23/465 test
  binaries plus the focused `idx_fused`/`rope_store`/`compress2_norm`/`expert_prune`/`exact`/`round`/`ops` cases) in the
  work tree and in the reproduced one; `clang++ -fsyntax-only` over `dsv41_torch.cu`, `dsv41_ops.cu` and
  `dsv41_shard.cu` with stub CUDA headers is clean. New host tests pin 0206's key bytes against `scoreKey`, 0208's
  rebuilt keys and validity bytes, 0209's codes/scales against the separate-pass pipeline at four shapes, 0207's two
  softmax forms, 0211's early-out against the full rule, and 0205/0204/0212's predicates. **Nothing has run on a GB10:
  the kernels' bit identity, the SASS claims 0202/0210/0212 owe, and every timing question are open** and are the
  benchmark's first work.
- Open: the batch's one gate (39 exact cases, long greedy hashes, PDL both ways, fused-vs-unfused bytes for 0206/0208-0210,
  and the unfused comparison for 0207/0211) is recorded in `todo.md` and `continue.md`. `0211` adds six predicate tests
  to a six-slot rule to save two adds and two compares: it may well be a net loss at the served shape and should be
  judged by the measurement, not kept by default.

## 0200-0212 bench: how it was compared, and the key-encoding mismatch it found (2026-11-21)

- Scope: major (the thirteen patches' first GPU gate: standalone parity, timings, SASS, and eight paired arms).
- Considered: (a) build the full recipe series 0001-0212 for the bench image; (b) layer 0200-0212 over the prune-six tree
  that the previous two campaigns measured; (c) skip the bench image and change switches on production's pinned image.
  Also: which exact-suite reference to judge the fused arms against - (a) production, (b) the previous fused bundle, (c)
  both; and whether to extend the audit harness or start a new one.
- Decision: (b) layered image over `bench-prune-six-20261120` (production source + 0195/0197/0198/0199 + the 13), shipped
  to both ranks as `bench-followup-0212`; (c) both references - production for the inert/single-switch arms, the previous
  fused bundle for the bundle; the existing `probe.py`/`decode.py` harness extended rather than replaced
  (`runs/fused-followup-20261121/probe.py`, `build2.sh`, `sass.py`).
- Why: the full series also carries 0182-0190 (measured and not adopted), so a full-series image would confound the 13
  with unrelated patches; the prune-six base makes every arm differ from the previous bundle by exactly these thirteen.
  Production is the right reference only where the switch is off (the patches must be inert); the previous bundle is the
  right reference where it is on, because 0197 alone is known to move a 1,024-token greedy reply, so production equality
  is unachievable for `TF_DS_FUSED_COMPRESS2_NORM` arms by construction.
- Finding: the standalone gate passes (byte parity on every live output, dead buffers unwritten, no spills) but
  `TF_DS_IDX_FUSED=1` changes 4 of 39 exact cases (`len_20000`/`30000`/`50000`/`64000`; `len_100000` and everything
  shorter match). The previous bundle never produced keys at those widths, so one of the thirteen changed what the pruned
  selector sees. First suspect: the fold's key encoding - `idx_fused.zig:scoreKey` XORs the key's top bit where the kit's
  `_score_keys` does not (`0x80000000ffffffff` vs `0x00000000ffffffff` on the same cell), which would reverse
  negative-vs-positive scores under a signed comparison in the kit's `topk_select_pruned`. It is unproven: it does not
  explain `len_100000`, so the width window needs instrumenting before either side is edited.
- Harness limits worth keeping: the kit's `_score_keys` faults with `CUDA_ERROR_MISALIGNED_ADDRESS` unless `width % 4 ==
  0`, so it cannot be the byte reference for odd served widths; both trees' decode compressor must be launched with `d`
  threads (at 128 both disagree with the kit on ~200 bytes a row, which is a harness error, not a tree difference).
- Open: nothing deployed; production restored and validated. `all`/`allpdl` inherit the `idx` difference; PDL on the
  fused bundle is unmeasured. The kit `_score_keys` launch the fold replaces is still unpriced.

## Fused follow-up collapse: six folded into 0195, seven kept separate (2026-11-21)

- Considered: (a) leave all thirteen as separate patches; (b) fold all thirteen into 0195; (c) fold only the six that
  measured faster (0200, 0201, 0202, 0209, 0210, 0212) and keep the seven that measured worse or remain unproven
  separate (0203, 0204, 0205, 0206, 0207, 0208, 0211); (d) fold by file rather than by measurement.
- Decision: (c). `0195-dsv41-tf-ds-fused-paths.patch` now carries the four switches plus the six wins; the seven stay
  as separate patches in the same order. 137 patches total (`zig/patches`), down from 146.
- Why: the user's rule was "collapse the faster patches into the base patches, leave the patches that had negative
  performance." The winners are 0200 (dead stores), 0201 (prompt query), 0202 (barrier), 0209 (key store rotation),
  0210 (held pair results) and 0212 (wider packing). 0207 measured 13-17% slower on the served decode shape, so it is
  kept despite being a speed win at the fastest shape; 0206 costs the key writes it adds and changed four long-prompt
  replies in the gate; 0208 caches 0206's words; 0203/0204/0205/0211 are diagnostics and gates.
- Build: the collapse was done in a scratch tree (`~/Workspace/fold-scratch`) by un-applying 0195-0212, re-applying
  the four bases, then applying only the winners as one combined patch, then replaying the seven kept patches in order.
  Conflicts were resolved by keeping both sides; the final tree is byte-identical to the pre-collapse tip
  (`git diff --stat ffdce65 HEAD` empty).
- Rejected (d): folding by file would split one measurement's change across two patches, which loses the "this patch
  is the one that costs X" property the gate depends on.
- Open: the series now needs a fresh `git apply --check` on a clean 0194 tree (done: all 137 green) and the same
  pre-collapse gate re-run on the collapsed image, which is a follow-up, not part of this change.

## Fused PDL entry: split only where a fixed weight can be read first (2026-11-21)

- Scope: minor, `fused-next` commits `f013282` (0250) and `59e71a8` (0251); evidence in
  `runs/fused-next-20261121/wp4/RESULTS.md`.
- Considered: (a) leave `pdl_enter` as wait-then-launch in every kernel; (b) move `launch_dependents` before the wait
  in every kernel; (c) (b) plus an L2 `prefetch` of the fixed norm weights before the wait; (d) (b) plus holding the
  norm weight in a register across the body, in the decode compressor and the indexer key only, with the compressor's
  non-PDL path keeping its old body; (e) split the entry in every kernel, including the ones with nothing independent
  to read.
- Decision: (d) for the two kernels that read a fixed model weight (the compressor's `comp_norm`, the indexer key's
  `idx_k_norm`); (a) for everything else. Every other kernel keeps wait-then-launch.
- Why: (b) alone measured slower on the decode chain at every shape above nb=8 (+0.6 µs at R=48 nb=8192: a successor
  scheduled with nothing to do before its wait only takes SM slots from the running kernel); (c) recovered part of
  that but not the load latency the body pays at its end; (d) is 0.17-0.35 µs faster per layer at every served decode
  shape with the non-PDL path unchanged, and the delayed-producer canaries show every dependent read still after the
  wait; (e) is the (b) penalty again for the rope/idx/prune kernels.
- Also decided: `TF_DS_FUSED_TRACE` (opt-in, off) rather than always-on counters, and the unfused fallbacks stay the
  engine's own launches (the harness now checks them rather than adding a fused variant for them).
- Open: the prompt half's `compress2_norm` has the same fixed weight and could take the split, but the prompt path is
  at the bandwidth roofline, so it was left alone.

## Pre-benchmark follow-ups closed without engine restart: B2, dec8, 0254 ties, harness guards and 0256–0258 (2026-11-21)

- Scope: minor-to-major verification and harness/host-trace hardening (`tools/fused_bench/`, `runs/pre-bench-20261121/`,
  and `tfref-fused-next` commits `e5110a0`/0256, `9ece091`/0257, `84806bb`/0258).
- Considered and decided:
  1. **Review Blocker B2 (`idx-small R=1 nb=8192`, `compress-decode R=32 D=512`) — fix `bench.py` stream synchronization, leave kernels unchanged.** Both kernels passed 12 `compute-sanitizer` (`initcheck`/`racecheck`/`memcheck`) suites with 0 errors and 14,400 stream-ordered parity runs with 0 byte diffs. The WP2 mismatch came from `bench.py` pairing `cuStreamCreate(..., CU_STREAM_NON_BLOCKING)` with legacy stream-0 `cuMemcpyHtoD_v2` (`<=64 KiB` pageable staging returns before DMA completes), reproduced on GB10 (`2/220` failures) and eliminated (`0/220`) by stream-ordered `cuMemcpyHtoDAsync_v2` + `cuStreamSynchronize`. [Evidence](runs/pre-bench-20261121/b2-investigation/RESULTS.md)
  2. **`dec8` `-0.59%`/`-0.69%` batch tok/s dip — no kernel rollback; use staggered admission in future 8-stream replays.** Exact decomposition of `s01..s05` showed true active post-prefill decode is `+0.003%` (Pair A), `-0.51%` (Pair B) and `-0.30%` (PDL); the median-wave batch dip came from random HTTP admission order under `TF_DS_ROUND_ROWS=16` (`+4`/`+5` rounds on straggler `s=7` = `83.6%`/`67.8%` of delta) plus `min(t_first)` excluding longer `ord=0` prefills in `base` while `next` prefilled all 8 prompts faster. [Evidence](runs/pre-bench-20261121/dec8-audit/RESULTS.md)
  3. **`0254` tied-score cost (`138.8 µs` vs `100.7 µs` at `128 × 32,768`) — close with no kernel edit.** `sm_121` already warp-coalesces uniform shared `atomicAdd`; `~82–86%` of the delta is `+2.01` extra row scans (`+32.1 MiB`), and cutoff-straddling `-∞` rows account for `<0.03%` of served wall time (`applyPool` L24–36 retains `16,384 >> 512` finite columns and finishes in `2.89` passes). [Evidence](runs/pre-bench-20261121/topk-and-patches/RESULTS.md)
  4. **Harness timing validity & `0215` threshold — keep `small_nbk = 6,144` until an idle `/health` window.** Fixed all four 3-chain/4-name `time_graphs` callers (`100` historical rows audited) and enforced strict `connections == 0` + valid unchanged token-counter guards (`9/9` unit tests). `0215` passed `36/36` GB10 byte-parity cases across `nbk = 2,049..16,384`; live serving prevented clean timings (`timing_valid: false` recorded rather than dirty medians). [Evidence](runs/pre-bench-20261121/crossover-0215/RESULTS.md)
  5. **Host dispatch & trace (`0256–0258`) — land on `fused-next`, defer recipe rebase to image-build window.** Made `fused_trace` thread-safe and single-sourced with `0254`/`0255` coverage (`TF_DS_TOPK_FAST` precedence preserved), extracted `planKvSource2` in `prompt.zig` with 4-chunk GB10 byte-parity (`0` diffs), and mapped the 17-commit `0213–0258` vs `0182–0190zc` rebase onto `zig/patches/` (`c3b3859`). [Evidence](runs/pre-bench-20261121/p0231-dispatch/RESULTS.md)

## Suite gate semantics for fused arms: flags, not arm names (2026-11-21)

- Scope: minor, `runs/suite-fused-next2-20261121/gate.py` (new suite; the old folder's gate stays a historical record).
- Considered: (a) reuse `suite-fused-next-20261121/gate.py` unchanged; (b) patch it in place; (c) fresh suite folder with a rewritten gate + CPU tests. Chose (c).
- Decision: the lead-reader banner is checked against each block's `TF_DS_ENGRAM_LEAD` flag (production `.env` default `1`), not its arm name; the "drafting must differ" rule applies only to arms whose flags change drafting/pruning switches (`TF_DS_DRAFT*`, `TF_DS_T26*`, `TF_DS_SPEC*`, `TF_DS_T13_EXPERT_CAP/PRUNE`, `TF_DS_FIX_K`, `TF_DS_T25*`); non-drafting arms REQUIRE rounds/accepted-drafts equality with the reference. The suite path derives from the gate file (`$GATE_SUITE` override), and bailed/exception cell shapes report a named problem instead of raising.
- Why: the old gate hardcoded `S = "runs/suite-fused-20261121"`, expected the lead banner only on `lead` arms while every arm runs production's `.env` (`TF_DS_ENGRAM_LEAD=1`), and treated any arm outside `{base, lead}` as a drafting-change arm — so a bit-exact fused arm reproducing base's rounds would fail with `switch NOT APPLIED`, which is exactly the evidence the pair gate exists to collect. `9/9` CPU tests (`test_gate.py`) pin each branch; an independent reviewer marked the suite `OK with notes` (both P2s fixed). [Suite README](runs/suite-fused-next2-20261121/README.md)


## 2026-11-22 0254–0258: measured, bit-identical, speed-neutral; adoption is the operator's call

- Scope: major (engine series `0213–0258`, the whole fused-next round) but the measurement changes no code.
- Considered: (a) adopt `84806bb` on the gate's pass criteria; (b) stay on `d5ecba0` and drop `0254–0258`;
  (c) adopt only `0254`/`0255`. Ran the ABBA pair gate (`runs/suite-fused-next2-20261121/`, 02:34–04:31 UTC).
- Decision: **recorded as "measured, not yet decided"** — the gate passed 5/5 blocks (exact 39/39, probe 11/11
  hashes with identical rounds and accepted drafts, `fixed_replay` inputs and token hash equal, `dec8_staggered`
  3/3 waves equal, validation rc 0), and speed is inside the window's own resolution floor: next-vs-base pure
  decode was `-1.35%` (pass 1) and `+1.71%` (pass 2) against a `-2.23%` base-vs-base control. So (a) is defensible
  for `0254`/`0255`'s standalone wins (−39…−60%, −15…−23%) with no serving cost, and (b) is defensible because
  nothing is gained in serving. This entry exists so the *result* is not lost; the pick needs an operator.
- Also decided here: fix rather than work around the two `preflight.sh` path bugs (`cd` one level too high, rsync
  target resolving to `~/runs/...`) and re-record s01's `dec8_staggered` cell (its script was never on the head)
  instead of gating around it. Not done: the guarded `0215` capture, which needs a *live idle* `/health` — a
  stopped production is reported by that harness as "no server", so the suite window cannot capture it even
  though the GPU is free. `idx_fused.small_nbk` stays 6,144. Not done either: `fixed_replay`'s prompt still
  answers in one token, so the deep decode buckets stayed unmeasured (todo.md).
- Why: a pair gate whose own harness is bypassed (gating around a missing cell) would not prove what it claims;
  and a "neutral" verdict is only meaningful when the control arm's spread is printed beside it.
- Open: the deep-decode `fixed_replay` prompt, the 0215 timing window, and the Engram lead-reader soak owed from
  2026-11-19. [Results](runs/suite-fused-next2-20261121/RESULTS.md)

## 2026-11-22 0254–0258 adopted; the round rebased onto `zig/patches/` as 19 patches

- Scope: major (engine series), following the gate in `runs/suite-fused-next2-20261121/RESULTS.md`.
- Considered: (a) adopt `84806bb` and rebase (operator's pick); (b) stay on `d5ecba0`; (c) adopt only `0254`/`0255`;
  (d) re-measure with a decode-bearing `fixed_replay` first. Chose (a).
- Decision: **adopt**. The pair gate passed 5/5 blocks bit-identically and speed is inside the window's control
  spread (`-1.35%`/`+1.71%` next-vs-base against a `-2.23%` base-vs-base control), so the series costs nothing in
  serving; `0254`/`0255` keep their `-39…-60%` / `-15…-23%` standalone wins, and `0256`/`0258` make the trace
  thread-safe. (d) was the honest alternative and is still in `todo.md`: the deep decode buckets were never
  measured because `fixed_replay`'s prompt answers in one token.
- Rebase: 19 patches (`0213–0217`, `0230–0231`, `0240–0242`, `0250–0258`) exported from
  `~/Workspace/tfref-fnext-rebase` (`c3b3859` + cherry-picked `fused-next`) by `scripts/export-fnext-series.sh`
  (`just export-fnext`), numbered from the commits so the round's gaps survive. Two hand resolutions: `0217` keeps
  `0182`'s `topk_fast` const and takes the `const o: Ops` dynamic-shared opt-in; `0256` takes the
  `topkIndicesArm` dispatch, keeps `.topk_indices_fast`, and keeps `0182`'s precedence
  (`if (o.topk_fast) o.topk_indices_fast else o.topk_indices`) alongside the trace note. `0254`/`0255` replace the
  hand-ported pair after an added/removed-line comparison showed them content-equal to their commits.
- Why not `scripts/port-patches.sh`: it applies every hand-written patch before every exported one, and this tree
  interleaves them (exported 0129–0171, hand-written 0182–0211, then this round), so it cannot reach the base.
- Verified: the chain applies onto a clean `TF_REF` in filename order and reproduces the rebased tip; `zig build
  test` passes on the rebased tree (one run printed the known rank-1 spill-scan flake); a `TF_REF + patches`
  checkout type-checks with `zig build install native -Dfatbins=<dummy dir>`. **Not done:** an image build from the
  recipe and its deploy gate (39-case exact, `validate.py --compare`, hardware smoke) — production stays on
  `bench-prune-six-20261120` until that passes.

## 2026-11-22: production onto the `fused-next` image (`bench-fused-next-84806bb`)

- Decision (operator): adopt `84806bb` into the production baseline now, by image, without waiting for the recipe
  build. "Wins are a win and they will keep adding up": exact plus serving-neutral plus faster standalone is enough.
- Done 05:58–06:01 UTC with `runs/deploy-fnext-20261122/deploy.sh`. Only `IMAGE=` changed in the head's `.env`
  (`bench-prune-six-20261120` → `bench-fused-next-84806bb`, `sha256:2408857…`). The switches are unchanged and
  `PREPARE=0`. Both ranks were proven on the image with production's switches. Native binary sha256 `4fd57ef1…`,
  lanes `0d60da68…`. Smoke ok. `validate.py --compare`: all passed, 8/8 saved replies, concurrent == solo 8/8.
  Undo: `cp ~/dsv41-env-backups/.env.bak-20261122-pre-fnext .env && ./start.sh restart`.
- Evidence chain: production → `d5ecba0` (`runs/suite-fused-next-20261121`, 39/39, neutral) → `84806bb`
  (`runs/suite-fused-next2-20261121`, 5/5, neutral).
- The recipe image (the rebased `zig/patches/`) is still unbuilt and has never run on hardware. Building it and gating
  it against this image is open. Do not `just sync-head` until that is done.
- From now on, `bench-fused-next-84806bb` is the control arm for every A/B.

## 2026-11-22: 0259 `TF_DS_KV_FETCH_REUSE` adopted (`bench-fused-next-69335b8`)

- What it does: fetch decode packs and exchanges a run's picked KV rows once. A layer reading the same KV source
  through the same selection with the same kk attends the rows already gathered. The checkpoint's 38 compressed layers
  form eight runs, so 30 of 38 exchanges are skipped. Source: branch `kv-fetch-reuse` `69335b8` in
  `~/Workspace/tfref-kvreuse` (worktree of `tfref-fused-next`, on `84806bb`).
- Sizing (`runs/kvreuse-size-20261122`): `sh_fetch` + `sh_rows` were 3.6–4.2% of a decode round, projecting 3.0–3.5%.
- Gate (`runs/kvreuse-ab-20261122`): ABBA on one image. Exact 39/39 identical in every arm, reuse engaged on both ranks,
  graphs-on rounds 0.5–2.5% shorter (mean −1.7%). About half the projection was realized; the rest is likely
  cross-rank wait that moves to the next collective.
- Decision (operator's rule: exact plus faster gets adopted): deployed 07:07–07:10 UTC by
  `runs/deploy-kvreuse-20261122/deploy.sh`. `IMAGE=bench-fused-next-69335b8` and `TF_DS_KV_FETCH_REUSE=1` were added to
  the head's `.env`. `validate.py --compare`: all passed, 8/8 saved replies. Undo:
  `cp ~/dsv41-env-backups/.env.bak-20261122-pre-kvreuse .env && ./start.sh restart`.
- Open: export 0259 onto the rebased recipe tree (`zig/patches/`); the recipe image is still unbuilt.

## 2026-11-22: production onto the recipe lineage, with 0260 `TF_DS_CONF_PACK` and 0261 `TF_DS_TOPK_MERGE`

- Image `bench-fused-next-086e8cb`: `TF_REF + zig/patches 0001–0261`. The 0260/0261 one-time engagement banners are
  folded into those patches (`recipe-0261`, tree `bfc02e5` = the image's). **Production now equals the recipe's patch
  chain**, which `scripts/export-fnext-series.sh` verifies.
- Evidence:
  - `runs/overnight-20261122/recipe-gate`: the rebased tree is exact on hardware.
  - `k260-gate`: 0260/0261 exact on the greedy and sampled panels, neutral.
  - `lineage`: 3 ABBA pairs against the previous production; exact and sampled identical in every arm; banners on
    both ranks; +0.30% within drift; the gate's earlier 128K × 4 +3.2% did not reproduce.
- Decision: the operator's adopt rule (exact plus neutral-or-faster). Deployed 10:11–10:13 UTC with
  `runs/deploy-recipe-20261122/deploy.sh` (IMAGE swap, plus `TF_DS_CONF_PACK=1` and `TF_DS_TOPK_MERGE=1`).
  `validate.py --compare`: all passed, 8/8. Undo: `cp ~/dsv41-env-backups/.env.bak-20261122-pre-recipe .env &&
  ./start.sh restart`.
- Not adopted from the round: 0262 (pool scoring; its scope is ≈0.02% of prefill, and its brief's stop condition hit)
  and 0263 (drafter KV ring batch; ≈0.02%, kit spec now pinned but parked).

## 2026-11-22: 0264 `TF_DS_PLACE_BOUND_ALL` adopted (`bench-fused-next-27baa93`, recipe-0264)

- With no applied cap, the decode experts' grids stop at `top_k × rows + 1` places, the proven most distinct experts
  (`exl3_experts`: the shared expert is the last id, in every row's last slot). Outputs are unchanged by construction.
- `runs/overnight-20261122/placebound`: exact 39/39 in all arms; rounds −0.41% mean, four-stream cells −0.25/−0.91/
  −2.15%. Deployed 10:42–10:44 UTC (IMAGE swap + `TF_DS_PLACE_BOUND_ALL=1`), validated 8/8. Undo:
  `cp ~/dsv41-env-backups/.env.bak-20261122-pre-0264 .env && ./start.sh restart`. Exported to `zig/patches/0264`.

## 2026-11-23: CUDA kernel microbenchmark process and GB10 hardware guide (`docs/`)

- Scope: minor (documentation / engineering process). Files: `docs/cuda-kernel-microbenchmark-process.md`, `docs/gb10-hardware-performance-guide.md`, `continue.md`.
- Choices considered: (1) minimal `tools/fused_bench` runbook only; (2) repo-grounded GB10 SOP + rubric; (3) expanded playbook with worked repo examples + companion GB10 hardware performance guide (selected by operator); (4) playbook + new starter templates/Justfile scripts.
- Decision: publish a two-document pair — an expanded GB10 `sm_121` process playbook covering pinned compilation (`-O3 -arch=sm_121 --fmad=false --ftz=false`), byte-parity/sanitizer gates, tiered block microbenchmarking (CUDA-graph + ablations → `clock64()`/`%globaltimer` probes → `ncu` SASS PC sampling/stalls), and SASS opcode optimality evaluation, paired with a standalone GB10 hardware performance & kernel design guide synthesized from verified `runs/*` measurements — and reference both in `continue.md`.
- Why: centralizes all empirical GB10 `sm_121` hardware ceilings, opcode behaviors, counter quirks, and bit-exactness contracts so future kernel work avoids repeating measured regressions (e.g., half-warp masked `REDUX`, strided 64B stores, warp-aggregated `ATOMS`, emulated `sm_121` FP4 intrinsics, stream-0 DMA races) without adding new scripts or harnesses to maintain.

## 2026-11-23: two-lane campaign — tier-0 blockers and the scheduler cluster (scoping)

- Scope: major (campaign scoping and delegation). Operator decisions recorded here: (1) lane 2 builds only the
  scheduler-cluster items (block-row clamp, `TF_DS_DRAFT_MS` + `round_ms` re-fit, follower confidence skip) plus the
  distinct-picks counter — not the Engram overlap or the tier-2 fusions; (2) **no production restarts this session —
  stage everything, only short local tests for initial signal**; (3) the k280 hardware window runs exactly as staged
  (no greedy/traced/from-source arms added to it).
- Choices considered: lane-2 scope — scheduler cluster (selected) vs +Engram pending-row landing/FAST (larger ceiling
  ~1-2% but host-threading and landing-storage lifetime risk, and its acceptance baseline is the owed slow-cell
  experiment) vs all tier-1 vs tier-1+2 fusions (fusions need standalone epilogue timing first). GPU window — no
  restarts (selected) vs lane-1-owned pair window vs idle-window microbenches vs one combined window later. k280
  content — run exactly as staged (selected) vs extended arms first vs passprof+run.sh only.
- Implementation topology (lane board): lane A `tier0` owns the indexer/profiler seam (patches 0297-0299,
  `runs/k280/**`, the `runs/overnight-20261122` scoring scripts, Justfile) in worktree `~/Workspace/tfref-tier0`;
  lane B `sched` owns the drafter/scheduler seam (patches 0293-0296) in `~/Workspace/tfref-sched`; both branch from
  `1421187` (`kernel-stack-0280` tip) so new patches inherit `TF_DS_PASS_PROFILE` for gating. `todo.md`,
  `continue.md`, `decisions.md`, `to-benchmark.md`, `zig/patches/**`, `start.sh` and `.env` are parent-only writes
  this session (lanes report, parent consolidates) so the two lanes cannot collide on `zig/patches/README.md` or the
  notes files. Each lane: build + short tests -> fresh read-only reviewer -> fix pass.
- Decision: every new patch off by default behind its own `TF_DS_*` switch (series rule); patch numbers split
  0293-0296 (lane B) and 0297-0299 (lane A) so exports are disjoint; micro-benchmarking is host-side evidence (work
  counts, host tests, CPU-emulation parity) plus staged GPU scripts, since no GPU window is opened.
- Why: the 2026-11-23 re-score showed the apparent kernel wins were metric artifacts, so the blockers (0206 key
  parity, the CLOCK_MARKS defaults bug, the unwritten k280 gate) must clear before any timing claim is believed; the
  scheduler cluster carries the only percent-scale exact-by-construction ceilings that need no new mechanism
  (~1.5% four-stream / ~5% eight-stream for the block-row clamp, one measured -2.65% precedent for the re-fit).

## 2026-11-23: 0293 premise correction — block-row clamp is not bit-exact (minor scope)

- The sched lane's premise check falsified the ds41 item's claim that clamping the drafter pass to `N * steps` block
  rows is "bit-exact by construction (rows are causal in attention)". The drafter's block attention is **non-causal**:
  every block row attends every block row of its stream (`draft.zig:695` pick fill includes `k > j`; `draft.zig:912`
  comment; upstream `dspark.py` "non-causal query-block forward"). Rows past `steps` feed the scored rows through
  every stage's intra-block attention, so the full clamp changes drafts and confidences and its byte-compare gate
  would fail.
- Choices considered: (A) abort 0293; (B) full stage-row clamp behind a switch with the gate relaxed to
  reply-hash/verify-token comparison (ceiling ~1.5% four-stream / ~5% eight-stream, but not exact-by-construction,
  draft-depth concurrency caveat applies); (C) bit-exact subset only — clamp the vocabulary-head and
  confidence-pack inputs to the first `steps` rows per stream (optionally subsumed by 0265 `TF_DS_DRAFT_HEAD_SCORED`);
  (B deferred to the window).
- Decision (operator, 2026-11-23): **C only** — safe subset; B declined and recorded as an open decision in
  `runs/lane-sched-20261123/0293.md`. No B code in any form.
- Why: the campaign's adopt rule is exact plus neutral-or-faster; B is the only scheduler item that is not
  exact-by-construction and would need its own relaxed gate and concurrency caveats. The larger consequence is a
  ranking correction: the block-row clamp's 1.5-5% ceiling was mostly the non-causal stages, so the safely
  recoverable share is the head/confidence rows (and 0265 already measured -0.025 ms a pass at one stream).

## 2026-11-23: two-lane campaign outcome — patches 0293-0299z staged out of series (major)

- Scope: major (patches + process). Lanes: `tier0` (blockers) in `~/Workspace/tfref-tier0` and `sched`
  (scheduler cluster) in `~/Workspace/tfref-sched`, both based on `1421187`. Records:
  `runs/lane-tier0-20261123/` (0a, 0a-closeout, 0b, 0c, REVIEW, REPORT, `patches/`) and
  `runs/lane-sched-20261123/` (0293-0296, REVIEW, REPORT, `patches/`).
- Result: tier0 **0297** `TF_DS_KEYS_PROFILE` + **0298** `kitKey`/`key_kit_flip` + **0299** `clock_fold` +
  **0299z** (the review's F3 fix); sched **0294** (`TF_DS_DRAFT_MS` + `chooseK` clamp + `TF_DS_ROUND_MS_FIT`,
  two commits) + **0295** `TF_DS_CONF_LEADER_ONLY` + **0296** distinct-picks counter. **0293 produced no patch**
  (already built as 0265; premise falsified). Every patch off by default or profile-riding.
- **Staging decision: the seven patches stay in the lane directories, NOT in `zig/patches/`.** They are based on
  `1421187` = `TF_REF` + `0001`-`0283`, but the exported series stops at **0264** (0265-0283 were never
  exported), and `zig/Dockerfile` applies `zig/patches/*.patch` in filename order. Appending 0297 after 0264
  would jump the chain across 19 unexported patches and break the build. They enter the series when 0265-0283
  are exported (or when the lane is rebased onto the current series tip), whichever comes first.
- Choices considered for staging: append to `zig/patches/` (breaks the ordered chain — rejected), fold into
  `patches-parked/0206` (only 0298's runtime half belongs there — deferred to the re-roll), keep in the lane
  dirs (selected), commit to a branch (no branch policy exists for the recipe repo).
- **0298 is dead code until parked 0206 adopts it**: the fix is the encoding spec (`kitKey`) plus pins; the
  runtime half is the two kernel stores + `foldKeys` line in the re-rolled 0206, and the re-roll must also carry
  REVIEW F1's one-line `key_prof` class fix (`if (keys_done) .fold else .kit`).
- **J1 accepted as a recorded caveat:** the unswitched `chooseK` clamp fix alters replay `window_ok` outcomes
  for recordings made at >= 24-row rounds (expectations produced under the flat clamp now read
  `window_equal: false`). Served 16-row rounds are unaffected.
- **0299z placement:** kept as its own patch (the round-count fix above the degenerate-buffer guard), not
  squashed into 0299, so the review's F3 fix is separately revertable.
- Why: the campaign's adopt rule is exact plus neutral-or-faster and nothing here has been on hardware. The
  blockers are now answerable in one window (Stage A localize / Stage B probe byte parity / Stage C bundle gate
  for the IDX_FUSED divergence; `runs/k280/window.sh` + `just k280-window` for the 0280-0283 suite), and the
  scheduler cluster's remaining value is the 0294 knob + re-fit, measured apart.

## 2026-11-25: telemetry collector ships the engine's own log lines (minor scope)

- Scope: minor (config on both Sparks; no engine code, no engine restart). Record:
  `runs/observe-20261125/REPORT.md`; reapplied with `just observe-engine-logs`.
- Choices considered for getting per-request records into the store: (A) the `journald` docker log driver plus the
  collector's existing journald receiver (needs a `start.sh` change and an engine restart, and mixes engine lines
  into the syslog stream); (B) an engine change emitting the `done` fields as counters/histograms (clean, but a
  patch + image + deploy gate); (C) a `filelog` receiver on the container's json-file log (selected).
- Decision: **C**, shipping every engine line rather than a `done`-only filter. The filter was tried first and
  passed everything anyway (the engine prefixes lines with `[tensorfold] `); the volume is ~2 MB a day a rank, and
  rank 1's warnings and errors were previously invisible, so filtering bought nothing.
- The collector now runs as root. `/var/lib/docker/containers` is `0710 root:root`, so the glob needs **readdir**;
  `SupplementaryGroups=root` was tried first and failed, root also clears the pre-existing `/var/log/boot.log` denial.
- Also removed the worker's `tensorfold` scrape job: rank 1 is the follower and serves no `/metrics`, so it had been
  `up=0` on every scrape since 2026-11-11 and wrote a warning twice a minute.
- **Correction recorded:** `requests_total` was never "scraped but not stored". The engine renders it only when its
  reply list is non-empty, and the list is filled only by a hook installed when auth is enabled — production runs
  with no API keys, so the family does not exist. Per-client separation therefore also needs keys enabled; every
  `done` line carries an empty key label today.
- Why: this is the instrument the 2026-11-24 attribution said was missing (the content split of the live acceptance
  gap), and it costs no window and no rebuild. `queue_s`/`restore_s` arriving with it also splits queueing from a
  slow prefill, which TTFT alone cannot do.

## 2026-11-25: capacity window — the box saturates at 8 streams; row budget is not the lever

- Scope: **major** (capacity ceiling, production `PARALLEL` bound, and the `ROUND_ROWS` question). Evidence:
  `runs/capacity-20261125/REPORT.md`, four suites on production's own image, all arms bit-exact.
- Choices considered for the row budget: keep 16 (do nothing) / 32 / 48. **Decision: keep 16.** Round 32 is
  neutral within the 1.1% control band and 48 is −10%; the extra rows cost exactly what the extra drafts earn, so
  the "drafts 2–5 are free in the drafter" premise in `todo.md` is false. No patch, no window.
- Choices considered for the stream count: keep 8 / raise to 13 / raise to 16. **Decision: keep 8, and record 13 as
  the hard ceiling.** Aggregate throughput is 98.4 / 104.8 / 108.8 tok/s at 8/13/16 streams (+10.7% for 2× lanes)
  while per-stream halves. The round is row-bound, so more lanes buy almost nothing. The live 422 s mean TTFT with
  35 waiting is a throughput wall, not a lane shortage — raising `PARALLEL` would add queueing, not throughput.
- **Defect found:** `draft.max_rows = max_streams*5 = 80` (`draft.zig:38`) against `model.zig:565`'s
  `streams*DRAFT_BLOCK > max_rows` refusal means production's `DRAFT_BLOCK=6` caps the server at **13** streams,
  while `start.sh` advertises `PARALLEL` up to 16. 14/15/16 fail at startup with `DraftBlockTooWide`. The first s16
  attempt failed this way on all four arms. Config-only workaround: `DRAFT_BLOCK=5` at 16 streams. Follow-up in
  `todo.md`: tie `start.sh`'s `PARALLEL` bound to the block size so the server cannot be configured into a startup
  failure.
- **Incidental:** the 0017 16-row split ran on hardware for the first time and held bit-exactness on every arm
  (exact + sampled panels), so `ROUND_ROWS` above 16 is safe to sweep in future.
- Remaining lever: **acceptance rate**, not the row budget — tokens per round is pinned near 12–17 whatever the
  rows or streams. That is the M3 draft-triage question (item 6).

## 2026-11-25: M3 draft triage closed — no signal in a mid-depth hidden-state probe (major scope: a lever retired)

- Scope: **major** (a named decode lever, M3, is retired; the engine gains an off-by-default probe-dump patch). Evidence:
  `runs/m3-20261125/REPORT.md`; engine branch `probe-dump` (`27090ee` + `535b40d`), image `bench-probe-dump`.
- Question (M3's own step 0): can a linear probe on a layer-13 or layer-19 hidden state predict "this draft row will
  be rejected" at **95% precision / 50% recall**? If not, drop M3.
- Choices considered: (A) run the study as specified — done; (B) skip it as too hard — rejected, it was the largest
  unmeasured bit-exact decode lever; (C) infer from the existing `conf:` arm's −0.5% — that alone was not decisive.
- **Decision: M3 is closed. It fails its criterion at both layers.** Layer 13 AUC 0.621, layer 19 AUC 0.651, against
  a free positional baseline of 0.588 / 0.590 (~40k draft rows, honest round split). No threshold reaches 95%
  precision at either layer; even a leaky row split reached it only at 0.2% recall.
- Why: the target is a sampling outcome at a *different* row (row r dies when row r−1's pick misses its draft), and
  row r's hidden state does not encode row r−1's draw. Rows after the first break are dead by construction, which is
  why position alone scores 0.59. So the `conf:` arm's −0.5% was never a probe-quality problem: the signal is absent.
- Consequence: **do not build the `file:` probe path** (`TriageProbeUnsupported` stays). `0177`'s dead-row mechanism
  stays built and unarmed; production is unchanged. The decode lever is acceptance (workload-dependent) and
  contention, not drafting.
- Kept anyway, because it is a reusable instrument and the next probe idea should not rebuild it: the probe-dump patch
  (off by default, no new CUDA kernel, rank-gated, unit-tested) plus `capture.sh`/`train.py`. `runs/m3-20261125/README.md`
  records the two traps that cost a window (the smoke test opening the dump before the workload, and `find -xdev`
  skipping the `/spill` bind mount).

## 2026-11-25: operator decisions — keep ENGRAM_LEAD; adopt 0281+0282 once the from-source control passes

- Scope: major (a production default is ratified and a production deploy is authorised, conditionally).
- **ENGRAM_LEAD: keep it on.** It is exact (39/39 in all four arms), costs decode nothing (all cells within 0.5%), and
  gains on every cell measured: cold prefill +1.54%/+2.51% at 32K and +2.40%/+3.23% at 96K (ABBA). The record is
  corrected from "provisional" to "kept", and the gain from the Nov 19 narrow run's +4.9/+5.0/+4.5% down to ~2-3%.
  **Still owed and still the deciding evidence:** the ≥1.5M-uncached-token soak and the hour of process age, which
  the reader has never had — it was on during the Nov 13 age-related slowdown. If age ever shows a regression, the
  `.env` line comes out (`~/dsv41-env-backups/.env.bak-20261119-pre-lead` is the undo).
- **0281 `TF_DS_MARKOV_CACHE=4096` + 0282 `TF_DS_DRAFT_PREFETCH=1`: adopt, after the from-source control.** Both clear
  their gates inside image `1421187` (0281 −0.20/−0.22 ms a pass, −5.0%, miss 52.2→24.2%; 0282 `dpf1` −0.18/−0.17 ms,
  no forward cost; drafts IDENTICAL 19/19; exact IDENTICAL every arm). The condition is the build: `1421187`'s two
  executables were cross-built on the workstation, production's came from a GCC 13.3 from-source build, so a win
  against a slower cross-built baseline may not transfer.
  - **Control (no build needed):** one arm on production's own image `bench-fused-next-2bb052f` under the k280 workload,
    compared with `1421187`'s `ctl1`/`ctl2`. `1421187` is `2bb052f` + 0280–0283 with every switch off, so the two
    differ only in how the host code was compiled. If production's `pass_ms`/`forward_ms` land inside the controls'
    spread, the cross-build did not distort the baseline and the deltas transfer.
  - If it does not match, rebuild `1421187` from source on the head and re-run the mc/dpf pairs against a from-source
    control before deploying.
- **Not adopted:** 0283's L2 knobs (no winner; the kit's 3000 ns / 230 GB/s / 16 MiB stay) and 0282's `dpf2` (a wash
  on forward+pass).

## 2026-11-25: production onto `bench-fused-next-1421187` with 0281 `MARKOV_CACHE=4096` and 0282 `DRAFT_PREFETCH=1`

- Scope: major (production image and two switches). Deploy record `runs/deploy-1421187-20261125/`.
- **From-source control passed** (`runs/k280/fromsource/REPORT.md`): two arms on production's own from-source
  `2bb052f` against `1421187`'s cross-built `ctl1`/`ctl2` under the same workload give `pass_ms` 4.24/4.22 vs
  4.20/4.22 — within 0.04 ms (0.9%), an order of magnitude below the adopted gains. So the cross-built baseline is
  not distorted where the patches act, and the deltas transfer. (The forward is ~1.1% slower from source, which does
  not touch the pass; absolute forward figures remain non-comparable across the two builds.)
- **Deployed:** image `tensorfold-dsv41:bench-fused-next-1421187` (= `2bb052f` + 0280–0283) with
  `TF_DS_MARKOV_CACHE=4096` and `TF_DS_DRAFT_PREFETCH=1`, both ranks.
- **Gates, all pass:** exact panel **IDENTICAL 39/39** vs `runs/deploy-10c9866-20261122/exact.json`; `validate.py`
  **rc=0** (`all passed`, 8/8 replies identical to `runs/matrix/base.validate.json`); container env proof shows the
  image and both switches on both ranks; **MemAvailable 7.4 GiB head / 8.2 GiB worker** against the 2.5 GiB memguard
  floor (4,096 rows cost 473 MiB a rank).
- **Not adopted, deliberately:** 0282 `dpf2` (a wash on forward+pass) and 0283's L2 knobs (no winner; the kit's
  3000 ns / 230 GB/s / 16 MiB stay).
- Undo: `cp ~/dsv41-env-backups/.env.bak-20261125-pre-mwb .env && ./start.sh restart` (back to `2bb052f`, both
  switches unset). The previous production image is still on both ranks.

## 2026-11-25: the Markov cache's dynamic region — sized, worth building, not yet built

- Scope: major (a design decision and a measured sizing), no production change.
- Question (operator): replace the 4,096 static Markov bias rows (473 MiB) with a dynamic write-back region of ~256
  rows (~31.5 MiB), configurable, with an eviction signal that says when to grow it.
- **Finding 1 — a host-side write-back is impossible.** The Markov loop is `mk.stepsTapped(...)`, entirely on device;
  the pass's single host read happens after the loop, by which time `stage` holds later steps. A per-step D2H would
  break graph capture. So the region needs a small **repo-owned device kernel** after `_bias` (the kit cubins stay
  untouched) → nvcc → a from-source image build and a byte-parity gate.
- **Finding 2 — sized first, and it pays.** `TF_DS_MARKOV_TRACE` (`6a048d1`, off by default, host-only) on 2,568 real
  passes: static-256 alone hits 40.4% of step rows at 3.12 miss steps a pass (~0.446 ms). A **256-row dynamic LRU
  region hits 72.4% and saves 0.240 ms a pass — more than the 0.20/0.22 ms the deployed 4,096 static rows deliver, at
  31.5 MiB instead of 473 MiB.** LRU beats FIFO; 256 rows is the knee; 2,048 is the floor (0.71 miss steps a pass).
- Decision: **build the kernel, with 256 as the default row count (`TF_DS_MARKOV_WB`) and LRU replacement.** The
  intended steady state is static 256 + dynamic 256 (~63 MiB), which meets or beats today's 473 MiB configuration and
  frees ~410 MiB a rank.
- Monitoring signal, as asked: expose `markov_wb_{rows,hits,misses,evictions}` under the round profile and warn when
  `miss_steps/pass > 1.0` with `evictions/lookups > 0.25` (the table puts that at a region of 256 or below on this
  traffic), pointing at `TF_DS_MARKOV_WB=512/1024`.
- Caveat: the trace is one stream a pass (production's own figure is 1.42), so the figures are single-stream-pass
  numbers. Re-trace at a higher streams-a-pass before the kernel's default is locked.

## 2026-11-25: Markov dynamic region — default 256 rows; the kernel is a follow-up build

- Scope: minor (a default choice and a follow-up task; no code, no production change).
- Operator decision: **the dynamic region's default row count is 256** (the measured knee: 0.240 ms a pass at
  31.5 MiB), not 1,024 (0.330 ms at 126 MiB). Growth to 512/1,024 only on the eviction signal.
- The kernel is recorded as a follow-up action in `todo.md` with its spec (LRU, `TF_DS_MARKOV_WB`, device slot/victim
  bookkeeping, counters), its instrumentation (`markov_wb_{rows,hits,misses,evictions}` and the `miss_steps/pass > 1.0`
  with `evictions/lookups > 0.25` advisory), its cost (nvcc → from-source image build + byte-parity gate) and its gate
  (exact/sampled/`gate_drafts` then `pass_ms` ≥0.15 ms down at 256 rows).
- **Production is unchanged by this entry:** it stays on `1421187` with the static `TF_DS_MARKOV_CACHE=4096`. Selecting
  1,024 would have been a regression of the static list (0281 measured 24.2% miss at 4,096 against ~38-42% at 1,024),
  which is why the 1,024 request was resolved to the dynamic region instead.
