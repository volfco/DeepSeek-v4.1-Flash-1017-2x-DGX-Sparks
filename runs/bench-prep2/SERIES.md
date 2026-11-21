# Second-pass series (plan `0160`-`0181`, landed `0160`-`0171`, `0182`-`0186`, `0188`-`0190`, next `0191` (0164, 0166 closed)): the manifest

**LANDED (2026-11-16): the twelve built items are `zig/patches/0160`-`0171`** (contiguous numbers in commit order;
the plan→landed map and the full rows are `runs/bench-prep2/env-vars.md`, the benchmark rows `to-benchmark.md`
"Ready to benchmark: second pass"). **The odds then landed as `0182`-`0186`, `0188`-`0190`** (plans 0172-0175, 0163, 0165,
0168; 0166's `0189` is the test that pins behaviour already shipped), so `TF_REF fbbc971` + every patch in filename order
= `second-pass-int` `432d800` (verified by the export). Plans 0164 and 0166 are closed - already built as `0053`'s
`TF_DS_T15_PLACE_BOUND` and as `depth.Rule.budgets`' granted+slack. The one remaining odd takes `0191`+ (the **Number**
column's bold means a patch not yet written; the first column keeps the plan's number for reference), and the next export
runs `--base 432d800 --first 0191` (`just export-second-pass` carries it).

Implementation plan for the patch series from `docs/baseten-ultrafast-plan.md` "Second pass: the two engines compared
subsystem by subsystem" (lines 541-647) and `docs/benchmark-gaps-and-moe-review.md` §4.3 (M3/M4/M5/M7) and §5
(zstd/Bloom). Engine branch `second-pass` of `/home/colum/Workspace/tfref-next` (worktree
`/home/colum/Workspace/tfref-2ndpass`), base = `bench-prep` @ `44b2cbb` (the tip behind `zig/patches/0129`-`0145`).
Numbering starts at 0160 (0146-0159 left free). Export: `scripts/port-patches.sh --first 0160 --keep-generated`
(Justfile `export-second-pass`), so 0129-0145 stay untouched.

Rules (from the plans and `decisions.md` 2026-11-16): every switch off by default; pair-wide switches ride the
switches frame and refuse a mismatch; rank-local switches are set alike by convention; host tests
(`/home/colum/.local/zig-0.17.0/zig build` + `zig build test`) were the per-commit development gate. Integrated 0169 and 0170 host tests subsequently failed. The table preserves the design classes. See the [second-pass screen](../combo3/RESULTS.md) and [odds tests](../combo4/RESULTS.md) for pair results.

Writer lanes (exclusive file owners; the parent integrates by cherry-picking each lane's commits onto `second-pass`
in numeric order, then runs `zig build test` on the merged branch before export):

- **Lane A (core)** `/home/colum/Workspace/tfref-2ndpass`, branch `second-pass`: 0160, 0161, 0163, 0164, 0165,
  0166, 0172, 0173, 0174, 0175, 0177, 0178, 0179, 0180, 0181 (engine core: model/round/lanes/prompt/exl3/engram).
- **Lane B (server)** `/home/colum/Workspace/tfref-2ndpass-srv`, branch `second-pass-srv`: 0167, 0170, 0171, 0176
  (server/*, native/cuda.zig, core/prompt_cache.zig storage half).
- **Lane C (load)** `/home/colum/Workspace/tfref-2ndpass-load`, branch `second-pass-load`: 0162 (weights.zig).
- Conflict hotspots at cherry-pick time: `families/dsv41/native.zig`'s env doc comment (every switch adds a line)
  and `core/prompt_cache.zig` (0176 vs 0178).

| Patch | Number | Item (plan source) | Switch | Scope | Bits | Notes / gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0160 | 0160 | absorb reorder (2nd pass §1.2) | `TF_DS_ABSORB_AFTER_PICK` | rank-local | identical | pick launches before the taps' absorb; absorb runs under the host's accept + frame send |
| 0161 | 0161 | Engram submit early (2nd pass §4.1) | `TF_DS_ENGRAM_SUBMIT_EARLY` | rank-local | identical | scout verdict: hash at the drafter pass's return into a per-round stash (own buffer — `eh.hashes`/`eh.flat` alias the prompt fallback), submit at the round's start where the wins are final (the verify frame's trim falls back to the inline path); do NOT submit at pass return (buffer aliasing + verify-time trim); AIO only (`TF_DS_ENGRAM_AIO=1`); rank-symmetric (both ranks hold identical `l.held`); rounds keep `dedupe=false` and the DEDUPE scratch untouched |
| 0162 | 0162 | parallel expert read at start (2nd pass §5.1) | `TF_DS_LOAD_READERS=N` | rank-local | identical | several parallel O_DIRECT readers, read overlapped with upload; do NOT mmap-and-populate (memguard) |
| 0163 | 0186 | greedy rounds exchange argmax (2nd pass §4.3) | `TF_DS_GREEDY_ARGMAX` | pair-wide | identical if "first largest, rank 0 wins ties" | scout: nothing but the pick is consumed on a pure-greedy round EXCEPT a score stream (forces full logits — that round keeps the gather); per-rank (value,index) scan + reduce in rank order with strict `>`; NaN rule ("0 when the first value is NaN") keys off the global row's first value (rank 0's local[0]) — carry one flag; `acceptChain` needs picks ON DEVICE in `amax` (fold before it); `--logits-sha` and the call gate are not round consumers |
| 0164 | closed | expert grid sized by picks (2nd pass §4.4) | `TF_DS_T15_PLACE_BOUND` (decode; landed `0053`) | rank-local | identical | **closed 2026-11-16**: the decode half is T15 (under an applied prune cap); the prompt half has no bound to use (the pruner runs on decode windows only) and a place list is the kit's half. Arm: `to-benchmark.md` |
| 0165 | **0188** | discarded head at chunk straddle (2nd pass §3.4) | `TF_DS_HEAD_LAST_CHUNK` | rank-local | identical | skip `prompt.head` for a non-final chunk of a pass; the head of the straddling chunk is thrown away today |
| 0166 | 0189 (test) | Markov chain stop (2nd pass §1.3) | none - already shipped | pair-wide | sequential | **closed 2026-11-16**: the cheap variant (granted + slack) is what `depth.Rule.budgets` already does in the node-probability branch the served engine takes; `0189` is the host test that pins it. The in-graph confidence stop needs conditional nodes both ranks agree on: not built |
| 0167 | **0163** | streamed reply text scan (2nd pass §5.2) | `TF_DS_REPLY_SCAN_SUFFIX` | rank 0 | identical | scout: `Stops.visible` + `splitThinking` get a resume record like `reply_text.Hide` (per-marker first-hit offset + watermark; scan `text[watermark-(tag.len-1)..]`; full rescan when the buffer moves); text only appends so first-hit semantics are stable; the per-connection arena DOES accumulate per round until the reply ends (the robustness item is real, `server.zig:300-306`) |
| 0168 | 0190 | Snap decoder-ring trim (2nd pass §5.3) | `TF_DS_SNAP_TRIM` | pair-wide (store format) | identical if no path reads decoder rings before the replay floor | scout: layers 20-39's rings (2.81 MiB/slot) are dead weight — the replay fully rebuilds them; bump `spill/format.zig` version + `StateLayout` digest (old points become cache-misses: they lose kept state on upgrade); REFUSE the switch with `TF_DS_REPLAY_FLOOR=0` (that path reads ring rows below the floor) |
| 0169 | **0191** | restore lookahead (2nd pass §5.4) | `TF_DS_RESTORE_AHEAD=N` | pair-wide | identical | scout: split `lane_host.park` into `prepare(job)` (probe + `restoreBeginOver` + `Load`) and admission unchanged; prepare the first N queued jobs from `service`/`run`, bounded by `Settings.restores`/`max_restores` and `Store.room`; guards: room-eviction of good memory states, N reservations held while lanes busy, 30 s `load_wait_ns` |
| 0170 | **0164** | prompt-state budget from MemAvailable (2nd pass §5.5) | `TF_DS_PROMPT_CACHE_AUTO` | rank-local | identical | scout: `native/cuda.zig:464` `orelse 0.5` GiB; the CLI's promised "70% of RAM" sizing exists only on the Metal path (`cache_fit.zig`); reuse it on CUDA with the `meminfo` already read at cuda.zig:113; `--prompt-cache-gib` stays an override |
| 0171 | **0165** | server request knobs (2nd pass §5.5) | `TF_DS_MAX_TOKENS_CLAMP`, `TF_DS_BODY_LIMIT_MB` | rank 0 | behaviour change (opt-in) | scout: clamp = drop the explicit-max_tokens refusal (`chat.zig:199`) and use the clamp at :201 (goldens assert the 400: switch off keeps them); body = `http_body.zig:8` module constant 32 MiB → env-sized 64; **the plan's "stop cancels in-flight replies after 2 s" is NOT in the code** (a stop cancels per-token/immediately; no 2 s constant) — claim corrected in the docs, no patch |
| 0172 | 0182 | exact one-pass top-k (2nd pass §2.2) | `TF_DS_TOPK_FAST` | pair-wide | identical (exact top-k, same tie rule) | threshold+compact with exact fallback in `dsv41_ops.cu`; UNVERIFIED-BUILD (.cu never through nvcc here) |
| 0173 | 0183 | prompt gathers overlap (2nd pass §3.1) | `TF_DS_PREFILL_GATHER_OVERLAP` | pair-wide | identical (transport) | scout: `gather_m` is first consumed at the NEXT layer's mixes — overlap it with the next layer's Engram block (side stream or double-buffered gm); gather_a and the Engram gather have zero slack; the ring is decode-shaped (≤ ~217 rows, one gather in flight) |
| 0174 | 0184 | indexer buckets, repo half (2nd pass §2.1) | `TF_DS_IDX_BUCKET_FINER` | pair-wide | identical if fills reproduced | scout: pow2 lives in the kit `_topk_sel` cubins + graph cache; finer nb falls into `topkI64` + `_topk_finish` (works, loses the fused launch), `candFast` wants nb%8==0, `idx_all` nb≤512; dead tiles make finer buckets work-sizing only (selections unchanged) |
| 0175 | 0185 | bf16 attention partials (2nd pass §4.5) | `TF_DS_BF16_PARTIALS` | pair-wide | LOSSY | scout: stage it repo-only — wo GEMM writes bf16 pa, gather .bf16 (the payload halves), widen to fp32 before the kit `_hc_*` sum (same RNE-before-sum semantics as Baseten); ring gather is fp32-only so NCCL; expect little (cost is mostly fixed) |
| 0176 | **0166** | token-spelling align (2nd pass §5.6) | `TF_DS_TOKEN_ALIGN` | rank 0 | LOSSY | scout: NO stored spelling exists — needs new storage (each reply's emitted ids keyed by conversation + byte-span digest, beside Kept/Point) and an encode-path hook (`server/prompt.zig prepare`) that splices stored ids on byte-identical spans before the cache lookup; normal encode on any mismatch |
| 0177 | **0167** | M3 draft triage (review §4.3) | `TF_DS_DRAFT_TRIAGE=<probe>` | pair-wide | sequential | scout: repo-side — dead rows = `pick = -1` in every slot (incl. shared) at the layer-14 cut (the host already syncs there for Engram); `decode_prep` combines no-slot rows (residual only); host treats dead as rejected (sentinel in `o.sampled`, `acceptPath` truncates); the mask is prefix-closed so later rows' garbage taps/keys are only read by dead rows; probes: `conf:<t>` (host rule from the previous pass's `survivals`, no new kernel) and `file:<w>` (a linear probe GEMV over `rd.taps`/`rd.h` at the cut — UNVERIFIED, no trained weights exist); step 0 (probe study) still owed |
| 0178 | **0170** | M4 identical-prompt end state (review §4.3) | `TF_DS_PROMPT_ENDSTATE=N` | pair-wide | identical (the state is a cold pass's own) | scout: hook is `Rules.eligible` (prompt_cache.zig:455-460 refuses `at + window <= len`, and `at >= len` — the `tail_window` miss cause); the end state already exists as Snap + `Tail` + drafter rings + `ch.head_g` (exact_resume keeps it off-grid); keep it keyed by full prompt ids for a resend, zero replay; step 0: `tail_window` counter on production traffic |
| 0179 | **0169** | M5 hot-expert decoded cache + prefetch (review §4.3) | `TF_DS_EXPERT_HOT_CACHE=<spec>` | rank-local | identical | scout: NO decoded-weights buffer or decode step exists (trellis decodes in-register inside the kit `grouped_cp` cubin) — the decoded cache is a documented kit contract (switch value `decoded` refuses with `error.KitKernelMissing`); implementable: an online per-token→expert prior (ids compacted per round, read back with the round's existing sync) + prefetch of predicted/hot experts' trellis ranges on the paced prefetch stream (16 MiB budget, `round.zig:791-808`); step 0: W4's pure-read probe (`0142`) |
| 0180 | **0171** | M7 pooled warm tails (review §4.3) | `TF_DS_TAIL_BATCH=N` | pair-wide | sequential (W8 closed negative: bits move with row shape) | scout: honest blockers — one cache view / one `wlo` / one Engram `seq` / one head row per chunk, kit has no per-row-view attention variant; reduced slices: per-row `wbase` exists in ring mode (≤16 rows) and `Caches.twinned` two-plane views (KV-shard borrowing) + negative-id dead-n-gram seq trick may carry 2 tails; land the largest honest slice or the design note (blocker list + kit/host work) |
| 0181 | **0168** | Bloom front for the row cache's `has()` (review §4.5) | `TF_DS_ENGRAM_ROWCACHE_FILTER` | rank-local | identical | scout: plain Wyhash 2-seed Bloom over `Key{layer,row}` (atomic u64 bits, insert under the existing mutex, lock-free negative in `has()` — false positives fall through to today's locked probe; stale bits after eviction only cost a probe); plus the "does the lock still show" check counter on `TF_DS_ENGRAM_METRICS` (`api.EngramStats`: lock waits / probe lengths — RowCache has no counters today); after W1 (landed `0136`) |

Checks beside the patches (not engine patches):

- **zstd recompression** — `runs/zstd-recompress/check.sh` + `just zstd-check` (one 1 GiB slice of the rank cache
  through `zstd -19`, lossless-verified; under 3% saving: close). Written and smoke-tested locally; not run on the
  pair. Source: review §5's "a lossless recompression of the trellis" row and plan §4.5's entropy-coding row.
- **Bloom "is the lock still showing"** — rides the Bloom patch's counter (plan `0181`, landed `0168`; check the
  row-cache lock wait/probe lengths on `TF_DS_ENGRAM_METRICS` output before enabling the filter).

Not patchable here (kit or config, recorded so nobody re-derives): larger draft block (kit, low prior), replay scores
every key at layers 24-36 (kit variant for 128 rows), old/new key split at the KV-source layers (kit), the indexer
buckets' kit half (`_topk_sel` powers of two), chunk_rows (pool block / resume grid / fingerprint — not a free knob),
device accept chained into the pass (declined: "do not build the one-frame round", review §3.3), the allocator's cost
table (`TF_DS_ROUND_MS` config; `0143` is the learned variant), split-by-expert TP and the other §7 ruled-outs.
