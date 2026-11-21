# env-vars — the second-pass series' switches, as landed (patches `0160`-`0171`)

**Status: landed 2026-11-16.** The twelve patches of `zig/patches/0160`-`0171`, exported by
`just export-second-pass` from engine branch `second-pass-int` (base `bench-prep` @ `44b2cbb`, behind
`zig/patches/0129`-`0145`). The series manifest (item provenance, lanes, gates) is [SERIES.md](SERIES.md); the
benchmark rows are in `to-benchmark.md` "Ready to benchmark: second pass".

**Patch numbers.** The landed numbers are contiguous in commit order; the plan's item numbers (SERIES.md's table,
the patch bodies' own `01NN` mentions) map as follows. **0176-0181 are plan numbers that landed as `0166`-`0171` -
they are not free slots.** `0182`-`0186` (plans 0172-0175 and 0163) have landed, and plan 0164 is closed: its decode half already exists as
`TF_DS_T15_PLACE_BOUND` (`zig/patches/0053`) and the prompt half is kit-bound. `0188` (plan 0165) and `0190` (plan 0168) have landed, and plan 0166 is closed (`0189` is the test that pins the shipped granted+slack stop).
The one remaining plan item takes `0191`+ (forward rows at the end of this file); the next export runs `--base 432d800 --first 0191`.

| landed | plan | item |
| --- | --- | --- |
| 0160 | 0160 | absorb reorder (`TF_DS_ABSORB_AFTER_PICK`) |
| 0161 | 0161 | Engram submit-early (`TF_DS_ENGRAM_SUBMIT_EARLY`) |
| 0162 | 0162 | parallel load readers (`TF_DS_LOAD_READERS`) |
| 0163 | 0167 | reply scan suffix (`TF_DS_REPLY_SCAN_SUFFIX`) |
| 0164 | 0170 | prompt-cache auto (`TF_DS_PROMPT_CACHE_AUTO`) |
| 0165 | 0171 | server knobs (`TF_DS_MAX_TOKENS_CLAMP`, `TF_DS_BODY_LIMIT_MB`) |
| 0166 | 0176 | token align (`TF_DS_TOKEN_ALIGN_MB`) |
| 0167 | 0177 | **M3** draft triage (`TF_DS_DRAFT_TRIAGE`) |
| 0168 | 0181 | Bloom front + lock check (`TF_DS_ENGRAM_ROWCACHE_FILTER`) |
| 0169 | 0179 | **M5** hot prefetch + kit contract (`TF_DS_EXPERT_HOT_CACHE`) |
| 0170 | 0178 | **M4** end state (`TF_DS_PROMPT_ENDSTATE`) |
| 0171 | 0180 | **M7** tail rows (`TF_DS_TAIL_BATCH`) |

## The rules every row is held to (inherited from `0129`-`0145`)

- **Off by default.** Unset leaves replies byte-identical and the wire protocol unchanged, except where a row says
  "behaviour change" (those are opt-in for that reason).
- **Rank scope.** `rank-local` = each rank's process reads it, no fingerprint bit, no `switches` frame; an uneven
  pair is still bit-identical. `pair-safe` = the two ranks agree by construction (the state rides the frame).
- **Bits class.** `identical` (same replies), `sequential` (a reply keeps its tokens alone; round composition or
  shapes change — gate with `tools/validate.py`'s concurrent check too), `LOSSY` (opt-in only; bits differ from a
  cold run by design).
- **Gate.** `tools/validate.py … --compare` + `tools/exact.py` where a row claims identity; `sequential` rows add
  the concurrent check. The table records intended behavior and gates. See the [armed switch screen](../combo3/RESULTS.md) and [odds failures](../combo4/RESULTS.md) for actual outcomes. The integrated 0169 and 0170 host tests also failed.

## The table

| Patch | Switch / field | Default (unset) | Scope | Bits | Semantics | Gate | Keep/drop bar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0160** | `TF_DS_ABSORB_AFTER_PICK` | off (the served order) | rank-local | identical | the round's pick launches first; the drafter taps' absorb launches after its read-back and runs under the host's accept and the pass frame's send (the drafter pass is stream-ordered after it). Both round paths. With `--profile` the absorb synchronizes again for `absorb_ms` | validate + exact | `absorb_ms` is 0.26-0.29 ms a round; the plan's ceiling is under 1%. Kept only if the round wall moves outside the noise band at 1 and 4 streams |
| **0161** | `TF_DS_ENGRAM_SUBMIT_EARLY` | off (the inline hash after stretch 0) | rank-local | identical | the round's Engram hashes compute at the drafter pass's return (both ranks) and the flatten + submit runs at the round's start where the wins are final; an uncovered window falls the whole round back | validate + exact | `engram_hash_ms` 0.98 a round leaves the critical path; kept if the round wall or `engram_host_ms` moves |
| **0162** | `TF_DS_LOAD_READERS=N` (1..8) | 1 | rank-local (load) | identical | a layer's expert span fills from N parallel O_DIRECT readers over disjoint aligned regions; uploads/layout/digests unchanged | validate + exact | the load wall: 87.8 s of 101.5 s is one serial reader; kept if the start wall moves at N=4/8. Follow-up: pinned staging + `uploadAsync` |
| **0163** | `TF_DS_REPLY_SCAN_SUFFIX` | off (whole-text scans) | rank 0 | identical | the per-round scans (stops, thinking split) resume where the last round stopped (a moved buffer restarts). The per-request arena accumulation remains (the follow-up) | validate + exact + the patch's equivalence tests | rank 0 CPU over a 32K-token streamed reply; kept if CPU or `MemAvailable` moves |
| **0164** | `TF_DS_PROMPT_CACHE_AUTO` | off (0.5 GiB) | rank-local | identical | the kept-states budget is what 70% of RAM leaves past the loaded server, at least 0.5 GiB. `--prompt-cache-gib` overrides; 0 disables | validate + exact | `prompt_cache_held_bytes` + eviction counters. On the pair's unified memory expect the floor — the arm is a machine-with-room arm |
| **0165** | `TF_DS_MAX_TOKENS_CLAMP`, `TF_DS_BODY_LIMIT_MB` | off / 32 MiB | rank 0 | behaviour change (opt-in) | clamp an explicit `max_tokens` past the room like the implicit one; the body limit is env-sized (64 matches Baseten) | validate (the clamp arm is the delta) | counts of 400s/413s a restart: clamping turns refusals into replies without surprises |
| **0166** | `TF_DS_TOKEN_ALIGN_MB` | off | rank 0 | LOSSY (opt-in) | each reply's generated spelling is kept; a resent prompt's piece carrying it byte for byte takes the recorded ids (whole-piece matches only; the substring splice is the follow-up) | validate + exact where it hits; the plan's step 0: the share of turns whose ids diverge inside the previous reply | prefix-cache hits on resent conversations (`prompt_tokens_cached`) |
| **0167** | `TF_DS_DRAFT_TRIAGE=conf:<t>` | off | pair-safe | **sequential** | **M3**: rows whose chain survival is under `<t>` (and the tail) route nothing and are treated as rejected (the input sentinel truncates the accept; a wrong call defers one token — the reply's tokens are invariant). `file:<path>` is the trained probe's hook | validate **+ concurrent** + exact | wasted rows are 0.78 a round (~3.8 ms, 11% of a step; the review's ceiling ~6%). The probe study (95%/50% at layer 13/19) keeps or drops it; `conf:` is its stand-in |
| **0168** | `TF_DS_ENGRAM_ROWCACHE_FILTER` | off | rank-local | identical | a Bloom front for the row cache's `has()`: a miss takes no lock, a positive falls through (never a false negative). The check counters `engram_rowcache_{has,fast,lock}_total` ride `/metrics` | validate + exact | the "does the lock still show" question: with the filter on, if the locked probes and the read wall stay put, the lock is not the thing to optimize |
| **0169** | `TF_DS_EXPERT_HOT_CACHE=hot:<n>` | off | rank-local | identical | **M5**'s prefetch half: the picks counted per layer on the device, the top `<n>` hot experts' trellis prefetched beside the paced one; `decoded...` refuses until the kit grows the kernels (the contract is the patch body) | validate + exact | W4's pure-read probe decides life or death (bus-bound: the candidate is dead). Measure `moe` ms with the prefetch on |
| **0170** | `TF_DS_PROMPT_ENDSTATE=N` (1..8) | off | pair-safe | identical | **M4**: the pass's end state (all rings + the drafter's rows + head_g) is kept for an identical resent prompt, which resumes with zero rows (a retry, a regenerate, `n > 1`). At most N; process-local | validate + exact | the `tail_window` miss-cause counter (0116): under ~2% of requests resending full prompts: drop. The warm-turn TTFT of such a turn is the ceiling (0.7 s at 100K) |
| **0171** | `TF_DS_TAIL_BATCH=N` (1..8) | off | pair-safe | **sequential** | **M7**'s first slice: a pass resuming a full end state computes the new rows only (the 128-row replay goes). N names the batching width the next step takes (the design is the patch body) | validate **+ concurrent** + exact's cold-delta | the warm-tail TTFT: the replay is ~0.15-0.2 s a turn (derived). The batching step's two blockers are the patch body's; the slice alone is the win |
| **0182** | `TF_DS_TOPK_FAST` | off (the eight-digit select) | both ranks by convention | identical (the same k keys, the same order) | the indexer's top-k (`tf_ds_topk_indices_kernel`) runs as `tf_ds_topk_indices_fast_kernel`: a systematic sample (every n/4096-th key) radix-selected to the sample's rank of k estimates the k-th largest key, one pass compacts every key at or above that estimate (cap 4096 a row, shared), the exact k-th is selected among the candidates by the same eight-digit rule; a row the estimate does not honour (under k above it, or over 4096) falls back to the select over the row. `topk_i64` untouched | validate + exact; the algorithm is checked host-side against a brute force (`runs/bench-prep2/topk-fast/`) | `a_idx` at layers 24-36, where scoring is constant: three reads of the row against ten. The first nvcc is the image build |
| **0183** | `TF_DS_PREFILL_GATHER_OVERLAP` | off (the main stream's gather) | both ranks by convention | identical (the same all-gather, ranks, order) | the chunk's MoE gather on a stream of its own, forked behind the MoE and joined where the pending post first reads `gm`; at an Engram layer the post waits for the rows to be read and uploaded, so the exchange overlaps that block. Two or more ranks, no 2D split. The `gather_m` mark then clocks the fork, not the gather | validate + exact | the chunk wall and tok/s at 2K and 8K: the window is the two Engram layers' host read (~70 ms a layer at 2048 rows) against a measured 8.31 ms an exchange - only those two of 40 gathers have anything to run beside them |
| **0184** | `TF_DS_IDX_BUCKET_FINER` | off (the served powers of two) | both ranks by convention | identical (same scores, the masked keys past `vis` never selected, the bucket still covers every real key) | the round's bucket rounded to a multiple of 1024 instead of the next power of two (70,000 positions: 70,656 keys against 131,072), never past the power-of-two bucket the buffers and graph keys are sized for. A non-power-of-two `nb` leaves the fused `_topk_sel` for `topk_i64` + `_topk_finish`; more captured graphs | validate + **exact.py** | `a_idx` at ~65,000 against ~66,000 tokens (the bucket step) and at a 70K context; with `0182` on, `torchTopk` is the fast select, which is the pairing to measure |
| **0185** | `TF_DS_BF16_PARTIALS` | off (fp32) | both ranks | **LOSSY** | the chunk's attention partial written bf16, gathered bf16 (the payload halves; NCCL, the ring stays fp32 and the decode rounds keep fp32) and widened to fp32 into `ga` before the FFN mixes' sum - Baseten's round-before-the-sum. Two or more ranks, no 2D split; `pb`/`gb` are not allocated when off | **exact.py is expected to differ**; validate + the recorded delta | `gather_a` and the chunk wall at 2K/8K prompts, 1 and 4 streams: the plan expects little (the cost is mostly fixed) |
| **0186** | `TF_DS_GREEDY_ARGMAX` | off (the logits' gather) | both ranks by convention | identical (same values, tie and NaN rules) | a greedy round's head exchanges each rank's (value, index) a row (R * 8 B a rank) instead of its vocabulary half (258,560 B a row) and reduces them to the same pick; `Round.map` is per round, the graph key carries it, and a sampled/draw/score round keeps the logits | validate + exact | the `head` phase and the round wall at 1/4/16 streams; nothing else reads logits on a greedy round, so a flat head means the gather was not the cost |
| **0188** | `TF_DS_HEAD_LAST_CHUNK` | off (every decoder chunk runs it) | rank-local | identical | a prompt chunk that is not the pass's last skips the head (collapse_norm, the head's group, the gather): the straddling window's chunk had its logits thrown away. `moreRowsAfter` is the test; the chunk still returns `.head`; the drafter's absorb still runs | validate + exact | the chunk wall of a prompt whose length crosses a chunk multiple (the item is "a few ms a request" by the plan's own note) |
| **0190** | `TF_DS_SNAP_TRIM` | off (every layer's ring) | both ranks | identical | a kept state carries no ring for the decoder's last half (2.81 MiB a state the replay rebuilds); `LayerShape.ring` puts that in the layout, so its planes, sizes and digest follow. Refused without `TF_DS_REPLAY_FLOOR=1` (the deployment's default). No `format.version` bump: the layout digest rides the fingerprint and the store's compat digest | validate + exact across a keep and a resume | `prompt_cache_held_bytes` and the eviction counters; the first enable makes the existing store's points foreign (they miss) |

## Not built: the odds and follow-ups, numbered `0191`+ (2026-11-16)

Numbers assigned 2026-11-16. **Nothing in this table has code**: no switch of theirs exists in
`families/dsv41/native.zig` and no engine branch (`second-pass`, `-srv`, `-load`) carries a commit for them. Rows keep
the plan's switch names so `docs/baseten-ultrafast-plan.md`'s section references still resolve; scope and bits class
come from `SERIES.md`'s plan rows, the gates are what `to-benchmark.md` will need when they land. `0182`-`0186` (plans
0172-0175 and 0163) and `0188` (plan 0165) have landed and sit in the table above; plan 0164 is closed (its decode half
is `0053`'s `TF_DS_T15_PLACE_BOUND`).

| Number | Item (plan) | Switch | Scope | Bits | First gate |
| --- | --- | --- | --- | --- | --- |
| **0191** | restore lookahead (0169) | `TF_DS_RESTORE_AHEAD=N` | pair-wide | identical | split `lane_host.park` into `prepare(job)` + unchanged admission; bounded by `Settings.restores`/`max_restores`/`Store.room`; watch room-eviction of good states and 30 s `load_wait_ns` |
| **0192** | M7 batching step (lands on `0171`) | `TF_DS_TAIL_BATCH` (wider) | pair-wide | **sequential** | forced rows must suppress emission and pass window validation before `cache_len`; `rows_max`/`round_rows` for k*T rows |
| **0193** | token-align substring splice (lands on `0166`) | `TF_DS_TOKEN_ALIGN_MB` | rank 0 | LOSSY | encode the piece's prefix/suffix around the recorded span; whole-piece matches stay the fast path |
| **0194** | load staging + `uploadAsync` (lands on `0162`) | `TF_DS_LOAD_READERS` | rank-local | identical | double-buffered spans: read overlapped with upload |
| **0195** | reply per-round arena (lands on `0163`) | none | rank 0 | identical | the scans' share shrank; token decodes and `Shown` copies still accumulate until the reply ends |
| **0196** | decoded-cache kit contract (lands on `0169`) | `TF_DS_EXPERT_HOT_CACHE=decoded` | rank-local | identical | the kit must ship a decode-to-buffer kernel + a decoded `grouped_cp` variant; until then `decoded` refuses. W4's `0142` probe decides whether the candidate lives |

Measurement-only, no number: M3's probe study (`0167`), `0168`'s row-cache lock check, the zstd recompression check,
M5's W4 gate.
