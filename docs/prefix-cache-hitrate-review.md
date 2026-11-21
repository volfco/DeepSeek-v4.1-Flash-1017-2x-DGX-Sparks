# Prefix-cache hit-rate review

A read-only review of the prefill/prefix-caching system (the prompt cache that decides how much of a prompt
skips prefill, plus Engram's row reads, which the prompt-cache hit rate directly drives) for improvements that
increase the cache hit rate. Target workload: **agent/tool-loop traffic** (shared system prompts, sub-agent
fan-out, multi-turn tool sessions) **and long chats** up to the 1,048,576-token window, including after
eviction and restarts. **No code was changed**; every item below is a proposal for the operator to pick.

Code read: `/tmp/tf-engine` = BertholomusAI TensorFold `deepseek-v41-zig` @ `fbbc9713` + `zig/patches/`
0001-0098 (commit `f72c1573`), and this repo's patches/docs/scripts. Four independent review lanes wrote
`/tmp/cache-review/{A,B,C,D}-*.md`; two independent validators re-checked every cited claim against the code
(`/tmp/cache-review/V1-validate-AB.md`, `V2-validate-CD.md`). **Every finding below passed validation** unless
marked *partial*; two validation corrections are folded in (see §Validated). Findings are ranked by expected
hit-rate impact for the target workload. Bit-identity is sacred here: anything that can change reply bits is
flagged **LOSSY** and must stay opt-in behind the identity harnesses (`tools/exact.py`, `tools/validate.py`).

Hit rate is `usage.prompt_tokens_details.cached_tokens / prompt_tokens` per request (and
`prompt_tokens_cached_total` on `/metrics`). Note the engine's `prefix_cache_hits_total`/`prefix_cache_queries_total`
report cached **tokens**, not lookup counts (`server/metrics.zig:157`), so request-level hit rates need item 12.

## Ranked findings

### 1. Shared-prefix discovery stops at the first user message — per-agent tool sets share nothing
- **Mechanism:** `zig/src/server/prompt.zig:146-163` (`systemPrefixLen`) probes the render up to the first `user`
  message (synthetic probe user; 0 below 512 tokens); `server/chat.zig:192-196` plans three shared cuts
  (`system_len-2048`, `system_len-512`, `system_len`, each ≥ 512). `Store.marks` (`prompt_cache.zig:925-950`)
  plans only those plus the request's own history/stable marks; anything shared *after* the first user content
  is never kept as a shared cut, and `find` matches kept entries only.
- **Miss:** sub-agent fan-out where agents share a system prompt but get **different tool lists** (prompts
  diverge inside the tools block) or where the shared document/task brief sits in the first user message —
  every follower recomputes its whole prompt. Hit rate across sub-agents falls to ~0.
- **Proposal:** plan cuts at several block boundaries (system-only, system+tools, first-user-content); hash the
  rendered (system, tools) blocks and key discovery on block identity; allow client-supplied `shared_prefixes`
  (`engine_api.zig:80` is engine-internal today).
- **Impact:** high for fan-out with per-agent tools; none for uniform-tool fan-out. **Risk:** bit-safe for dsv41
  (extra marks only add keep points at existing 2048-row chunk ends); gate for `planned` families (GLM), where
  marks re-cut the pass. **Confidence:** high (validated; V1).

### 2. Edit/regeneration recovery is gated away (`history_len/2`) and can diff against a stale prompt
- **Mechanism:** the only history-edit recovery is the `stable` mark in `Store.marks`
  (`prompt_cache.zig:925-950`): kept only when `stable >= history_len/2 and stable < history_len`
  (`:933`), with `previous` = `e.last` of the entry `find` returned (`:855-856`), refreshed only when that
  entry is kept or resumed (`:909-922`). *Validated nuance:* against a stale same-conversation `e.last` the
  divergence point is still computed correctly; the operative miss cause is the `history_len/2` gate, which
  drops the mark for any edit in the first half of history. `min_gap` (256) drops near-resume marks too.
- **Miss:** regenerate/edit of an early message, Anthropic `clear_thinking_*` edits (patch 0030), harnesses
  that re-scope old reasoning: the mark is dropped, the deepest match is an old turn's chunk end or the shared
  system cut, and **every later turn re-prefills from there** (a 100K+ session recompute per turn).
- **Proposal:** (primary) replace the `history_len/2` gate with an absolute floor (`stable >= from + min_gap`
  and `>= min_prompt/4`); (secondary) compute `stable` against the most recently *served* prompt (a small ring
  of last keys) instead of `e.last`.
- **Impact:** high for edit-heavy agent traffic; zero cost when no edit happens. **Risk:** none to bits (adds
  kept states only). **Confidence:** high (V1: mechanism verified, proposal re-weighted).

### 3. Under pressure, the states everyone needs die first — in both eviction layers
- **Mechanism (two independent layers, both live):** (a) `Store.victimBut` (`prompt_cache.zig:1103-1125`)
  evicts "moved on" states first, and *any* longer descendant entry (`born > e.born`) counts as a superseder —
  so a fork/branch parent's turn state is the first victim exactly when its children fan out (only `shared`
  entries are exempt). (b) The KV pool's `placeKept` (`families/dsv41/lanes.zig:1373-1409`) stales the **first
  non-stale Snap in keep order** (oldest keep first) with no shared bit consulted; a staled Snap fails
  `Lanes.usable` and the hit is gone (`keptOver` `:1412-1415` only steers resume placement). Store-level shared
  protection never reaches the pool layer.
- **Miss:** 8-stream agent traffic with divergent tails: the shared system-cut state (oldest keep) is staled or
  the branch parent evicted; every subsequent follower recomputes the whole shared prefix. One-fill fan-outs
  become N fills after hours of traffic.
- **Proposal:** (a) count "moved on" only within the same conversation (`Entry.group`) or via the lineage link
  (`KeepInfo.parent`/`fork`); (b) give `Snap` a shared flag (plumbed from `KeepInfo`) and order `placeKept`
  victims: superseded → non-shared oldest → shared oldest.
- **Impact:** high at 8-stream/long-context pressure; near zero at light load. **Risk:** eviction/placement
  order only; no bit risk. **Confidence:** high on mechanism (both V1-verified); frequency under load is a
  projection — check the eviction counters (item 12) and `tools/prefix_fanout.py --expect-one-fill` interleaved
  with divergent traffic.

### 4. `SPILL_MIN_TOKENS=8192` keeps short agent sessions entirely off disk
- **Mechanism:** `store.zig:1265` refuses `at < min_tokens` (`.short`) and `spill_tier.zig:1217-1218` refuses
  in `keepFn` before posting to followers; keeps are only possible at chunk ends ≤ `len − 128`
  (`lanes.zig:913,1141,1162`, window = `sliding_window` = 128, `config.zig:177,265`). For prompts below ~10K
  tokens the only keep is under 8192 → nothing is written, so eviction or a restart loses everything
  (the `--learn` imprint is shared-cuts-only and not enabled in this deployment).
- **Miss:** tool-loop sessions of 4-9K tokens: fine in RAM, zero recovery after eviction/restart; sub-agent
  fan-out with short tasks never persists even the shared system state when the system block ends below 8192.
- **Proposal:** exempt `shared` points from the min_tokens gate (highest amortized value); and/or lower
  `SPILL_MIN_TOKENS` to 2048 (deployment knob, `scripts/port.sh:89`).
- **Impact:** high for the agent workload's restart/eviction recovery. **Risk:** more small points; no identity
  risk. **Confidence:** high (V2-verified).

### 5. The 4-turn disk retention deletes rewind/branch resume points while their blocks stay on NVMe
- **Mechanism:** `core/spill/store.zig:1167-1184` `trimPath` runs at every point publish (`:2180`): the newest
  `SPILL_TURNS` (=4, `port.sh:91`) turn points survive the active path; older manifests are removed
  immediately ("they free their small files only") — the blocks stay (the newer point's manifest names them)
  but the **cut** becomes unresumable.
- **Miss:** regenerate/edit/branch back past the newest 4 turns: the longest match falls to the shared
  system-cut point and the middle (up to 500K tokens) is re-encoded on every retry; same for sub-agent
  branches past 4 kept turns.
- **Proposal:** stride instead of truncate: keep every Nth older manifest (or one per K×2048-token stride,
  e.g. 64K) — manifests are kilobytes and blocks are shared, so disk cost ≈ 0. Deployment mitigation today:
  `SPILL_TURNS=0` disables the trim and leaves eviction to pressure (`victims()`).
- **Impact:** high for rewind/branch-heavy agents and interactive long chats; low for pure append.
  **Risk:** none to bits (more resume candidates). **Confidence:** high (V2-verified).

### 6. Engram ROWCACHE must be process-lifetime, keyed `(layer, row id)`, and hook the rounds' AIO path
- **Mechanism:** Engram row ids are a position-free n-gram hash of the backward ≤4-token window
  (`engram.zig:130-153`), so every unchanged history position yields byte-identical ids across turns and
  requests; the same reads repeat whenever the prompt cache misses or partially resumes (each turn re-reads
  the boundary chunk's rows; a fan-out miss re-reads the whole shared prefix at **24 rows/token/rank**,
  ~6.2 KiB table bytes, up to ~48 random 4 KiB reads). Row-id ranges **reset per layer** (`engram.zig:104-115`)
  — a cache keyed by id alone serves wrong bytes across the two Engram layers. *Validated correction:* with
  `ENGRAM_AIO=1` (the default) decode rounds read via `aio.start`/`aio.wait` (`round.zig:1697,1715`) and
  bypass `prompt.zig:1766`'s `readRows`; hooking only `readRows` captures **zero** decode-round reuse.
- **Miss:** after eviction/restart the recompute re-reads rows read minutes earlier; every decode token re-reads
  recurring bigram rows (rank 0's 8×2-gram columns).
- **Proposal:** read `to-benchmark.md`'s Row-cache row as **process-lifetime** (survives request/prompt-cache
  boundaries; only a restart clears it) and refine its spec: key `(layer table, row id)`; consult/insert at
  `readRows` **and** the rounds' AIO submit/wait path (plus the `round.zig:1716` pool fallback for
  `ENGRAM_AIO=0`); mutex or per-reader sharding (4 concurrent readers). Composition: with `0095` mapped
  scales, cache weight bytes only (scales come from the mapping) — ~3% footprint saved. Keep `0096` as the
  zero-memory intra-chunk switch; extending 0096 across chunks with eviction *is* this cache.
- **Impact:** high for both workloads — it is what makes every unavoidable miss (items 3-5, restarts) cheap;
  cross-request shared prefixes hit at 100%. **Risk:** silent bit corruption if keyed wrong (mitigate: include
  the table identity in the key; `exact.py` gate); memory counted against the guard. **Confidence:** high
  (V2-verified with the AIO correction).

### 7. The `shared` flag misses its own states: chunk-floor mismatch, `min_gap` drops, a 64-entry FIFO — and it is dead off-tier
- **Mechanism:** `keepWith` flags `shared` only if the state's key hash sits in a 64-entry FIFO noted at
  `lookup` (`prompt_cache.zig:883-893,962-975`). The note is keyed at `floorGrid(w)` of the raw cut while
  dsv41 keeps at the chunk end: when the `sys_len` cut is dropped by `min_gap` (short first questions) and
  `sys_len % 2048 < 512`, the deepest shared state (kept via the history mark at `floor2048(sys_len)`) has no
  matching note → `shared=false` (V1 cross-check confirmed the arithmetic; the ≥512 regime is the only one the
  tests cover). With the spill tier off, `rules.grid == 0` (`cuda.zig:458-470` gates grid/window/usable on the
  tier) so **every off-grid shared cut** notes a key that never equals the keep key (*validated:* an exactly
  aligned `w % 2048 == 0` cut still matches). ≥64 interleaved distinct cuts also rotate a key out.
- **Miss:** unflagged shared states are `victimBut`'s preferred victims once their own conversation grows
  (item 3) and are not `share`d on disk (charged to one conversation's spill quota). Off-tier deployments
  never protect shared states at all.
- **Proposal:** derive `shared` from the marks plan at the `kept` hook (the request already carries
  `shared_prefixes`; pass "this keep came from a shared mark" in `KeepInfo`) — one fix covers the FIFO, the
  floor mismatch and the off-tier case. Interim: set `rules.grid` unconditionally (window stays tier-gated).
- **Impact:** med-high for multi-turn agent sessions and multi-tenant fan-out. **Risk:** none to bits.
  **Confidence:** high (V1-verified).

### 8. The cache budget governs the wrong resource: 0.5 GiB default, ring-copy bytes while snaps pin pool extents, and stale snaps are invisible off-tier
- **Mechanism:** `cuda.zig:453` `want = (prompt_cache_gib orelse 0.5) GiB` (no `cache_fit` 70%-of-RAM default
  on the CUDA path — that is Metal-only; the deployment passes no `--prompt-cache-gib`). `Snapshots.bytes`
  counts constant ring bytes per state (`lanes.zig:1442-1445`) while a kept Snap pins its **whole pool
  extent** (`lanes.zig:1448-1466`) — ~0.9 GiB per 1M positions per rank (`scripts/config.sh:126`), ~150 MiB
  for a 171K-token turn. So the 0.5 GiB store budget (~30 states) never trips while pool extents fill and
  placement steals kept rows. Separately, `Snapshots.usable` (staleness) is installed **only with the spill
  tier** (`native.zig:299-302`, `cuda.zig:458-467`): with `KV_SHARD=1` and spill off, `find` can return a
  staled Snap whose restore fails → full prefill (`lane_host.zig:255-272`).
- **Miss:** 8 concurrent long tool sessions (and long single chats): the resource that actually exhausts
  (pool rows) is unmeasured by the cache's eviction policy, and stolen states cost full re-prefills instead of
  shallower resumes.
- **Proposal:** charge extents in `bytes(at)`/`charged` (~890 B/token) so `want`/`victimBut` measure real
  memory; size the deployment default explicitly (or port `cache_fit` to CUDA); install `usable` + grid/window
  whenever snaps can go stale (KV shard), not only with a tier.
- **Impact:** high for both workloads at once (governs the binding resource). **Risk:** a truthful budget
  evicts/refuses more at 0.5 GiB — raise `--prompt-cache-gib`/`TF_DS_POOL` alongside; no bit risk.
  **Confidence:** high (V1-verified; MiB/state figures estimated).

### 9. Small prompts share nothing: `min_prompt=4096` keeps nothing at all, and cuts below 2048 tokens cannot be kept
- **Mechanism:** `Rules.min_prompt` (4096, `prompt_cache.zig:399`) gates the whole `marks` plan (`:926`) —
  shared cuts included — and nothing in the engine or deployment sets it. Below one chunk (2048 rendered
  tokens) a cut floors to 0: the note is skipped (`prompt_cache.zig:857-858`) and nothing is ever kept at 0
  (`lanes.zig:913`, `keepWith` `:960`, `snapSaveFn` `NotAChunkEnd`).
- **Miss:** sub-agents with compact system prompts (< ~2-4K tokens) and short task lines — every request
  computes everything; `prefix_fanout.py --system-tokens 1500` shows it immediately.
- **Proposal:** exempt shared cuts from `min_prompt` (an extra save is amortized over N requests, unlike the
  single-conversation rationale in the rule); make `min_prompt` a serve knob.
- **Impact:** medium for small-agent fan-out breadth. **Risk:** more keeps against the budget (item 8 makes
  the trade visible); bit-safe. **Confidence:** high (V1-verified).

### 10. `presence_refused`: one rank's missing point forgets it everywhere and turns the hit into a full prefill
- **Mechanism:** `spill_tier.zig:1354-1410` `beginFn`: every rank must answer `present`; a follower's
  `missing`/`damaged`/`incompatible` sets `forget = true`, removes the point everywhere and the request
  prefills; only the next keep rewrites it. (`.incomplete` deliberately does not forget; rank 0's own-missing
  path refuses without a broadcast.)
- **Miss:** rank 1 prunes a point one publish earlier than rank 0 (asymmetric `trimPath` timing) or one write
  cancels under staging pressure → the paired request and every later turn of that conversation miss disk until
  a fresh keep lands. On a 500K conversation that is a minutes-long prefill cliff.
- **Proposal:** repair instead of forget on exactly-one-rank `missing`: keep rank 0's copy, re-post the write
  to the missing rank (the write path already repairs pruned followers, `spill_tier.zig:1249-1252`); forget
  everywhere only on `damaged`.
- **Impact:** medium (rare events, big cliffs). **Risk:** bounded repair retries; restore still waits for every
  rank, so no identity risk. **Confidence:** high (V2-verified; frequency check: `presence_refused` on
  `/health`).

### 11. Compaction is a bounded one-shot loss plus an attribution reset — and harness-side rewrites are the unbounded case
- **Mechanism:** `server/compact.zig:281-298` `assemble` keeps leading system/developer messages, inserts a
  generated note and the kept tail (≤20K tokens); `messages.zig:297-304` merges the leading block identically
  before and after, so the **system token prefix survives** and the compacted request resumes at
  `floor2048(system_len − 512..0)`; everything after the note is new tokens (one ≤~30K prefill). Later turns
  resume the new cut normally. The note replaces the middle, so the conversation's first user message is
  usually dropped (past ~20K cut tokens) → `roots`/`chat_match` attribution resets (item 16). The engine's
  `Stamp`/`conversationKey` are reply-metadata and note-file keys, not cache keys.
- **Miss:** each compaction of a long chat costs one bounded prefill; client-side harness compaction that
  *rewrites* the system block (prepended memory, reordering) destroys even the system-prefix match → full
  window recompute per turn. That is the unbounded case.
- **Proposal:** pass `compact_mem.conversationKey` (or the pre-compaction group) as the request's conversation
  identity so keeps stay in one conversation for retention/quota; plan a shared cut at the marker boundary;
  document the harness contract (append to the system message, never rewrite it). An **append-only** compaction
  variant (summary appended, nothing removed) keeps pre-compaction states exact prefixes — **LOSSY** vs today's
  compaction output, opt-in only.
- **Impact:** medium (bounded cliffs in long chats; the contract doc prevents the unbounded case).
  **Risk:** the variant changes compacted prompts (flag it); the rest is attribution-only. **Confidence:** high
  (V1+V2 independently verified and consistent).

### 12. A hit-rate campaign cannot see what it is trying to improve
- **Mechanism:** memory-cache counters exist in `/health`/`/metrics` only when a tier or grouping is on
  (`lane_host.zig:317-318`, `engine_api.zig:252-254`) — with `SPILL_GIB=0` the default pair publishes only
  token totals. There is no resume-depth distribution, and `misses/refused/evicted/failed` each merge several
  causes (`prompt_cache.zig:895-922,1006-1020`). `prefix_cache_queries_total` counts tokens, not lookups.
  Engram's `engram_lookup_seconds` counts rank 0's reads only (the ranks split columns 8/4
  2-gram/3-gram/4-gram — rank 0 is repeat-rich, rank 1 is not), so dedupe/row-cache A/Bs measured on rank 0
  alone overstate the keeping-worthiness.
- **Proposal:** publish the memory `CacheStats` regardless of tier; add a resume-depth histogram (`from/len`)
  and split the counters by cause (no-prefix / tail-window / stale / budget / restore-fail / superseded / LRU);
  record `engram_lookup_seconds` on **both** ranks for any Engram A/B. This is the prerequisite for ranking
  and verifying items 1-11.
- **Impact:** not a hit gain itself; required to measure all of the above. **Risk:** none. **Confidence:** high.

### 13. Engram: image positions' rows are read for a contribution that is discarded
- **Mechanism:** image positions hash to constant pad-only ids (`engram.zig:138-140`); their rows are read,
  projected and gated, then `engramApply` overwrites the gated output with pre-gate streams
  (`prompt.zig:2034-2041`). Within one chunk `0096` collapses the repeats; across chunks/requests they are
  re-read. **Proposal:** the item-6 row cache covers pad ids automatically (one read ever per (layer, pad-id));
  optionally a shape-preserving skip (fixed bytes in place of image rows' inputs — **LOSSY unless proven
  GEMM value-independent**, gate with `exact.py`). **Impact:** medium on screenshot-heavy agent traffic.
  **Confidence:** high (V2-verified).

### 14. Engram: cross-request prefetch of the next queued prompt's rows
- **Mechanism:** all read-ahead is within-pass (`model.zig:1425-1436`, `engram_lead.zig:83-93`,
  `round.zig:1675`); `engram_aio.touch` exists but has no served caller. The queue already knows the next
  request's resume point. **Proposal:** a low-priority worker hashes the next admitted prompt from its known
  resume point onward and fills the item-6 row cache on misses — its own ring (never share `paio`, which blocks
  in order), one staging chunk, drop on dequeue change. Caveat: with `ENGRAM_AIO=1` reads are O_DIRECT, so the
  fill must land in the row cache (page-cache warming buys nothing); under `ENGRAM_AIO=0` warming does work.
  **Impact:** medium on TTFT for queue-deep agent traffic. **Risk:** contention/memory guard; no bit risk.
  **Confidence:** medium (architecture fits the seams; needs a bench).

### 15. Tool-call and reasoning replay must round-trip byte-for-byte — enforce the contract, recover with item 2
- **Mechanism:** assistant turns re-render through `model/chat_template.jinja:182-200` (strings raw,
  `v | tojson` for other parameters) over `tool_parse.zig`'s compact re-serialization; with tools present the
  template keeps every turn's `reasoning_content`, and `reasoning_effort` sits near the prompt start. Any drift
  (float repr, schema-coerced `"7"`→`7`, client-reformatted arguments, a gateway losing *both* `reasoning` and
  `reasoning_content` — patch 0047 already heals the LiteLLM `reasoning` case) breaks the exact prefix at that
  turn and every deeper state stops matching.
- **Proposal:** no server-side normalization is bit-safe. Document the harness contract (echo
  `reasoning_content` and `tool_calls` untouched; stable tool order and effort per conversation); rely on
  item 2 for recovery at the divergence; an **opt-in** canonical tool-call renderer would make first-turn
  mismatch a one-time cost (**LOSSY**, flag-gated).
- **Impact:** high whenever a harness violates the contract; zero when it holds. **Confidence:** high on
  mechanics (V1-verified with the 0047 correction); live harness frequency unmeasured.

### 16. `join`/`continuedChat` merge conversations whose first 32 reply tokens match
- **Mechanism:** `chat_match = 32` (`prompt_cache.zig:371`): identical-task greedy sub-agents (same system +
  first message + templated first reply) fall into one conversation group; with `SPILL_QUOTA_GIB` on, one
  branch's growth quota-trims the other's turn states. Post-restart, groups are minted fresh (random
  `group_seed`) while adopted entries carry the old group (accounting splits until the branch merge absorbs it).
- **Proposal:** require the match at the fork point too (prompt equal at the matched entry's `at`), not just
  `root+32`; carry the root/group through the point for restart continuity. **Impact:** low-med on hit rate,
  medium on quota correctness for fan-out. **Risk:** none to bits. **Confidence:** high (V1/V2-verified).

### 17. Restore-room and keep-refusal economics under concurrency
- **Mechanism:** restores charge `restoring_bytes` to the memory budget and `keepWith` refuses new keeps while
  restores hold it (`prompt_cache.zig:995-1003`); pool room for a restore takes kept rows oldest-first
  (`lanes.zig:1377-1409`); `ContextFull` waits (`load_wait_ns`) then admits on memory. Restore always beats
  re-prefill on cost (~1.8 MB/block local NVMe), so the limits are caps and presence, not economics — but a
  refused keep is unrecoverable while a deferred restore is not.
- **Proposal:** when both would fail, defer the restore rather than refuse the keep; or cap `restoring_bytes`
  at half the budget; at minimum count keep-refusals-due-to-restores separately (item 12).
  **Impact:** low-medium (post-restart storms, many conversations). **Confidence:** medium.

### 18. Measurement context: the keep ceiling `floor2048(len − 128)` is identity-bound — do not "fix" it
- **Mechanism:** keeps/resumes only at 2048-row chunk ends ≤ `len − window` (`lanes.zig:913,1141,1162,1452`;
  `spill_state.zig:637-654`), window = `sliding_window` = 128. Each turn recomputes `delta + 128 + 0..2047`
  grid slack: a "perfect" 8K-token turn reads ≤ ~74% cached, a 500K turn ≥ 99.5%. *Validated refinement:* the
  "one chunk behind" straddle case is `len % 2048 ∈ [2,127]`, not `[2,128]`. The rule is **correctness**:
  rows before the replay cut run encoder layers only; a deeper keep would store decoder-layer state that a
  fresh pass does not compute → **LOSSY**.
- **Proposal:** none engine-side. Report the slack (`len − floor2048(len − 128)`) in the `prompt cache:` line
  or `/health` so hit-rate expectations are normalized; the report at most the last few percent of cached
  ratio is structural.

## Corrections to `to-benchmark.md` (Engram rows)

1. **Row cache row:** re-scope to *process-lifetime* and *cross-request/cross-turn* (its text says only "Repeats
   across chunks and rounds"), key `(layer table, row id)` (id ranges reset per layer), hook `readRows` **and**
   the rounds' AIO path (default `ENGRAM_AIO=1` bypasses `readRows` for rounds). See finding 6.
2. **Dedupe/Row-cache measurement:** `engram_lookup_seconds` covers rank 0 only (8×2-gram columns vs rank 1's
   8×4-gram); repeat-driven read drops are rank-asymmetric. Record both ranks before scoring the switches.
3. **Image-pad rows:** picture positions read constant pad rows whose gated output is discarded — with dedupe
   on, the within-chunk part is already collapsed; a row cache absorbs the rest. Context for the dedupe runs.

## Hygiene (stale references)

- `tools/prefix_fanout.py:2` references `PORT-PREFIX-SHARING.md` ("gates 1, 4 and 5") — the file is absent from
  the repo (it lives in the GLM-5.3-Flash recipe, or was never ported).
- `model.zig:1103` and patch `0012` cite `docs/port/dsv41-state.md` — absent from this repo.

## Validated / not verified

- Two independent validators re-checked every cited claim (`/tmp/cache-review/V1-validate-AB.md`,
  `V2-validate-CD.md`): all findings hold; PARTIALs folded in above (A-F-A1 → finding 2, A-F-A4 → finding 15,
  A-F-A7 → finding 18, B-F-D → finding 7, D's hook → finding 6). No contradictions between the four lanes.
- Not verified (needs the live pair or measurement): the served checkpoint's `config.json` `sliding_window`
  (128 is enforced by the engine's shape check, `config.zig:265`); per-state MiB and eviction frequencies under
  8-stream load (projections from victim-ordering code); Engram prose repeat-rate percentages (analytic); the
  live pair's `--prompt-cache-gib`/`TF_DS_POOL` (defaults verified in `scripts/`). Measurement proposals here
  need a bench campaign row, not a code read.

*Skipped: draft/speculative paths except draft-row reads; `spill_state.zig` byte layouts; upstream PR review
(GLM port provenance). Risk to know: findings 3, 7 and 8 are code-traced but not reproduced on hardware —
before implementing, take one baseline of the counters in finding 12 on the live pair.*
