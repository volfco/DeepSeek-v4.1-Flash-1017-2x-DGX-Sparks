# Implementation plan: four ideas from basetenlabs/deepseek-ultrafast

Four changes worth building after a read of [basetenlabs/deepseek-ultrafast](https://github.com/basetenlabs/deepseek-ultrafast)
at `76766a9` (a Rust/CUDA engine for 4-8 B200/B300 GPUs on the unquantized checkpoint). Its kernels, NVLink
comms and Redis/RDMA replica sharing do not transfer to an EXL3 TP2 pair on GB10; these four structural ideas do.
**Nothing here is built or measured.** Every item is a proposal with its gates.

Order (cheapest and safest first):

| # | Item | Switch | Bits | Patch |
| --- | --- | --- | --- | --- |
| 1 | Tokenization cache | `TENSORFOLD_TOKEN_CACHE_MB` (rank 0) | identical | `0118` |
| 2 | Teacher-forced scoring | `TF_DS_SCORE=1` (rank 0) + `tools/score.py` | identical when unused | `0119` |
| 3 | Exact-position resume | `TF_DS_EXACT_RESUME=1` (both ranks) | **to be measured** (phase A), **LOSSY** (phase B) | `0120`, `0121` |
| 4 | Speculative sampling | `TF_DS_SPEC_SAMPLE=1` (both ranks) | **LOSSY** for sampled streams | `0122` |

## Ground rules (this repo's, applied to every item)

- A switch, off by default; unset means the engine runs exactly as before. A switch both ranks must agree on is
  compared at start and a mismatch refuses to run (pattern: `0051`, `0054`, `0111`).
- Host tests first (`zig build test` in a dev checkout), then a **hardware smoke** before the series is called done
  (`start.sh restart` + one chat completion on the pair; `todo.md` "HARDWARE SMOKE IS PART OF LANDING").
- A new `Model` field needs a line in `Model.open`'s init list (`m.* = undefined`; the `0114` bug class).
- Identity gate: `tools/exact.py` (record on the old image, compare on the new) and `tools/validate.py`. A switch
  that can change reply bits is **LOSSY**: opt-in, documented as such, excluded from the identity checks it breaks.
- Then a row in `to-benchmark.md`, one block per arm, 3x alternation for anything kept.
- Export with `scripts/port-patches.sh`; add the row to `zig/patches/README.md` and `CHANGELOG.md`.

## Where the line numbers come from

- Engine: `~/Workspace/tfref-base` (`TF_REF fbbc971` + `zig/patches` 0001-0116; this repo also has `0117`). Paths
  below are relative to its `zig/src/`.
- Baseten: a clone at `76766a9`; paths are relative to its root.
- These were read directly, not searched: the session that wrote this plan lost its shell (full temp volume) before
  the code mapping finished. Items marked **locate first** name a symbol whose definition was not read. Do those
  lookups before starting the item; they can change its design.

---

## 1. Tokenization cache (`0118`)

### What Baseten does
`crates/server/src/token_cache.rs`: the rendered prompt is split at added (special) tokens; each plain-text piece
of 256 bytes or more is looked up by its bytes in an LRU (256 MiB cap, half evicted by age when full); misses are
encoded, in parallel when there is more than one and they total 64 KiB or more; the ids are concatenated. Its tests
compare the result with the plain encoder.

### What this engine does
- `server/prompt.zig:50-74` (`prepare`): `renderIds` runs **twice** per request (`:58` with the generation
  prompt, `:59` without, to find `history_len`). Each call renders the chat template and encodes the whole text
  (`server/model_text.zig:74-77`: `render` then `encode`).
- `server/prepare_gate.zig:1-7` says the same ("rendered twice and tokenized twice") and exists because that work
  is large for million-token conversations.
- No cache was seen in `server/hf_text.zig:1-150` or `server/model_text.zig`. **Locate first:** `HfText`'s `encode`
  and `mask` implementations (below line 150 of `hf_text.zig`) and the encode entry in
  `core/tokenizer/tokenizer.zig`, to confirm there is none and to find the added-token split.

### Step 0: measure before building
The win is unknown. The only number available is `ttft_s - prefill_s` on prompts of 100K tokens or more in `runs/`
(mean 0.77 s over 409 requests), which includes queueing, and it does not separate the template render from the
encode. A cache helps only the encode.

1. Add two timers around `t.render` and `t.encode` in `model_text.Text.renderIds` (or in `prompt.prepare`) and
   print them on the request's `done` line after `0070`'s fields (`render_s`, `encode_s`), so old parsers still match.
   **Locate first:** the `done` line builder (`server/request_log.zig`).
2. Run a resumed 100K, 250K and 500K-token conversation, one stream, and read both figures.
3. **Stop here if `encode_s` is under ~5% of TTFT at 250K.** If `render_s` dominates, the fix is a different one
   (render the history once and derive the second render from it) and belongs in its own patch.

### Design
- **Where:** inside `HfText`'s `encode`, after the added-token split and before the normalizer, pre-tokenizer and
  BPE of each plain piece. The key is the piece's bytes as the split hands them over.
- **`0028`'s plain mark:** `0028` writes U+FDD0 inside added-token spellings in client text before the template
  renders, and the encoder drops it before the pre-tokenizer. Key the cache on the piece **with** its marks (the
  bytes the encoder is given), so a marked and an unmarked spelling never share an entry. Confirm where the mark is
  dropped relative to the split.
- **Independence:** a piece may be cached only if its ids do not depend on its neighbours. Check the tokenizer's
  added-token flags (`lstrip`/`rstrip`, `single_word`, `normalized`) and whether the pre-tokenizer sees text across
  a special token. If any piece is context-dependent, cache only pieces bounded on both sides by tokens with no
  strip flags, and encode the rest plainly.
- **Second render for free:** the two renders of `prepare` differ only in the generation suffix, so with the cache
  on, the second encode should hit on every long piece even when the cache was cold.
- **Size:** `TENSORFOLD_TOKEN_CACHE_MB` (unset or `0`: off, the encoder as before). Suggested first value 32, not
  Baseten's 256: the head runs with a 2.5 GiB memguard floor. Entry cost is `len(text) + 4 * ids`. Skip entries
  larger than a quarter of the cap. Evict the older half by last use when over the cap.
- **Concurrency:** one mutex around the map; hold it for lookups and inserts only, never during an encode. Prepares
  run concurrently (`prepare_gate.zig`).
- **Parallel encode of misses:** leave out of the first patch. It adds threads on a head that has few cores to
  spare; add it only if step 0 shows cold first turns matter.
- **Images:** placeholders are expanded after encoding (`prompt.zig:78-117`, `spans`), so the cache is upstream of
  them and needs no special case.

### Host tests
- For a corpus of texts (plain prose, text with every added token, text with `0028` marks, a piece of exactly 255,
  256 and 257 bytes, empty pieces, adjacent specials, non-UTF-8 edge bytes): cached encode equals plain encode, on
  a cold cache and on a warm one.
- Eviction: fill past the cap, check the bound holds and results stay equal.
- Two threads encoding overlapping texts give the plain result.
- Switch off: no allocation, no map, the same ids.

### On the pair
- Smoke, then `tools/exact.py` IDENTICAL and `tools/validate.py` against the image without it.
- `to-benchmark.md` row: `encode_s` and TTFT on resumed 100K / 250K / 500K conversations, cache off against on;
  `MemAvailable` low on the head.

---

## 2. Teacher-forced scoring (`0119`, `tools/score.py`)

### Why
`todo.md` says "the quality set is undefined in this repo", the README says the pack's card has no perplexity or
KL figure, and the lossy switches (`0053` expert prune and cap) and EAQ v1 against v1.1 have no numeric gate. No
scoring harness exists in `tools/`, and the server refuses `logprobs` (`server/fields.zig:276-284`).

### What Baseten does
`crates/engine/src/bin/run.rs:253-305` (`score`): prefill the prompt, then feed a recorded continuation 8 tokens at
a time through the normal decode forward, read every row's full logits, and report mean NLL with its standard
error, perplexity, and how often the argmax equals the recorded token.

### What this engine offers
- A round already verifies host-supplied tokens: `families/dsv41/lanes.zig:1252-1264` (`roundWins`) builds a
  window from the pending token, the held drafts and `w.tokens` ("the held drafts (or the host's)", `:1187`). A
  window is at most `window_rows = 6` rows (`:48`), so 5 forced tokens a round.
- Full logits reach the host only on the `host_draw` path (`lanes.zig:1208-1226`: `m.roundLogits(n_rows)`, fp32
  `[rows, vocab]`). A greedy round returns only the device argmax (`m.roundArgmax`). A scored stream must force
  `host_draw`.
- Forced windows already exist in the lane core: `core/lanes/backend.zig:11-15` (`Window.tokens`: "drafts known on
  the host (forced, copied, tree)") and `core/lanes/windows.zig:16`, `:49-53` (`Kind.forced`, filled from the
  stream's `force` list). A score stream can ride that. **Locate first:** how acceptance treats a forced row and
  `sampling.RoundRules.path` (what selects `host_draw`).

### Design
- **Request:** a new non-standard field on `/v1/completions` only, e.g.
  `"tensorfold_score": [id, id, ...]` with `prompt` as token ids. Refused unless `TF_DS_SCORE=1`, `n == 1`,
  `stream` false, `temperature 0`.
- **Engine:** a stream with a score list (a) drafts nothing from the drafter, (b) offers the next up-to-5 recorded
  ids as its window's `w.tokens`, (c) forces the `host_draw` path for the round, (d) accepts every row whatever the
  target picked, and (e) records for each row `logit[recorded] - logsumexp(row)` in f64 and the row's argmax.
  The first recorded token is scored from the prompt pass's head (`prompt.zig:2253-2267`, last row's logits).
- **Reply:** in the existing `tensorfold` block: `score: {tokens, nll_sum, nll_mean, nll_se, ppl, argmax_match}`
  and optionally the per-token NLLs. No change to `usage` or `choices`.
- **Rank 1:** nothing new on the wire if the forced tokens ride the existing `verify` frame as window ids (they
  already do, `writeWin`). Confirm no frame carries the "accept all" decision; it should be rank 0's alone.
- **Other streams:** forcing `host_draw` for a round changes how every stream in it gets its tokens (logits copied
  to the host instead of a device argmax). That should not change a greedy stream's token, but it is the thing to
  prove, and forced 6-row windows are the same shape class as the depth pins that already break
  `concurrent == solo` (`todo.md`): run scoring alone at first (`PARALLEL=1` or refuse a score request while others are live), and lift that
  only after `validate.py`'s concurrent check passes with a scoring stream beside it.

### `tools/score.py`
- Mirrors `tools/exact.py`'s client and record format. Input: a JSONL file of `{id, prompt_ids, continuation_ids}`.
  Output: per item and overall `nll_mean ± se`, `ppl`, `argmax_match`; `--compare A.json B.json` gives the paired
  difference per item with a standard error, since items are the same across arms.
- **Where references come from** (there is no unquantized model on the pair):
  - *Self-consistency set:* record greedy replies of the base configuration on natural prompts (the `bench.py`
    prose and code prompts, 40-100 items of 256-512 tokens). Score them under each arm. A lossless arm must give
    the same NLL to the last bit and `argmax_match` 100%; a lossy arm's delta is its cost. This needs nothing external.
  - *Reference set:* continuations generated by the unquantized model elsewhere (DeepSeek's API or any host that
    runs it), tokenized with this tokenizer. This is what gives a number comparable across packs.
  - *Human text:* held-out natural text split into prompt and continuation, for a plain perplexity.
- Store the sets under `tools/score_sets/` with their provenance; do not generate them at run time.

### Host tests
- NLL arithmetic against a hand-computed toy row; f64 logsumexp stable for large logits.
- A score stream's windows: 5 forced ids a round, a short last round, a continuation of length 1.
- Switch off: the field is refused and nothing else changes.

### On the pair
- Smoke; `exact.py` IDENTICAL with the switch on and no score requests.
- **Self-check:** scoring the base arm's own greedy replies gives `argmax_match` 100%. If it does not, forced
  rounds do not reproduce decode rounds, and the tool is wrong before any arm is judged.
- Repeatability: the same set twice gives the same NLLs bit for bit.
- First use: the `0053` rows and EAQ v1 against v1.1 in `to-benchmark.md`'s "quality set" column.

---

## 3. Exact-position resume (`0120` phase A, `0121` phase B)

### What it costs today
Kept states exist only at 2048-row chunk ends at or before `len - 128`:
- keep hook: `lanes.zig:916` (`f.start <= f.replay and (f.start == f.last_end or markEnds(...))`),
  `last_end = min(replay, len - 1) / chunk_rows * chunk_rows` (`:1165`);
- `snapSaveFn` refuses anything else (`:1467`, `error.NotAChunkEnd`);
- `usable` refuses a state unless `at % chunk_rows == 0 and at <= len - window` (`:1372`).

So a turn recomputes its new input, the previous reply, 128 tokens and up to 2,047 tokens of grid slack. Across
the 629 resumed requests in `runs/` the mean recompute was about 2,660 tokens (bench traffic; not split into its
three parts). `docs/prefix-cache-hitrate-review.md` finding 18 calls this ceiling identity-bound and says not to
fix it.

### What Baseten does
Prefix state is encoder-side only, and "decoder replay KV is never prefix state" (`docs/SERVER.md`).
- A ring of the encoder's outputs for the last 128 rows (`EncoderTail`, `crates/engine/src/model.rs:605-623`,
  written by `stash_tail`, `:2213`).
- A turn that appends resumes at the exact token: the encoder runs over the new rows only, and the decoder is
  replayed over the last 128 rows from the ring plus the new rows (`replay_decoder`, `:2238`).
- If the ring does not cover the window, the encoder is re-run read-only over at most 128 rows to rebuild it
  (`:2316-2328`).
- After a reply, its tokens stay as prefix state (`finish_decode_cache_with_stats`, `:4089`).

It makes no claim that a resumed reply equals a cold one bit for bit, and it has no cross-conversation sharing.

### What finding 18 gets right and wrong
Its stated reason ("a deeper keep would store decoder-layer state that a fresh pass does not compute") does not
hold: no decoder state needs storing. The real questions are these, and they decide whether phase A is bit-safe:
1. **Chunk shapes.** An off-grid resume runs chunks of other sizes than a cold pass. `0026` and `0051` both record
   that row count selects kernels and rounding (grouped EXL3 linears up to `gemm_rows`, the prefill GEMM above;
   `prompt.zig:940`), and a chunk of `ring_extra` rows or fewer reads its window from the ring instead of the
   gather path (`ring_mode`, `prompt.zig:936-937`, with `replay_floor`).
2. **Kit variants.** The kit's kernels were captured for certain shapes and alignments; `0038` exists because an odd
   block had no variant (`MissingTritonVariant`), and `replayCut` already copies positions to a 16-byte boundary
   for odd cuts (`prompt.zig:2222-2230`). Only a run on the pair finds a missing variant.
3. **Ratio-2 layers.** Layers 2-19 compress two tokens to a key (`compress_ratios`, `config.zig:287`). A chunk that
   starts at an odd position starts mid-pair, and the engine refuses it today: `kvSourceUpdate` returns
   `NotPortedYet` when `start % ratio != 0` (`prompt.zig:1167`, "the group's earlier rows from the positional
   store"). So phase A resumes at **even positions only** (keep at `len` rounded down to even), or ports that case.

### Step 0: the identity experiment (no engine change beyond a debug switch)
Decide bit-safety by measurement before designing around it.
0. **Check the end state is ever a prefix.** `len` includes the generation suffix, and the template may render a
   past assistant turn differently from how it was generated (think tags, `0057`, `clear_thinking` edits). Offline,
   with real harness traffic: for consecutive turns, is turn N's prompt ids (and turn N's prompt + reply ids) a
   prefix of turn N+1's prompt ids? `prepare` already computes `history_len` (`server/prompt.zig:58-62`). If the
   prompt end is not a prefix, the state to keep is at the common-prefix point (likely `history_len`), and the
   payoff shrinks accordingly. This gates the whole item.
1. Add a debug-only way to make a **cold** pass split one grid cell: `TF_DS_DEBUG_SPLIT=p` cuts the chunk that
   contains position `p` in two at `p` (`[grid, p)` then `[p, next grid)`), every other chunk as today. This is
   the shape an off-grid resume produces (see "Resume" below) and needs no Snap changes.
2. With `--logits-sha 1` (`lanes.logits_sha`, `lanes.zig:54`), compare the prompt logits' sha of the same prompts
   unsplit (today) against splits that leave a second part of 1, 16, 17, 64, 65, 128, 129, 1000 and 2047 rows (even `p` only: an odd start is refused, see 3 below), and against prompts whose length leaves a last chunk of 1, 16, 17, 128, 129 rows. Use prompts of
   3K, 10K and 70K tokens.
3. Outcomes:
   - all shas equal: phase A can be specified as bit-identical and gated by `exact.py`;
   - equal only for some shapes (for example first chunk > `gemm_rows`, or even starts): restrict phase A's resume
     points to those shapes and keep it bit-identical;
   - a `MissingTritonVariant` or other refusal: that shape is unusable until the kit is recaptured; exclude it;
   - shas differ broadly: phase A is LOSSY-class (warm differs from cold), opt-in only, and probably not worth it.

### Phase A (`0120`): keep at the prompt's end, resume at the exact token
State computed by the prompt pass only; no decode-written rows.

**Stash.**
- At `replayCut` (`prompt.zig:2205-2232`), after `kvSourceUpdate` and before the decoder layers run, copy the
  rows at or past `replay` of `ch.h` (bf16, `hc * hidden * 2` bytes a row, `:2214`) and `ch.pre` (`hc * 4` bytes
  a row, `:2217`) into a per-slot tail buffer: 128 rows, about 5.2 MB of `h` and 2 KB of `pre` a slot and rank at
  `hc 4`, `hidden 5120`.
- A prompt's last 128 rows can span two chunks; append across chunks and record `[tail_start, tail_end)`.
- The rows a later turn needs always lie inside this cut: a longer prompt's replay starts later, never earlier.

**Keep.**
- After the last chunk and before the first round (`lanes.zig:927-934`), take one more Snap at `at = len` when the
  switch is on: the slot's own part (`snapCopy`) plus the tail buffer and its range. `Snap` (`lanes.zig:446-459`)
  gains `tail: ?cuda.DeviceBuffer`, `tail_start`. `snapBytesFn` (`:1457`) reports the larger size for such a state.
- `Model.snapBytes` / `snapCopy` (`model.zig:809-835`): a Snap is one slot's window ring of **every** layer (40) and the
  positional stores of every ratio-2 KV source; no drafter rings. Still to check: whether the encoder layers' rings are valid at `len` after the
  decoder ran. If the decoder's layers share a ring buffer with state the encoder needs, snapshot before the cut.
- The prompt cache must be told a state exists at `len`. **Locate first:** `Store.marks`, `min_gap`, `min_prompt`
  in `core/prompt_cache.zig` and how `reuse.marks` reaches `lanes.zig:916`; add an end mark there.

**Resume.**
- `usable` (`lanes.zig:1369-1373`) and `snapSaveFn` (`:1467`) accept, with the switch on, an off-grid `at` that
  carries a tail, when `at <= len` and `tail_start <= len - window` (the tail covers the new replay's rows before
  `at`).
- In the fill setup (`lanes.zig:1144-1166`): `start = sn.at` as today.
- **The first chunk after an off-grid resume runs only to the next 2048 boundary**, then grid chunks. The chunk
  loop today takes `n = min(chunk_rows, len - start)` from wherever it stands (`lanes.zig:899`, `:915`), so
  without this no later chunk end would land on the grid, and the keep hook (`:916`) and `snapSaveFn` (`:1467`)
  would never fire again for that pass: shared cuts, edit marks and spill keeps would silently stop. Change `n` to
  `min(chunk_rows - start % chunk_rows, len - start)`.
- In `replayCut` for a resumed pass: when `replay < ch.start` (the window reaches before the resume point),
  prepend the tail's rows `[replay, ch.start)` to the cut rows of `h` and `pre`, and to the positions, so the
  decoder layers run on `[replay, chunk end)`. Today `first = max(ch.start, replay) - ch.start` cannot be negative;
  this is the new case.
- The decoder's window floor stays `replay` (`rs.floor = replay`, `model.zig:1420`).
- Re-stash the new tail for the next turn.

**Wire and ranks.**
- No new frame kind is needed if the tail is saved and restored inside the existing `snap_save` / `snap_restore`
  handling on each rank; the existing `fill_chunk` frame already carries `start`, `n`, `replay`
  (`lanes.zig:902-906`).
- Both ranks must agree on the switch. **Locate first:** the hello/feature compare `0111` used for
  `TF_DS_EDIT_MARK`, and add the switch to it.

**Refuse at first.** Start with the switch refused together with `SPILL_GIB` (the block store is 2048-token
blocks; `spill_state.zig:3` requires `at % 2048 == 0`), `KV_SHARD=1`, `TF_DS_PREFILL_SLICE` and image prompts.
Off-grid states stay in memory only. Lift each after its own test.

**Grid states stay.** Keep the existing grid keeps as they are. The end state is one extra Snap a conversation;
it supersedes the conversation's previous end state. Shared cuts and edit marks keep using grid states.

### Phase B (`0121`): keep the reply's rows (LOSSY by construction)
After a reply ends, its tokens' encoder-side rows were written by decode rounds, not by a prompt pass: different
row counts and kernels, so a turn resumed there is not what a cold pass computes. This can never pass the
cold-equals-warm check. It is worth building only if phase A lands and the previous-reply share of the recompute
is large (split the 2,660 tokens first: `prompt - cached` against the previous turn's `tokens`).
- Needs a tap in `round.zig` that stashes each accepted row's `h`/`pre` at the decoder's first layer into the same
  128-row tail, and a Snap at stream end.
- **Locate first:** the stream end path in `lanes.zig` (no keep hook after decode was seen), and the buffer that
  holds the streams at the decoder's first layer during a round.
- Its own switch value (`TF_DS_EXACT_RESUME=reply`), its own `to-benchmark.md` row, and a quality gate from item 2
  (score the same continuation on a cold and on a reply-resumed history).

### Host tests (phase A)
- `usable` / `snapSaveFn`: off-grid `at` with a covering tail accepted with the switch on, refused off; a tail that
  does not cover the new replay refused; grid states unchanged in both.
- Tail bookkeeping: prompts whose last 128 rows span two chunks; `len < 128`; `len` an exact multiple of 2048;
  a delta of 1, 127, 128, 129, 2048, 2049 tokens.
- `replayCut` with a prepended tail on the toy host: the decoder's input rows equal a cold pass's rows for the
  same positions.
- Model init list: the new fields are initialised (`Model.open`).
- Switch off: the feature string and every frame's bytes are unchanged.

### On the pair
- Smoke, both ranks.
- A new `tools/resume_exact.py` (or a mode of `tools/prompt_reuse.py` / `tools/resume_bench.py`): a conversation
  of 6-10 turns with deltas of 1, 50, 300, 2,000 and 5,000 tokens; for each turn record `cached`, `prefill_s`,
  `ttft_s` and the reply's token sha, with the switch off and on, and a **cold** run of every turn's full prompt.
- Gate, if step 0 said bit-identical: every warm reply sha equals its cold sha; `exact.py` IDENTICAL including
  its cached-resume prompts; `validate.py` passes.
- `to-benchmark.md` rows: recomputed tokens a turn, `prefill_s`, TTFT, at 8K / 70K / 250K histories, one and four
  streams; Snap memory (`prompt_cache_held_bytes`) and `MemAvailable` low.
- Expected size of the win: the grid slack (0 to 2,047 tokens, ~1,000 on average) and the 128-token overlap a
  turn; nothing for a cold prompt. In seconds it is unknown: the resumed requests in `runs/` that recomputed under
  4,500 tokens (mean ~2,100) show a mean `prefill_s` of 3.7 s, far above the solo prefill rate, because they ran
  beside other streams. Measure it solo and at four streams.

---

## 4. Speculative sampling for sampled streams (`0122`)

### Today
- The target draws each row with a Gumbel key of (seed, position, token id) (`families/dsv41/sampling.zig:1-9`,
  `keyBits` `:40`), so "the drafted reply is the serial one" (`lanes.zig:14-15`).
- The drafter is greedy by default, so a draft lands on a sampled stream only when the target's keyed draw happens
  to equal the drafter's argmax. `0054` (`TF_DS_T26_KEYED_DRAFT`) adds the same keyed noise to the drafter's logits
  before each Markov step so the two picks agree more often, and keeps the invariant.
- README: prose at temperature 0.7 with thinking runs at 63.9 tok/s against 80.7 at temperature 0.

### What Baseten does
Distribution-exact sampling that does not reproduce the plain reply:
- the drafter samples its drafts from its own distribution `q` with Gumbel noise under a `RNG_DRAFT` key
  (`crates/kernels/cu/draft_markov.cu:176-189`), and stores its logits (`vq`) and their log-sum-exp;
- verification is **block verification**, not token by token: a running joint weight `w` (the product of `p/q`
  along the block, capped at 1), an accept test per position under a separate `RNG_ACCEPT` key, and on rejection a
  draw from the residual `max(w*p - q, 0)` (`crates/kernels/cu/sample.cu:697-812`);
- `run --samp-test N` checks the first token's distribution against exact sampling with a chi-square test
  (`crates/engine/src/bin/run.rs:204-250`).

### Design (start with the textbook token-level rule)
For each drafted position `i` with draft `x` sampled from `q_i`:
accept with probability `min(1, p_i(x) / q_i(x))`; on the first rejection, draw the token from
`max(p_i - q_i, 0)` normalised and drop the later drafts; if all are accepted, draw the next token from `p`
as today. `p` is the target's distribution **after** temperature, `top_k`, `top_p` and `min_p`; `q` is the
drafter's at the same temperature. Block verification is a later refinement once this is correct.

What it needs:
1. **Drafts sampled from `q`.** `0054`'s kernel already adds `T * gumbel` to the drafter's logits per step
   (`tf_ds_keyed_noise_kernel`). Reuse it with an **independent** key (a draft tag mixed into the seed), not the
   sampler's key: with the shared key the drafts are correlated with the target's draw and the accept rule is no
   longer exact. So `TF_DS_SPEC_SAMPLE` and `TF_DS_T26_KEYED_DRAFT` are mutually exclusive; refuse both at start.
2. **`q_i(x)` at verify time.** Today only the drafts and confidences return to the host (`draft.zig`, "one host
   read: the drafts and the confidences"), and the drafter's logits are rank-local columns on the device. This is
   the main cost. Options, cheapest first:
   - return, per draft step, the drafter's log-probability of the chosen token: needs the full-vocabulary
     log-sum-exp across both ranks (one extra reduction and gather a step) plus the chosen logit. Enough for the
     accept test.
   - for the residual draw on a rejection, the host needs `q_i` over the target's candidate set only (`p` is
     non-zero on at most `top_k + margin` ids, `sampling.zig:21-24`): gather the drafter's logits at those ids for
     the rejected position (a second small read, on rejection only), or keep the step's logits on the device and
     gather on demand.
   - **Locate first:** where the Markov loop's per-step logits live (`draft.zig` `Drafter.local`, `mk.rowCols()`),
     whether they survive until the verify, and `tri_markov.zig`'s per-step gather.
3. **Accept and residual on the host**, in `verifyFn`'s `host_draw` branch (`lanes.zig:1240-1247`), in f64 as
   `sampling.zig` computes scores. Accept uniform and residual Gumbel from two more independent keys
   (accept tag, residual tag; position as today). The device draw (`0055`) stays off for such rounds.
4. **The round loop must take a token the target did not "draw".** Today `o.sampled[r]` is the target's own pick
   and the core compares it with the draft. Report the draft itself as the sampled token on accept, and the
   residual draw at the first rejection. **Locate first:** how the core lane loop compares `sampled` with `drafts`.
5. **Draft depth.** The live depth rule is the lane core's `windows.allocate` over the confidences' prefix survivals
   (`core/lanes/windows.zig:164-205`, `core/lanes/depth.zig`; `draft.zig`'s `chooseK` is only called by the offline
   `layer_main.zig`). That survival estimate was trained for
   greedy agreement; under this rule a draft's chance of landing is `sum(min(p, q))`. Keep the rule as is at first
   and record acceptance; recalibrating is a follow-up (`0050`'s mechanism can do it).
6. **Wire.** Rank 1 must sample the same drafts: the seed and temperature already travel in `pass_keyed`
   (kind 14, `lanes.zig:67`, `:1321-1331`). Reuse that frame with the draft tag; no new kind if the tag is derived
   on both ranks from the switch.

### What it breaks, on purpose
- A sampled stream's drafted reply no longer equals its undrafted reply for the same seed. It has the same
  distribution. Greedy streams are untouched (temperature 0 takes the old path).
- So with the switch on: exclude sampled prompts from `validate.py`'s drafts-on-equals-drafts-off and
  concurrent-equals-solo checks (**locate first:** which checks those are), and say so in the patch's README row.
- A second invariant goes too: a sampled reply is no longer reproducible from its seed alone. Draft depth depends
  on load, so which positions are drafts (accept test, residual draw) and which is the round's plain row changes
  with the round's composition, whatever the keys. A sampled reply repeats only for a fixed seed and the same
  rounds. State this in the README row; clients that rely on `seed` must leave the switch off.

### Host tests
- The accept/residual rule on toy `p`, `q` (including `q(x) = 0`, `p(x) = 0`, `p == q`, disjoint supports): over
  200K seeds the output histogram matches `p` (chi-square, p > 0.01), for one position and for a 3-draft block.
- Key independence: draft, accept and residual keys differ for the same (seed, position, id).
- Both switches set: refused at start. Switch off: frames and replies unchanged.

### On the pair
- Smoke. `exact.py` IDENTICAL for greedy prompts with the switch on.
- **Distribution gate** (`tools/samp_test.py`): one prompt, `max_tokens 1` and then `max_tokens 3`, 5,000 seeds,
  with `DRAFTS=0` and with the switch on; chi-square between the two token histograms (and the second and third
  tokens conditioned on the first). This is Baseten's `--samp-test`, done through the API.
- Quality: item 2's scoring cannot judge a sampled arm directly; use the distribution gate.
- `to-benchmark.md` rows: acceptance, tokens a round, `s_per_token`, decode tok/s for prose and code at temperature
  0.7 and 1.0, one and four streams, against base and against `TF_DS_T26_KEYED_DRAFT=1` (which keeps the invariant
  and is the arm to beat).
- **Drop it if it does not beat `0054`.** `0054` costs no invariant; this one does.

---

## One more benchmark row (no code)

Baseten reports 5.6 tokens a step on one prompt ("implement the tetris game. output HTML code directly - no
explanation.", temperature 1, thinking off). This repo's matrix shows 2.0-2.35 tokens a round on prose and agent
traffic. The two are not comparable as they stand. Add that prompt as a `to-benchmark.md` row (greedy, and
temperature 1 with and without `0054`) to get a like-for-like figure before reading anything into the gap.

---

## To explore: how Baseten orders and chains a decode step

Not planned work. Three structural differences in how Baseten issues a decode step, recorded as things to
profile before deciding anything. Only the decode round was compared; prefill kernel ordering was not.

### Step 0 for all three: rank 0's round profile
The only number so far is rank 1's `--profile` line from the overnight run
(`runs/overnight/i01.rank1.profile.log`): `enqueue_ms` 14.32 against `forward_ms` 54.55 a round over 1,620 rounds.
It is attribution only (the profile adds a sync a round, and the block included 8-stream rounds). `enqueue` is the
host time to issue the whole `round.forward` (`model.zig:1504-1507`), which includes waiting for and landing the
Engram rows between stretches (`round.zig:1537-1551`); `forward` contains it. So it is **not** launch overhead:
it points at the Engram host step (item E and the second pass below) more than at items A and C.

- Run one block with rank 0's `--profile` serve argument and nothing else changed (`runs/overnight/RESULTS.md`
  notes it was never run: it is a CLI argument, not a recipe setting). Its line (`families/dsv41/lanes_main.zig:425`)
  gives `step_ms`, `send_ms`, `enqueue_ms`, `forward_ms`, `absorb_ms`, `logits_ms`, `sample_ms`, `pass_ms` a round.
- If `send + enqueue + logits + sample + pass` is a small share of `step_ms`, drop items A and C. If it is large,
  item A is the biggest decode lever found in this review.

### A. A whole step as one graph, with no host in the loop
- **Baseten:** draft, verify, accept and the next step's setup are one captured CUDA graph (`capture_spec_blk`,
  `crates/engine/src/model.rs:3956`; the accept is a kernel, `spec_accept`, `crates/kernels/cu/ops.cu:591`). The
  server launches steps ahead and reads the committed tokens from a snapshot two steps back
  (`crates/server/src/engine.rs:320-345`). With the confidence cut, the graph picks its verify depth itself through
  a conditional node (`capture_spec_vadapt`, `model.rs:4004`).
- **Here:** a round is two primitives, `verifyFn` then `draftFn` (`families/dsv41/lanes.zig:1189`, `:1279`), each
  preceded by a frame to rank 1 and followed by a host read (the rows' tokens or logits; the drafts and
  confidences). The window allocator, the sampler (unless `0055`) and the depth rule run on the host between them.
- **To explore:** how much of a round is that host and link time (step 0); whether the draft pass can be issued
  behind the verify without waiting for the host (the drafter needs the accepted token, which the host decides
  today); whether rank 1 can be sent both primitives in one frame. Multi-stream rounds and the cost-priced depth
  rule make a full port much harder here than in Baseten's one-stream engine.
- **Bits:** issuing order only, if the same launches run with the same inputs. Gate with `exact.py`.

### B. More of a layer on parallel streams
- **Baseten:** the compressor and KV-source update run on a second stream beside the query projection
  (`par_fork` / `attn_side`, `model.rs:2659-2662`), the indexer's weights on a third on some targets (`ixs`,
  `:2667-2673`), the DSpark tap's mean on the side stream (`:2622-2632`), and the drafter's input projection on
  the second stream while the main stream runs the verify head (`:2531-2545`).
- **Here:** the round's side streams carry the mixes' Sinkhorn half and the L2 prefetches only
  (`families/dsv41/round.zig:8-9`, `:200-207`); the compressor and the indexer run in the main stream's order.
- **To explore:** whether small-row decode kernels leave the GB10 partly idle (a kernel trace shows it); if so,
  fork the compressor/indexer work as `useSide` forks the mixes'. The graph capture already spans every stream of
  a stretch.
- **Bits:** a reorder of independent launches; arithmetic unchanged. Gate with `exact.py`.

### C. Programmatic dependent launch between kernels
- **Baseten:** almost every kernel begins with `griddepcontrol.wait` / `launch_dependents`
  (`crates/kernels/cu/common.cuh:14-15`), so a kernel's launch overlaps the tail of the one before it.
- **Here:** only the RDMA gather does (`TF_RDMA_PDL`, `zig/kernels/cuda/dsv41/rdma_gather.cu`). The kit's recorded
  Triton kernels are precompiled, so they cannot take it without a recapture; the hand-written `.cu` kernels could.
- **To explore:** the gap between consecutive kernels inside a replayed graph on GB10, from an nsys export.
  Baseten's `tools/nsys_breakdown.py` does this per step (it handles graph-launched kernels) and needs this
  engine's kernel names in its groups; that nsys works on GB10 was not checked. Not sized.

### D. Launches a sublayer: where Baseten fuses and this round does not
Read side by side: `families/dsv41/round.zig` (`stretch` `:1562`, `attention` `:827`, `moe` `:1298`) against
Baseten's `layer` (`crates/engine/src/model.rs:2606`), `attn_side` (`:3316`) and `ffn` (`:3497`). Launch counts
below are read from the code, not traced. **Most of this round's kernels (`tri_*`) are the kit's recorded Triton
kernels: fusing them means recapturing the kit.** Only the `e.ops.*` and `e.exact.*` launches
(`zig/kernels/cuda/dsv41_ops.cu`) can change in a patch.

| Part of a layer | Here (launches) | Baseten | Patchable here? |
| --- | --- | --- | --- |
| Compressor of a ratio-2 KV source (4 layers) | after the grouped linears: `scatterRows` x2, a memset, `gatherRows` x2, `compress2`, `rmsnorm`; then the index key (`grouped`, `rmsnorm`, `ropeHeads`, `fp4Store`) and the latent (`ropeHeads`, `fp4Store`): ~13 (`round.zig:935-960`) | the linear writes its rows into the history at their position (`out_pos`), `compress_pool` pools and normalizes, the key's linear takes its norm as a prenorm, one `rope_pair` rotates, quantizes and stores both: 4 (`model.rs:3329-3368`) | the scatter/gather/`compress2` chain is this repo's own CUDA (6 launches to 1, 4 layers); the rest is the kit's |
| Indexer query of a compressed layer (~38 layers) | `ropeHeads`, `fp4QdP2`, `rowmmWts`, `indexKeys`, `topkSelect`: 5 (`round.zig:972-1000`) | the query's RoPE and FP4 quantization ride `wq_b`'s epilogue (`iq`, `model.rs:2693-2697`), `idx_weights` is one GEMV with its scale folded in (`:3467`), scoring and top-k are one or two launches | no (kit) |
| MoE | `rowmmGate`, `route`, the experts' decode (`round.zig:1303-1314`) | gate, route, both expert GEMVs and the FFN mix's dots in one launch (`moe_fused` with `GateFused` / `DotsFused`, `model.rs:3573-3588`) | no (kit + EXL3 experts) |
| After a gather | the post is fused into the next mix (`hcPre2`, `round.zig:1650-1662`) | the post is fused into the all-reduce kernel itself (`CM_SUM_HCPOST`, `model.rs:2982`, `:3606`) | equivalent in launches |
| Head (once a round) | `collapseNorm`, the head, the gather, `copyRows` a rank (`round.zig:1630-1640`), then the argmax or draw | the norm is the head GEMV's prenorm and the sampler's statistics are recorded inside it (`samp_rec`, `model.rs:2556`) | the `copyRows` pair only |
| Engram layer | `grouped`, gather, `add2Bf16`, `engramGate` | gather, all-reduce, GEMV, gather, apply | about equal |

- **To explore:** the engine already counts this. `tf-dsv41-lanes --profile 2` (`PhaseTimer`, `round.zig:1341`)
  gives a round's GPU time by phase (`mix_attn`, `attn` with `a_in`/`a_q`/`a_idx`/`a_core`/`a_woa`, `gather_a`,
  `mix_ffn`, `moe` with `m_rt`/`m_gu`, `gather_m`, `engram`, `head`) and the launches of each phase
  (`rd.launches`, `:682`), on eager rounds only (`GRAPHS=0`). Run it at 1 and 6 rows, at a short and a 250K
  context. It answers: how many launches a round is, which phase owns the time, and whether `a_idx` (the indexer)
  or the gathers dominate. Nothing in D is worth doing if the gathers and the experts own the round.
- **Smallest patchable candidate:** one kernel for the ratio-2 compressor's scatter, gather and pool
  (`e.ops.scatterRows`/`gatherRows` + `e.exact.compress2`), keeping `compress2`'s operation order so the bytes stay
  equal (`exact.py`). It removes ~20 launches a round; expect a gain only if launch overhead shows up in the phase
  profile.
- **Kit-side candidates** (for whoever recaptures the kit, not for a patch here): the indexer query's RoPE and FP4
  quantization in `wq_b`'s epilogue; `rmsnorm` as a prenorm of the linear behind it; score and top-k in one launch
  for short buckets.

### E. Things Baseten's layout shows that are not launch counts
- **Three streams' work is forked at fixed points, not opportunistically.** The side work is always forked right
  after the mix that produces its input and joined right before the first reader (`hc_fork_dots` / `hc_join`,
  `par_fork` / `par_join`, `model.rs:1792-1899`). This round does the same for the Sinkhorn half; item B extends
  the pattern, it does not need a new mechanism.
- **Engram is forked at the start of the forward for every Engram layer** (`engram_fork`, `model.rs:3238`) and
  joined before the layer that reads it. Here the reads go out after the first stretch is issued and land on the
  host before each Engram layer's stretch (`engramAhead` / `engramLand`, `round.zig:1528-1546`), which is why a
  round is cut into stretches (one graph each) at the Engram layers. Baseten's tables are in GPU memory, so its
  step needs no host landing and stays one graph. **To explore:** whether a round's Engram rows could be staged to
  the device without a host step between stretches (for example a pinned buffer the next stretch's first kernel
  reads once an event fires), so a round is one graph. This is the same question as item A seen from the Engram side.
- **The drafter's three blocks share one grouped KV projection** (`mtp_wkv`, `model.rs:3695-3702`) and the draft
  head is a cheaper copy of the head (`head_fp8`) with the verify using the exact one. The first is a grouping
  this engine's `grouped` could express if the drafter does not already (not checked in `draft.zig`); the second
  is a weight-format choice that does not apply to an EXL3 head.

### Already equivalent, and not portable
- Equivalent here: q's RoPE in `wq_b`'s epilogue and `wq_b` grouped with the indexer's `wq_b` in one launch
  (`round.zig:883-890`); the mix's post fused into the next sublayer with the Sinkhorn half deferred; device argmax
  and `0055`'s device draw; the prefill indexer split across ranks (`0001`).
- Not portable: Baseten's tensor-core GEMV, cluster and shared-memory top-k kernels (written for
  `sm_100a`/`sm_103a` and FP8/BF16 weights; the linears here are EXL3 trellis kernels), and its comm kernels, which
  write into peer GPUs' memory over NVLink (the ranks here are separate machines).

---

## Second pass: the two engines compared subsystem by subsystem

A deeper read of both codebases, one subsystem at a time (decode step and drafting; attention, indexer and KV
layout; prefill; TP exchanges, MoE and Engram; server, cache and loading). **Nothing was run on the pair.**

**Verdict.** There is no large algorithmic gap. Both engines do the same work per row: bounded replay, the same
TP split, two full-width exchanges a layer, per-expert grouping in the MoE, the same drafter inputs and Markov
chain, positional overwrite after a rejection. Baseten's speed is hardware, in-box GPU-to-GPU collectives and one
request at a time. This engine is **ahead** on memory (about 890 B a token a rank against roughly 60 KB in
Baseten's device layout), on pool-only index scoring in decode, on cut-ins and interleaving, and on cache depth
(grid states, shared cuts, edit marks against one snapshot plus one checkpoint a session). What is left are
host-path, exchange and bookkeeping items, most of them small and unmeasured.

How to read the tables:
- **Read:** *checked* = the local line was read while writing this; *sub-read* = reported by a per-subsystem read
  and not re-read. Baseten-side lines are all sub-read unless an earlier section cites them.
- **Bits:** *identical* = same launches, same inputs. *sequential* = a reply keeps its bits alone, but the change
  alters round composition, and `todo.md` already records that depth pins break `concurrent == solo`; gate with
  `validate.py`'s concurrent check, not only `exact.py`. *LOSSY* = opt-in only.
- **Where:** *repo* = this repo's Zig or own CUDA, patchable. *kit* = needs the kit's recorded kernels recaptured.
- Paths: local under `zig/src/`, Baseten under `crates/`.

### Instrumentation first
None of the items below is sized. These four make them sizable, and are cheap:
1. Rank 0's `--profile` serve argument, one block (`step_ms`, `send_ms`, `enqueue_ms`, `forward_ms`, `absorb_ms`,
   `logits_ms`, `sample_ms`, `pass_ms`; `families/dsv41/lanes_main.zig:425`). Never run (`runs/overnight/RESULTS.md`).
2. A send column for the draft pass's frame: the pass timer starts after the flush (`families/dsv41/lanes.zig:1335`),
   so that frame's cost is in no column today.
3. A caller for `Ring.probes()` (`families/dsv41/rdma.zig:717-718`, no caller found): it splits a gather into
   stage, flight and copy.
4. `tf-dsv41-lanes --profile 2` on eager rounds for the phase split and launches a phase (section D). For
   prefill, only the Engram split exists (`prefill_engram_ms`) and `docs/zig-engine.md` says the phases were never
   profiled; a per-chunk phase line (attention, gathers, MoE, indexer) is the missing instrument.

### 1. Decode step and drafting

| Finding | Read | Bits | Where | Step 0 |
| --- | --- | --- | --- | --- |
| **The accept could run on the device and chain into the draft pass.** Baseten's `spec_accept` (`kernels/cu/ops.cu:591`) compares on the device and sets up the next draft. Here a round is a `verify` frame, the forward, a sync, the pick, a sync and a host read, the host's `conclude`, then a `pass` frame and the drafter pass with its own host read (`families/dsv41/lanes.zig:1189-1249`, `:1279-1349`; `model.zig:1500-1537`, `:1589-1603`). The lane core already has an early-speculation hook that dsv41 opts out of (`speculate_early = false`, `lanes.zig:596`; `r.early` is refused, `:1296`). A device prefix compare of the pick against the window ids could write the pass's inputs on both ranks, so verify, pick, accept and pass go out in one frame; when the host cuts, forces, copies or finishes, discard and draft late. Greedy and device-draw rounds only; shared rounds need core work. This is item A made concrete. | checked (local flow, the opt-out) | sequential (steps fixed before the outcome) | repo | profile columns `logits + sample`, the unmeasured pass frame, round wall minus (forward + absorb + pass) |
| **Absorb sits on the critical path.** `verify` runs the forward, then `absorbMany` of every window row's taps into the drafter rings, then a full sync, and only then is the pick launched and synced (`model.zig:1523-1537`, `:1589-1593`). Baseten runs the same work beside its head (`engine/src/model.rs:2531-2545`). Launch the pick first and wait on an event for the pick alone; the absorb then runs under the host's accept and frame send. | checked | identical | repo | the `absorb_ms` column |
| **The Markov chain always runs to the largest budget.** Each step is a cross-rank gather (`tri_markov.zig:146-155`), `steps` is the largest depth asked, the confidences come after the loop, and the host trims afterwards. Baseten stops the chain where its confidence product falls under the threshold (`kernels/cu/draft_markov.cu:131-148`). An in-graph stop needs conditional nodes agreed by both ranks (not checked on GB10); the cheap variant is the slack: a stream offers 2 drafts past its last grant (`core/lanes/depth.zig:13`). | checked (loop, slack) | sequential | repo | `pass_ms` a round; granted against steps |
| **The allocator's cost table may not match this pair.** The default is 30.0, 35.2, 39.9, 44.9, 48.6, 52.1 ms for 1-6 rows (`families/dsv41/draft.zig:697`) with 8 ms of overhead until measured (`core/lanes/depth.zig:255`); the recipe never sets `TF_DS_ROUND_MS` (`families/dsv41/native.zig:255` reads it). Only the per-row slope and the overhead change the allocator's choice, not a uniform scale. **Low prior:** the depth pins and `FIX_K` rows in `to-benchmark.md` show deeper windows losing on prose. | checked | sequential | config only | forward ms by rows from `TF_DS_FIX_K` runs, compared with the table's slope |
| **A block larger than 5.** Baseten uses 6 or 7 on some targets with lowered last thresholds (`cuda/src/profile.rs:59-65`, `engine/src/model.rs:3899-3903`). Here the window is 6 rows (`lanes.zig:48`) and the block 5. **Low prior** for the same reason (the depth-5 pin's 25.7% acceptance); it would need the row gates rerun and possibly the kit. | sub-read | sequential | kit, possibly | acceptance at depth 5 on code prompts first |

Already equivalent: drafter inputs (anchor plus noise rows, the three taps, 128-row window plus the block's own
rows), three stages, Markov rank 256 with a cross-rank best a step, the confidence head and its cumulative
sigmoid, eager absorb, no rollback, drafting straight after the prompt, longest-prefix accept plus one drawn token.

### 2. Attention, indexer and KV layout

| Finding | Read | Bits | Where | Step 0 |
| --- | --- | --- | --- | --- |
| **Indexer work is sized by a power-of-two bucket of the deepest row in the round.** `nb = rd.bucket / ratio` (`families/dsv41/round.zig:880`), the bucket from `bucketFor` (`:52-56`). A 70K-token stream launches 131K-wide indexer work, and a short stream batched with a long one pays the long one's bucket. Dead tiles skip their dots (`tri_index.zig:75`), so the cost is not doubled; what a dead tile still costs is inside the kit. Baseten sizes by the exact position (`kernels/cu/attn.cu:52-55`). Finer buckets cost more cached graphs; the kit's `_topk_sel` is specialised on powers of two. | checked (`nb`), rest sub-read | identical if the fills are reproduced | repo for the own-CUDA parts, kit for the rest | `a_idx` at ~65,000 against ~66,000 tokens; a short stream alone against batched with a long one |
| **The repo's own top-k makes 8 radix passes over every key a row** (`zig/kernels/cuda/dsv41_ops.cu:42-160`), at all 8 indexer layers. Baseten's prefill top-k samples a threshold and compacts in one pass, exact through a fallback (`kernels/cu/attn.cu:969-1010`). | sub-read | identical (exact top-k, same tie rule) | repo | `a_idx` at layers 24-36, where scoring is constant |
| **Replay scores every key at layers 24-36, decode does not.** In decode those layers score only the 2,048 pool blocks once the context is long (`round.zig:983-988`); the prompt path scores all keys and masks (`prompt.zig:1299-1309`). Affects the 128-row decoder replay only, so warm-turn TTFT at long context, not cold prefill. Not from Baseten (it scores every key everywhere). | sub-read | identical expected | kit variant for 128 rows unverified | TTFT of a short append at 250K+ |
| **Old/new key split at the KV-source layers:** Baseten scores existing keys while the compressor runs, then the new key, and merges (`engine/src/model.rs:3383-3465`). Needs a second stream and a kit launch over a key sub-range. | sub-read | identical | kit | `a_idx` minus the compressor launches |

Local is ahead: bytes a token (FP4 planes and 144-row rings against Baseten's full-history bf16), pool-only scoring
in decode, the tile-pruned select, and `idx_all` for short contexts. Nothing to port for memory.

### 3. Prefill

| Finding | Read | Bits | Where | Step 0 |
| --- | --- | --- | --- | --- |
| **Prompt gathers go through NCCL, not the RDMA ring, at full fp32 width.** The ring is for decode gathers (`families/dsv41/model.zig:110`, `:336`). An encoder chunk makes 40 gathers of rows x 5120 x fp32 (41.9 MB a rank at 2,048 rows) plus Engram's. The only measurement: rank 1's profile shows `exchange` 8.31 ms a call for the Engram gather in prefill (`runs/overnight/i01.rank1.profile.log`; per call, `lanes_main.zig:240`), over 200 calls of mixed chunk sizes. If layer gathers cost in proportion, exchanges are a double-digit share of a ~1 s chunk; that is an estimate, not a measurement. Baseten sends bf16 partials, which would be LOSSY here; the bit-safe options are a faster transport for the same bytes or overlapping a gather with work that does not need it. | checked (ring scope, profile units); payload sub-read | identical (transport), LOSSY (bf16) | repo | a prefill phase timer; NCCL throughput on the link |
| **Per-chunk constants.** Every dense linear's weights are unpacked from the trellis on every call above 128 rows (`exl3_prefill.zig:103-106`), the gate is widened a layer a chunk (`prompt.zig:1373`), ~42 collectives start, plus the frame and the read-ahead thread. Bench shape: 2K prompts run ~1,550 tok/s against ~2,000 at 8K+ (`docs/measurements.md`). Baseten's default chunk is 8,192. `chunk_rows = 2048` here is a constant, not a visible kit limit, but it is the pool block, the resume grid and part of the state fingerprint, host arrays cap a chunk at 4,096 rows, the kit was recorded at 2,048, and cuBLAS shapes can change bits. Not a free knob. | sub-read | unknown (cuBLAS shapes) | repo + kit check | ms a row of a full chunk against a 256-row chunk |
| **The indexer split runs only from 65,536 compressed keys** (`prompt.zig:1284`, `:917`); Baseten shards rows across ranks for every chunk. `TF_DS_IDX_SPLIT=1` already forces it and `docs/measurements.md` records no gain up to 128K. Recorded so nobody re-derives it. | checked | identical | exists | none needed |
| **A discarded head when the last 128 rows straddle a chunk end:** both chunks run the decoder's fixed costs and the first one's head is thrown away (`model.zig:1472`). Small. | sub-read | identical | repo | none; a few ms a request |

Baseten's `short_append` path skips no model work: it only moves appends of up to 999 tokens onto a smaller
exchange channel. Neither engine skips encoder work for rows that will be cut, and both run the 20 decoder layers
on at most 128 rows.

### 4. TP exchanges, MoE, hyper-connections, Engram

| Finding | Read | Bits | Where | Step 0 |
| --- | --- | --- | --- | --- |
| **A round's Engram reads go out after the first stretch is issued, and the first stretch is one layer.** The round is cut before each Engram layer (layers 1 and 14) so the host can land the rows: stretches are layer 0, layers 1-13, layers 14-39 plus the head, one graph each (`families/dsv41/round.zig:1528-1551`). The window ids are known before the round starts (the drafts come from the previous pass), so hashing and submitting could move off the round's critical path, to when the pass returns. **Caution:** the current order looks deliberate (the host hashes while layer 0 runs), the host buffers are shared with the prefill read-ahead (`verify` joins it first, `model.zig:1501`), and the open DEDUPE+LEAD crash is in this area. | checked | identical | repo | `enqueue_ms` against `forward_ms`; `tensorfold:engram_lookup_seconds`; `engram_read_ms` 0.67 and `engram_hash_ms` 0.98 a round on rank 1 |
| **Both Engram layers' rows could land at the first cut,** with one gather for both projections (the projection depends on the rows alone, `round.zig:1588`), dropping the cut at layer 14: two graphs a round instead of three. It puts layer 14's read latency on the critical path at layer 1, where today 13 layers hide it. | sub-read (the cuts checked) | identical | repo | the same; only with the read moved early first |
| **Greedy rounds exchange full logits.** The head's gather is each rank's vocabulary half in fp32, 258,560 B a row (`round.zig:1636-1641`), then two `copyRows`, then the pick. Baseten exchanges candidate records, never logits (`engine/src/model.rs:2603`), and the drafter here already exchanges a (value, index) best a step (`draft.zig:125-131`). An all-greedy round could exchange each rank's argmax. **Check first** that no greedy path needs the logits (the call gate `0004`, `--logits-sha`, the planned score tool). One exchange of ~83 a round, so the ceiling is the `head` phase. | checked (gather), rest sub-read | identical if "first largest, rank 0 wins ties" is kept | repo | the `head` phase, `logits_ms` |
| **Expert grid is static at rows x 7 places** (`exl3_experts.zig:390`); Baseten sizes it from the distinct experts picked. Rows that pick the same expert already share one decode here. Gain depends on whether empty places exit cheaply in the kit's kernel. | sub-read | identical expected | repo | `moe`, `m_gu` phases |
| **LOSSY: bf16 attention partials,** as Baseten rounds them before the sum (`kernels/cu/variant_zcomm_dec.cu:49`). Halves a ~120 KB payload whose cost is mostly fixed; expect little. | sub-read | LOSSY | repo | `gather_a` |

The 80 layer gathers cannot merge or move earlier: each feeds the next mix. Baseten has the same two a layer and
no low-rank or merged exchange. MoE routing, cross-row expert dedupe, the single shared-expert place and the
hyper-connection mixes (Sinkhorn on a side stream, the lagged `pre`) are equivalent.

### 5. Server, cache, loading

| Finding | Read | Bits | Where | Step 0 |
| --- | --- | --- | --- | --- |
| **Start time is one serial read.** `rank cache: 95.1 GiB of expert entries read whole in 87.8 s` of `loaded in 101.5s` (`runs/matrix/hr-base.engine.log`): one thread, O_DIRECT, ~1.08 GiB/s (`families/dsv41/weights.zig:151-157`, `:341-361`). Baseten repacks in parallel into pinned staging with async upload (`engine/src/weights.rs:431-443`). Several parallel O_DIRECT readers, or read overlapped with upload. Do **not** copy its mmap-and-populate: that fills the page cache and would trip memguard. | checked (log); code sub-read | identical | repo | the two log lines; `MemAvailable` low during load; whether the NVMe has headroom above 1.08 GiB/s |
| **A streamed reply's whole text is reprocessed every round** for stops and the thinking split (`server/chat.zig:423-428`, `server/reply_text.zig:472-487`); Baseten holds back only a suffix (`server/src/stream.rs:61-79`). Quadratic in reply length, and `splitThinking` is given the generation's allocator each round, so **check whether that is an arena that grows until the reply ends**: on a 32K-token reply that would be head memory, the one robustness item here. | checked (the calls); the allocator not traced | identical | repo | rank 0 CPU and `MemAvailable` over a 32K-token streamed reply |
| **A Snap holds all 40 layers' rings** (`families/dsv41/model.zig:809-835`, ~6.7 MB a rank) though a kept point is before the replay and the decoder's rings are rebuilt by the replay. Skipping layers 20-39 saves ~2.9 MB a kept state; needs a spill format bump. Valid only if no path reads a decoder ring row from before the replay floor (`TF_DS_REPLAY_FLOOR=1` is the served setting; not verified for `0`). | checked (bytes) | identical if that holds | repo | `prompt_cache_held_bytes`, eviction counters, `exact.py` |
| **Restores start only once a lane is free** (`core/lane_host.zig:604`, `:637`); Baseten prepares up to 6 queued jobs ahead (`server/src/request_queue.rs:88-112`). With `SPILL_GIB` on and more clients than `PARALLEL`, a queued request's disk restore could begin early. Risk: restore room can evict. | sub-read | identical | repo | `restore_wait_ns`, `ttft_s - prefill_s` |
| **Smaller ones:** the prompt-state budget is a fixed 0.5 GiB (`native/cuda.zig:453`) where Baseten sizes its cache from `MemAvailable` after load (`server/src/host_budget.rs`); an explicit `max_tokens` past the window is refused here and clamped there (`server/src/api.rs:310-316`); a stop cancels in-flight replies after 2 s here and drains there; the body limit is 32 MiB here and 64 MiB there (`server/http_body.zig:8`). | sub-read | identical (the clamp is a behaviour change: opt-in) | repo | counts of 400s, 413s, truncated replies a restart |
| **For plan item 3 phase B only:** Baseten rewrites a new prompt to the stored token spelling when the bytes match (`kv-transfer/src/align.rs:31-91`), because re-tokenised reply text can differ from the generated ids. Keeping reply rows here would need the same, and it changes prompt ids against a cold tokenisation. | sub-read | LOSSY | repo | share of turns whose ids diverge inside the previous reply |

Baseten re-renders and re-tokenises the full conversation every turn too; beyond the token cache it has no
suffix-only render. Its queue is strict FIFO with no reordering by warmth or length; this engine's cut-ins and
interleaving are ahead.

### Where to start
In order, by what they cost to learn: (1) the four instruments above; (2) the absorb reorder and the Engram
submit, both small, bit-identical and in this repo's Zig; (3) the parallel expert read at start; (4) the device
accept chained into the pass, only if the profile shows the host path is a real share of a round. Everything in
the kit's kernels waits for whoever recaptures it.
