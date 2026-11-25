# report-0270 — TF_DS_PROMPT_PICKS_ONCE (prompt attention pads its picks once per selection)

**Status:** implemented; premise holds. Commit **`6926a5eb7c37895b50294682b51947af82194caa`** (`6926a5e`) on
`kernel-0270` (`/home/colum/Workspace/tfref-k0270`, base `27baa93` = recipe-0264): exactly one commit, subject as the
brief's export format, working tree clean. Switch: **`TF_DS_PROMPT_PICKS_ONCE`** (off by default).

## What it does

`prompt.zig`'s `attention` pads the chunk's pick list with −1 to whole 16-column tiles (`cidx -> cidxp`: a
`cuMemsetD8Async` fill of `n * kp * 8` bytes plus a `copyRows`) at **every** compressed layer, although a layer whose
indexer did not run shares the previous indexer's selection (`sh.kk`, `prompt.zig:1124`, read at `:1162`): a cold
127-token prompt pads the same 63 picks six times over layers 2–7. With the switch on, a layer whose selection,
rows and pick count are still the ones the previous padded list was made from **reuses** it (no fill, no copy); any
host-known event that can change the picks re-pads:

- `indexer`'s first statement calls `ch.picks.stale()` (every arm rewrites `cidx`),
- `replayCut`'s row-rewrite block calls it (`ch.start`/`ch.n` move, even on a same-count shift),
- `prompt.begin` calls it (a new chunk),
- each pad records `held(n, kk)` and a read reuses only on an exact `(rows, kk)` match (defense-in-depth).

The engagement banner logs once at the first reuse, behind an atomic `swap(true)` guard, in the common brief's
format (`dsv41: TF_DS_PROMPT_PICKS_ONCE: the padded picks are reused across the shared selection (first at 127 rows
x 63 picks)` at the served shape).

## Files touched

- `zig/src/families/dsv41/prompt_picks_once.zig` (new, pure): `env_name`/`parse`/`fromText` (unset/`""`/`"0"` off,
  `"1"` on, anything else prints `TF_DS_PROMPT_PICKS_ONCE: expected 1 or 0` and refuses startup via the `fromText`
  error), host reference `padWidth` (whole 16-column tiles), `padList` (the pad's bytes: kk picks then −1), and the
  `Tracker` (the host-known reuse rule); 5 `std.testing` tests.
- `zig/src/families/dsv41/prompt.zig`: `Chunk.picks` field + its `init` line, the `attention` pad block's reuse arm
  + banner, the three `stale()` hooks, the banner latch. The pad condition, the fill/copy arguments, `padded` and
  all stride handling (`shardAttend`'s `kos`, `unionRenumber`'s compact `kk`) are untouched.
- `zig/src/families/dsv41/ops.zig`: `picks_once: bool` parsed at `Ops.load` via
  `prompt_picks_once.fromText(envSpan(prompt_picks_once.env_name))`. **Deviation from the switch convention:** no
  `?cuda.Function` field — there is no new kernel (the arm only drops launches), same deviation 0268 took.
- `zig/src/families/dsv41/dsv41.zig`: export + `refAll(prompt_picks_once)` so `zig build test` runs its tests.

Diff: 4 files, +171/−3.

## Launch counts at the served shape (cold 127-token prompt chunk; per rank, prefill only)

The served map (config.zig's `compress_ratios`/`index_source_layer_ids`): layers 2–19 ratio 2 (n_comp_end 63 →
kk 63 → padded to 64), layers 20–39 ratio 1 (kk 127 → 128); all 38 compressed reads pad today (both kk are not
whole tiles, and `n = 127 > tri.decode_rows`):

| Layers (selection groups) | pads before | pads after | saved fills + copies |
| --- | --- | --- | --- |
| 2–7 (the brief's group, kk 63→64) | 6 + 6 | 1 + 1 | **5 + 5** (the brief's expected saving) |
| 8–13, 14–19 (kk 63→64) | 12 + 12 | 2 + 2 | 10 + 10 |
| 20–23, 24–27, 28–31, 32–35, 36–39 (kk 127→128) | 20 + 20 | 5 + 5 | 15 + 15 |
| **whole chunk** | **38 + 38 = 76 launches** | **8 + 8 = 16** | **60 per chunk per rank** |

The brief's estimate counts only the layers 2–7 group; the same rule naturally covers every shared-selection run
(the harness measures 8/30 per chunk for 127-, 40- and 17-row chunks alike). Decode is unaffected (round.zig has
its own `rd.cidx` and never pads `cidxp`); `TF_DS_PREFILL_SLICE=5` changes nothing (state crosses slice boundaries;
decode rounds between slices touch no prompt pick state).

## Premise findings

1. **The saving exists and is host-decidable.** `weights.zig:665` sets `idx_wq_b` only on `index_source_layer_ids`
   = {2, 8, 14, 20, 24, 28, 32, 36}; every other compressed layer reads the latest selection (`prompt.zig:1124`).
   Layers 2–7 therefore share layer 2's 63 picks ((0+127)/2, `min(index_topk, n_comp_end)`) — exactly the brief's
   claim. Validity is decided from that host structure alone (which layers ran an indexer, chunk/cut boundaries),
   never from comparing device buffers. Complete writer enumeration (review 1, `h0270/review-code.md`): `cidx` is
   written only inside `indexer`, `cidxp` only by the pad, `sh.kk` only by `indexer`, `ch.n`/`ch.start` only by
   `begin`/`replayCut` — each behind a `stale()` hook.
2. **A compressed-layer boundary re-pads** because every boundary layer in the served map is an index source whose
   indexer runs (and `stale()`s) first — the contract's event list is covered without a separate boundary rule.
3. **Prefill is NOT under graph capture** (the contract's question): the only capture sites are decode rounds
   (`round.zig:1904` `cuda.graph.beginCapture` in `Graphs.run`, `draft.zig:224`); `prompt.zig`'s own comment at
   `unionOf` ("its count read on the host (a stream synchronize: a prompt chunk is never captured)") agrees. The
   host branch and the banner are capture-safe regardless; nothing new runs inside `body`/graph-observed paths.
4. **Replay-cut semantics.** The rule is "until … a replay cut **changes the rows**": `stale()` fires only in
   `replayCut`'s row-rewrite block (`first > 0 or pre > 0`). Belt-and-braces analysis (harness): even without the
   hook, bytes would match whenever `cidx` is unre-written and `(rows, kk)` is keyed; the hook additionally enforces
   the contract's re-pad on same-count shifts and guards future cut paths. Both behaviors are pinned by tests.

## Off path / parity

- **Off: launch- and byte-identical.** The else arm is today's `fill` + `copyRows` verbatim; all new off-path code
  is host bookkeeping (`Tracker` stores; the banner var is touched only inside the on+reuse arm). Zero new launches,
  zero changed device bytes.
- **On: byte-identical.** The padded list is a function of `(cidx, rows, kk)` alone; reuse requires an exact match
  of the recorded identity with no intervening event. Every padded read is byte-compared against an independent
  reference in the harness (below). The compact row stride used by the sharded fetch (`TF_DS_KV_SHARD=1`) and the
  −1 padding at the attention input are unchanged (stride handling not touched; harness checks both).
- **Ranks:** no rank-conditional code; both ranks see the same host sequence and decide alike. The switch is not in
  `lanes.zig`'s pair-switch handshake (like `TF_DS_PDL` it changes no frames and no bytes — mixed ranks would still
  be bit-identical), but set it alike on both ranks by convention.

## Gate results (all pass)

1. `zig build test` (worktree, zig 0.17.0): **exit 0**, 27/27 steps, 511/511 tests (includes the new module's 5).
   The known stray `failed command:` line appears on some runs with exit 0 and all tests passing (same at the
   pristine baseline; report-0268's deliberate-failure probe stands).
2. CUDA syntax check: the shared `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails on the pristine base
   itself** (stub drift older than the tree; the same finding report-0260 recorded at `8d838cc`) — this patch adds
   no `.cu` at all (the diff is 4 `.zig` files). The round-local fixed stubs
   (`runs/kernel-round-20261122/stubs/check.sh`, untouched) print **`cuda syntax check: ok`** on this tree.
3. CPU-emulation harness `runs/kernel-round-20261122c/h0270/` (`build.sh`, `src/harness.zig`,
   `src/prompt_picks_once.zig` copied from the worktree at run time, `run.log`, `run-negative-control.log`,
   `review-code.md`, `review-harness.md`): **10/10 tests, exit 0** (5 harness cases + the copied module's 5 tests).
   There is **no new kernel** in this patch, so instead of a `.cu` extraction the harness emulates the two pad ops
   the arm skips (the `cuMemsetD8Async` byte fill; `copyRows`) exactly as `attention` calls them and runs the real
   `Tracker`; the off reference is written from BRIEF-0270's spec alone (elementwise: kk picks then −1 to whole
   16-column tiles). Case matrix, every padded read compared byte-for-byte, −1 columns checked at the attention
   input, `cidx`'s compact stride verified intact after every layer:
   - cold 127-token prompt: 8 pads / 30 reuses (hand constants; pad reads at layers 2, 8, 14, 20, 24, 28, 32, 36),
     banner at (127, 63); off run 38 + 38 (today's launches); `TF_DS_PREFILL_SLICE=5` produces identical events;
   - multi-chunk prompts (3 chunks of 127/40/17 rows, 5-layer slices): 8/30 per chunk, each chunk's first padded
     read pads (no cross-chunk reuse);
   - a replay cut mid-prompt (replay 64 of 127): served map 8/30 with the post-cut re-pads; a same-count shift cut
     with **no indexer at or after the cut** re-pads at layer 20 (the hook alone, invisible to the `(rows, kk)`
     key); a prepend-growth cut (127→135 rows) re-pads (the key, where stale bytes would otherwise be read); a
     no-op cut keeps the pad;
   - pick counts 1..64 exhaustively plus 65/80/100/127/200/511/512, rows {1, 5, 7, 16, 17, 127}, both chunk
     classes (ring-mode compact reads and post-cut one_split pads), two selections each; kk 0 reads none;
   - a hand corner table (10 literal expectations × 2 selections) pins the pad predicate's edges independently of
     the transcription (whole tiles, ring mode, one_split, `TF_DS_REPLAY_FLOOR=0`).

   Negative controls (`run-negative-control.log`): cut hook removed → the mid-prompt test fails (`expected 3, found
   2`); the Tracker's `(rows, kk)` shape check removed → the module's tracker test fails; the pad predicate widened
   → the sweep and corner table fail. All three check families are non-vacuous.

## Review findings

Two independent reviewer subagents ran after implementation (both **VERDICT: OK**, no S1; full texts in
`h0270/review-code.md` and `h0270/review-harness.md`; both children hit provider 500s mid-review and were revived
to completion — process note only). Dispositions:

- **S2 (code):** `Chunk.init` builds `var ch: Chunk = undefined;`, so the `Tracker`'s field default was dead and
  the safety rested on `begin` preceding every `attention` — **fixed** (explicit `ch.picks = .{};` in `init`).
- **S2 (harness):** `build.sh`'s drift greps missed the pad-predicate seam — **fixed** (greps for the predicate,
  the `ring_mode`/`one_split` lines and `model.zig`'s cut point).
- **S3 (harness):** the reference inherited the tile width from `padWidth` — **fixed** (`checkPadded` recomputes
  `(kk + 15) / 16 * 16` by hand); the sweep's `want_pad` was derived from the transcription — **fixed** (the hand
  corner table); `TF_DS_REPLAY_FLOOR=0` untested — **fixed** (one corner row).
- **S3, kept as notes:** the shape check is defense-in-depth only at harness level (every real writer of `cidx`,
  `sh.kk`, `ch.n`/`ch.start` accompanies `stale()`, so a shape change without an event is unreachable in the real
  flow; the module's unit test guards it); `build.sh` pins the worktree rather than a commit (the tree content is
  the committed content — run.log was produced after all review fixes); env divergence across ranks is unchecked
  (same convention as every other switch; byte-safe either way).

## Design decisions

- **The reuse key is `(holds, rows, kk)` with `stale()` at the three events** — no generation counter: `stale()`
  clears `holds` outright, and the `(rows, kk)` match catches shape changes without an event. Validity comes only
  from host structure (the contract's rule).
- **The cut hook fires only when the cut rewrites rows** (`first > 0 or pre > 0`), matching "until … a replay cut
  changes the rows"; a no-op cut keeps reuse (byte-identical either way — both pinned by tests).
- **The banner logs at the first reuse** (the arm's first run), `swap(true, .monotonic)` guarded, per the common
  brief; the hardware gate counts the line to prove the arm fired.
- **No `?cuda.Function` and no `.cu`** — the whole saving is elided launches; nothing to look up or compile.
- **No `decisions.md`/`continue.md` writes:** this task's write scope is the worktree commit + `h0270/` + this
  report (the 0268 precedent); this section is the decision record.

## Known gaps (hardware gate only)

1. The saving is prefill-side host launch overhead (~2–3 µs an op): 60 ops a 127-token chunk per rank is real but
   small; price by launch count, expect within-noise wall time. The hardware gate should count the banner line
   (proves the arm ran), verify `cidxp`-consuming outputs are bit-identical switch on vs off (the standing exact
   panel + long-reply hashes), and confirm fill/copy counts (38→8 a chunk) in a profile.
2. Byte-identity of the reused pad is host-argued and harness-proven against the ops' byte semantics; the GPU gate
   should confirm the attention reads exactly the `n × kp` region the harness models (no wider tile reads).
3. `tri.decode_rows` is a mutable global (raised while a concurrent decode round runs); both arms evaluate the pad
   gate identically per call so bytes are unaffected, but the pad/reuse counts can differ from the headline table
   if it is raised mid-chunk (pre-existing; unchanged by this patch).
4. The switch is not part of the pair-switch handshake (byte-safe mixed ranks); set it alike on both ranks.

## Follow-ups (todo.md)

None from this change. The decode path never pads `cidxp` (`rd.cidx` is read compact), so there is no decode twin
of this item.
