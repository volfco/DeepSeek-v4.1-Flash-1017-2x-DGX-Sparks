# REVIEW.md — lane sched (0293/0294/0295/0296) independent review

**Target:** worktree `~/Workspace/tfref-sched` (base `1421187` + `cbd8a23` + `831786c` = 0294, `5560936` = 0295, `87e92a2` = 0296), lane notes `0293.md`, `0294.3b9d9990.md`, `0295.md`, `0296.ede2102b.md`. Read-only; no files touched; no production access of any kind.

## Method and hard limitations (read first)

- **This reviewer runtime has no shell/exec tool** (only read/grep/find/ls/watchdog_diff). Consequently: **(1) tests could not be re-run** — the notes' test claims were verified by reading the test blocks and recomputing their arithmetic by hand, not by execution; **(2) `code-review-graph` (CLI `~/.local/bin/code-review-graph`) could not be run** — caller enumeration was done exhaustively with tree-wide grep instead. Exact commands the supervisor must run are in "Test commands" below.
- `watchdog_diff` covers the launch repo's working tree only (parent consolidation files) — it cannot see committed ranges, and the lane worktree is a separate checkout. Per-commit diffs were **not** reviewed as diffs; the reviewed artifact is the **combined checked-out state**, cross-verified against the notes' file:line maps (they are mutually consistent: 0295's map `draft.zig:84-131/:723` matches exactly; 0294's map offsets −69 lines = 0295's insertions; 0295's `model.zig` map offsets −11 = 0296's +11 — confirming all three items are present in order). "Default table unchanged vs base `1421187`" is corroborated, not diffed (see 3.3).

## Per-point verification

### (1) Focused tests — claims checked statically; re-run pending (supervisor)

The notes claim: 0294 standalone extraction 3/3 (`TF_DS_DRAFT_MS`, `chooseK tail slope`, `round_ms_fit`), 0295 standalone extraction 2/2 (`confLeaderOnly`, `confWork`), 0296 module-filtered 5/5 (`pick unions` ×2 + smoke). I read every test block and re-derived its expectations by hand — all match the notes' output (details in 3/4/5). **Not executed here.** Known-flake baseline for comparison: core/ stream+tokenizer binaries; `round.test.TF_DS_EXPERT_HOT_CACHE`; `prompt_cache.test.0178`; `native.test` follower-killed; `lane_host.test` prefill interleave.

### (2) Caller enumeration (grep-substitute for code-review-graph)

| changed function | callers (tree-wide) | production? |
|---|---|---|
| `chooseK` (draft.zig:1138) | `layer_main.zig:805`, `layer_main.zig:1305` (+ its own tests, doc comments in lanes.zig:1846/2270/3009) | no — replay tool only |
| `draftMsFromEnv`/`parseDraftMs` | `model.zig:343` (Model.open), `layer_main.zig:805/:1305` | yes (startup parse) |
| `Drafter.pass` (want_drafts) | `model.zig:2004` (serving), `layer_main.zig:790/:1290` (replay) | complete |
| `Model.pass` | `lanes.zig:1825` (draftFn, leader), `lanes.zig:2838` (follower `.pass` frame) | complete |
| `stashWants` | `model.zig:2004` (want_drafts), `:2029` (stashPass gate) | complete |
| `confLeaderOnly` | `model.zig:599` only | complete |
| `pickUnionNote` | `round.zig:1602` (fetchDecode, after reuse early-return `:1587-1601`), `:1694` (ownerDecode), `:1756` (ownerRing) | all three decode fetch sites |
| `tallyPickUnions` | `lanes.zig:1570` (profile defer) | complete |
| `printPickUnions` | `native.zig:561`, `lanes_main.zig:452` | complete |
| `useClock` (the 0296 gate) | `native.zig:449`, `lanes_main.zig:239/:346` — all inside profile conditionals | complete |

### (3) 0294 judgment audit

**Claim A — "chooseK has no production caller": VERIFIED.** `layer_main` is the layer-replay gate (`build.zig:133` builds it as module `ds_layer`; `model.zig:2` "the concurrent decoder ... up (layer_main.zig)"; its body is `fx.find`/recorded-window comparison). It constructs `Drafter.init` directly (`layer_main.zig:723`, `:1138`) and never enters the serving loop. Nothing in `lanes.zig`/`model.zig`/`native.zig` calls `chooseK` (the lanes mentions are doc references to the multi.py rule).

**Claim B — "clamp unreachable at served 16-row rounds": VERIFIED.** `depth()` (draft.zig:1060-1064, `by = {5,5,3,3}`) gives `depth(8,·,16)=1`; at `layer_main.zig:805` `others = 0` (live=1), rows = kk+1 ≤ 6; at `:1305` live = nl ≤ 8 → `others ≤ 14`, rows = 15+kk ≤ 20 with kmax ≤ 5 (even kmax=8 via `TF_DS_DRAFT_BLOCK` gives 23 < 24). The clamp needs rows > 24: reachable only at ≥24-row rounds with ≥8 streams (27 rows at 24/8, 34 rows at 48/8) or deeper pins — exactly as the note says. The tail-slope fix itself is arithmetically correct (`round_ms[23] − round_ms[22] = 3.1`, draft.zig:1076/1148-1152) and the test at draft.zig:1163-1178 is a genuine regression pin (old flat clamp strictly increases `v` per kk → k=kmax at 8×48; new prices → k=0). **Both claims hold at every call site.**

**TF_DS_DRAFT_MS parse/refusal vs block-knob rule: MATCHES.** `parseDraftMs` (draft.zig:1104-1110) refuses `0/neg/nan/inf/garbage` with `BadDraftMs`; `model.zig:343` `try draft.draftMsFromEnv()` propagates → refused at startup (before weights). Block knob: `parseBlock`/`blockFromEnv` (draft.zig:72-81) refuses `BadDraftBlock`, re-raised in `Model.agree` (`model.zig:757`, `_ = try want_block` at `:778`). Both malformed-refuse at startup. Minor asymmetries: see N1/N5.

**Re-fit provenance: VERIFIED.** `runs/trace-review-20261123/out/layer_fit.txt:43` fx1L footer = attention 10.322 + MoE 24.334 = **34.656 ms at 6 rows**; slopes 2959.4 + 193.7 µs/row ≈ **3.1536 ms/row** (per-layer sums; footer rounded). `tools/round_fit.py:60-66` asserts the footer cross-checks; `tools/round_fit_residuals.txt:8` is **byte-identical** to `draft.zig:1017`'s `round_ms_fit`; recomputed entries (r1=18.888→18.9, r6=34.656→34.7, r24=91.42→91.4) match. Residuals numbers in the note (0.15% fx1L/fx1Lb spread, +17.4/+17.8 served-over-fit, +25.5/+8.2 wall non-forward share, −9.8 wall-vs-served) all reproduce from the files. The note's todo.md:714 correction (20 rows = 96.8, 99.9 is 21 rows) is consistent with the table.

**Default round table unchanged with switches off: VERIFIED (corroborated, not diffed).** `model.zig:341`: `o.round_ms orelse if (envOnOff(TF_DS_ROUND_MS_FIT)) fit else &draft.round_ms`; `native.zig:394` feeds `TF_DS_ROUND_MS` into `o.round_ms` — **explicit comma list still wins**; `envOnOff` = "1"/"on" only (model.zig:2060-2063). `round_ms` (draft.zig:1076) is byte-identical to `round_fit.py`'s `SERVED` and to 0293's base-era citations (`round_ms[23]=109.2`, `[22]=106.1`, `[19]=96.8`). Served policy reader `lanes.zig:730-732` is width-bounded (`t[@min(i, t.len-1)]`, window rows ≤ 17) — the clamp cannot fire. Caveat: no git diff against `1421187` was possible.

### (4) 0295 output invariance

**Follower confidences are consumed nowhere that affects output.** Tree-wide `confs` readers: `draft.zig:794` (the gated fill), `lanes.zig:1834` (`holdDrafts`, in `draftFn`), `lanes.zig:1860` (`survivals`, in `probabilitiesFn`), `lanes.zig:1450` (`triageLive`, in `verifyFn`), `layer_main.zig:805/:1305` (replay). `verifyFn`/`draftFn`/`probabilitiesFn` are the lane-core backend callbacks (lanes.zig:756-766) built **only on rank 0** (`lanes_main.zig:289-290` "rank 0: the requests, the backend and the lane core"; followers run `followWith` and `return 0` at `:274-288`). The follower's `.pass` handler (lanes.zig:2838-2847) reads only `d.drafts` and sets `dep = @splat(steps)` — held depth is host-side and untouched; triage's `win.live` rides the frame from the leader. Keyed/spec sampling (`pass_keyed`, SpecTap/NoiseTap) runs in the pass body before the skip point (draft.zig:813-895) and reads no confs. `dr.conf` (device) has exactly two references: the gated DtoH (draft.zig:785) and the skipped GEMV (`:924`) — no stale reader.

**SUBMIT_EARLY stash path survives the skip.** `Model.pass` passes `want_drafts = m.prof or m.stashWants()` (model.zig:2004); `read_drafts = want_drafts or !conf_leader_only` (draft.zig:783) keeps the `ho` download + `dr.drafts` fill (`:784/:790-792`) whenever the stash needs it. `stashWants` (model.zig:2018-2021) preserves the exact three conditions (Engram host + `esh.enabled` + AIO ring); `stashPass` gates on it (`:2029`) before reading `drafts`. With the switch off (default), `read_drafts` is true and both downloads/fills run as before — byte-identical.

**Invalid-ID checks survive.** `ch.invalid`'s writers are the conf pack/gather kernels (draft.zig:910-923, skipped) — but the skip is disabled whenever `TF_DS_CHECK_INVALID` is on (`confLeaderOnly` condition 3, draft.zig:95-96 wired from `m.ops.check_invalid`, the same instance `m.eng.ops` points to, model.zig:450/:495/:599). The reader (`round.zig:1066-1070` `checkInvalid`) is gated on the same flag. Enabled diagnostic ⇒ no skip ⇒ checks preserved.

**Replay keeps the full head.** `layer_main.zig:790/:1290` pass `want_drafts = true`; `Drafter.conf_leader_only` defaults false (draft.zig:425, `:457`) and is set only at `model.zig:599` inside `Model.open`, which `layer_main` never calls (only `Model.openRing`, an RDMA helper, `:459`).

### (5) 0296 instrumentation

**No behavior change off-profile:** `pickUnionNote` early-returns before any GPU/host work when `!u.on` (round.zig:1537-1539); per-round slots zero is gated (round.zig:2130); `u.on` is set **only** in `Model.useClock` (model.zig:1833), which is called only under `TF_DS_ROUND_PROFILE=1/2` (native.zig:438-450) or the batch driver's `--profile` (lanes_main.zig:235-240, `:338-346`); the tally rides the profile's `defer if (self.prof)` (lanes.zig:1570) and is double-gated (`tallyPickUnions` checks `u.on`, model.zig:1895). Unconditional allocations are two tiny static-shape buffers (round.zig:437-440, `layers×8 B` + host mirror) — see N4.

**One line a run:** `printPickUnions` (round.zig:1559-1578) emits one self-contained JSON object (brace escaping verified correct: `{{`/`}}` pairs produce `{"rank":..., "kv_pick_union":{...,"by_depth":[...]}}`); called once per `/metrics` scrape (native.zig:560-564, gated on `pick_unions.on`, before `profile_note`) and once per driver run (lanes_main.zig:451-454). Cumulative-with-deltas matches the profile line's convention.

**Distinct-picks-by-depth arithmetic: correct.** `bandOf` = floor-log2 (`63 − clz(bucket|1)`, round.zig:1474) — tests pin 32768→15, 131072→17, 30000→14, 1024→10 (round.zig:2536-2540), matching the gate depths' labels. `tally` (round.zig:1479-1486): `picks += rows×kk`, `distinct += count`, zero-count slots skipped, cumulative bands — the test (round.zig:2517-2533) matches the note (2 fetches, picks `4×(512+256)=3072`, distinct 812; a second tally at 131072 leaves band 15 alone). Replicate fetches copy the fresh count (`FetchKey` freshness, round.zig:1541-1557) keeping `fetches`/`picks` exact. The `sums + scan_segs*4` count location matches `shard_ops.unionCount` (shard_ops.zig:126-151) and the prompt fetch's own read (prompt.zig:1299).

**Series gate:** 0295/0296 are each one item, off by default or profile-riding. 0294's knob and fit table are off/default-identical; the chooseK clamp fix is the one unswitched behavior change (sanctioned by the recovery follow-up; see J1/N2).

## Findings

**J1 — judgment (0294): the unswitched clamp fix changes replay `window_ok` outcomes at ≥24-row recordings; the caveat is not recorded.**
`layer_main.zig:805/:1305` compare `chooseK`'s kept window against recorded windows. At served 16-row rounds nothing changes (verified above). At `TF_DS_ROUND_ROWS ≥ 24` with ≥8 streams (e.g. the four-node lane's 48; the note itself names this reachability) the fixed tail slope moves `k` away from what the old flat clamp (and possibly the Python `MultiDecoder._choose_k` reference, whose source is **not in this repo** — port parity there is unverifiable) produced — a replay of such recordings can now flag `window_equal: false`. 0294's open decision 3 records only the *knob's* replay-parity caveat, not the clamp's. Smallest fix: one line in the consolidation notes/decisions recording that wide-round replay expectations were produced under the flat clamp (todo follow-up 4 ("round tables past 24 rows") is its natural home). Non-blocking: served shape is unaffected and the fix is what the recovery follow-up ordered.

**J2 — judgment (0296): the `picks = rows × kk` denominator's "what every-pick fetching sends today" basis is fetch-site-dependent.**
`fetchDecode` ships `R × kk` picks (round.zig:1604-1608, `rows=R, kos=kk`), but the owner sites mask the non-held rows' picks (round.zig:1693-1695 "the picks of the rows this rank holds, none for the others'") — pair-wide traffic sums to `R × kk`, per-rank traffic is `own × kk`, while the counter counts `R × kk` per fetch on rank 0. The `distinct/picks` ratio is internally coherent (both over the same `R` rows) but its "wire bytes saved" reading differs per site. Record the basis when reading the 32K/128K gate. Non-blocking.

**N1 — nit (0294): `parseDraftMs` accepts hex-float spellings** (`std.fmt.parseFloat`, draft.zig:1107) and enforces an **undocumented 60s cap** (`v > 60_000.0`, `:1108`) — the block knob (`parseBlock`, decimal-only) and the doc comment (`:1097-1101`) don't mention either. Cosmetic.

**N2 — nit (0294): `cbd8a23` bundles two semantic changes** (the knob + the clamp fix) in one commit — a "one item per patch" process note. Both are independently tested.

**N3 — nit (0295): skip banner text incomplete.** draft.zig:902 says "the draft ids still read back for the Engram stash" — they also read back for `m.prof`'s pass counters (model.zig:2004). Cosmetic.

**N4 — nit (0296): doc/behavior wording.** `PickUnions.on`'s comment "off: nothing runs and nothing allocates" (round.zig:1461) vs the unconditional tiny allocations in `Round.init` (round.zig:437-440). Wording only.

**N5 — nit (0294): `TF_DS_DRAFT_MS` is not in `Model.agree`'s cross-rank vector** (model.zig:753-772 covers mode/pool/window/round_rows/window_rows/gemm_rows/block only) — an uneven pair is not refused, unlike `TF_DS_DRAFT_BLOCK` (`DraftBlockMismatch`). Mitigated: only rank 0 prices rounds (`facts()` → lane core, leader-only) and the staged plan sets both ranks alike. Worth one line in the knob's doc if ever promoted.

**No defects found.** Nothing blocks consolidation; J1/J2 should be relayed into the parent's notes.

## Test commands (supervisor must run; none were run in this review)

```
cd ~/Workspace/tfref-sched
# full gate (fix stage), tee + tail polling, capture failing names:
( zig build test 2>&1 | tee /tmp/lane-sched-0293-96-test.log ) &
# poll: tail -n 20 /tmp/lane-sched-0293-96-test.log   (10+ min; compare vs flake baseline below)
# flake baseline: core/ stream+tokenizer binaries; round.test.TF_DS_EXPERT_HOT_CACHE;
#                 prompt_cache.test.0178; native.test follower-killed; lane_host.test prefill interleave
# focused (the notes' precedent: build.zig has no test filter; use the module graph, zig 0.17.0):
zig017 test -lc --test-filter "TF_DS_DRAFT_MS" --test-filter "chooseK" --test-filter "round_ms_fit" ...   # draft.zig blocks
zig017 test -lc --test-filter "pick unions" --dep cuda --dep lanes --dep core --dep engine_api --dep fabric \
  -Mroot=zig/src/families/dsv41/dsv41.zig ...   # 0296's exact command in 0296.ede2102b.md
zig017 test -lc --test-filter "TF_DS_CONF_LEADER_ONLY" ...   # 0295's confLeaderOnly/confWork blocks
python3 tools/round_fit.py runs/trace-review-20261123/out/layer_fit.txt   # footer asserts; literal must equal draft.zig:1017
```

---

**VERDICT: findings**

*(No defects. Findings are 2 judgment + 5 nit, all report-only and non-blocking: J1 wide-round replay-parity caveat for the clamp fix, J2 the picks-denominator basis, N1–N5 as listed. Both of 0294's audited claims — no production caller, unreachable at served 16-row rounds — verified at every call site. 0295 output invariance and stash/invalid-ID survival verified. 0296 verified instrumentation-only and arithmetically correct. Test execution and code-review-graph remain outstanding due to the no-shell limitation — commands above.)*

---

**Skipped / not checked (ponytail):** git-level diff-vs-base verification, the commit contents/line counts, and all test execution (no shell in this role — commands handed to the supervisor); Python `multi.py`/`MultiDecoder._choose_k` source is not in this repo, so 0294's port parity at wide rounds is argued, not proven. Risk you must know: J1 means replaying ≥24-row-round recordings after this lane can fail `window_equal` purely from the clamp fix — treat such a failure as expected until the recorder's rule is compared.