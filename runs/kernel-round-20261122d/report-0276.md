# report-0276 — TF_DS_CACHE_REPLACE_STALE

**Commit:** `7b7e4e1` (`7b7e4e14c3902408697a67d34ff8a37779fd4fc3`) on `kernel-0276` (parent `27baa93`). **Switch:** `TF_DS_CACHE_REPLACE_STALE` (off by default; `1`
on, `0`/`""`/unset off, anything else refuses startup with `TF_DS_CACHE_REPLACE_STALE: expected 1 or 0`).
**Subject:** `0276 core: TF_DS_CACHE_REPLACE_STALE (off by default): keeping a state the cache already holds replaces
a stale duplicate`.

## Premise

Holds, with one re-derivation (below). In our tree `keepWith` (`zig/src/core/prompt_cache.zig:1146`) refreshes
`e.used` and keeps the old state whenever a keep duplicates an existing entry (same `at`, same tokens), except the
pre-existing `full_end and !e.full_end` displacement. When the duplicate's saved state has gone stale (KV-shard
placements gave its rows away, the family dropped it, a restore is installing over it), the refreshed entry is dead:
every later lookup passes it over (`liveWhy` -> `.stale`) and the recomputed state that just passed the keep is
thrown away again at the next eviction. Upstream PR #134's keep half fixes exactly that: replace the dead copy with
the fresh one at the keep that recomputed it. The brief scopes out the PR's other half (wiring `usable` without a
spill tier) — see Design/Gaps.

## Files touched

- `zig/src/core/cache_replace_stale.zig` (new, pure): switch parse (`idx_fused.zig` pattern: `env_name`, `parse`,
  `fromText` refusing startup on anything but unset/`""`/`0`/`1`), the staleness-probe rule `probeLen(full_end, at)`,
  tests (parse truth table + probe lengths).
- `zig/src/core/prompt_cache.zig`: `Store.replace_stale` (off by default), the duplicate arm in `keepWith`
  (replaces a stale, point-less duplicate; refreshes otherwise, unchanged), the one-shot engagement banner
  (`replace_stale_noted`, quiet under `zig build test` per the file's `note` idiom), re-export of the switch module.
- `zig/src/core/prompt_cache_test.zig`: two tests — stale duplicate replaced on / refreshed off (with live-accounting
  and "usable again" after), and full ends: stale duplicate replaced at its own keep, live end never copied again.
- `zig/src/core/prompt_cache_tier_test.zig`: one test — a stale duplicate with a durable point (`e.point != null`)
  is refreshed, not replaced (the tier's `adopt` replaces its stale twin on restore).
- `zig/src/core/engine_api.zig`: one-line re-export (`cache_replace_stale`), beside `prompt_cache`.
- `zig/src/families/dsv41/lanes_main.zig`: the switch parsed at startup via `fromText` (malformed refuses), set on
  the store (`tf-dsv41-lanes`'s construction site).
- `zig/src/native/cuda.zig`: same at the served path's construction site (`openWith`, any family with `snaps`).

## What "stale" means here (the brief's documentation ask)

- **Source patch** (`src/upstream-pr134-0069.patch`): a duplicate is replaced when `e.point == null` and the family's
  `usable(saved, maxInt(u32))` is false — its reading: "the family let it go (stale): it no longer restores", probed
  at the maximum length so no prompt-length gate can count ("a valid state near the prompt end is never copied
  again").
- **Our tree**: the family rule is `Lanes.usable` (`zig/src/families/dsv41/lanes.zig:1863`): the state must still be
  in the snaps list (resident), not held by the KV-shard planner, `!stale` (rows written over: `noteStale`, e.g. a
  stream's placement took them), `!loading` (a restore's destination) and `at > 0` — the prompt-length-independent
  part. Everything after is length gates: `at == len -> is_end`; `is_end -> tail_batch != null and even at` (0180);
  tail-carrying (TF_DS_EXACT_RESUME) snaps -> `tailPre != null`; else on the 2048-row grid and `at <= len - window`.
- **Tier state**: `e.point != null` (a durable point stands behind the memory copy) is excluded from replacement,
  verbatim from the source: the stale twin is the restore machinery's business (`adopt` already removes it there,
  `prompt_cache.zig:869-877`). Full-end keeps never write a point (`keepWith` returns before the tier write), so the
  guard never blocks an end-state replacement.
- **Re-derivation (deviation from the source's probe):** at `maxInt(u32)` our `Lanes.usable` still applies the
  `is_end -> tail_batch != null and at % 2 == 0` capability gate — it is not a staleness gate. A live end state
  (tail batch off, or an odd prompt length) would be judged stale and copied again on every duplicate keep,
  contradicting the source comment's own intent. `probeLen(full_end, at)` therefore probes a `full_end` entry at its
  own length (where the family answers `is_end` — pure staleness for an end snap) and any other entry at
  `maxInt(u32)` (where every length gate is vacuous for the states `snapSaveFn` can produce: grid ends, tail snaps
  via `tailPre`, end snaps flagged `full_end`). The harness case `end-live-refresh-odd-off` demonstrates the
  divergence (`usable(maxInt)=false` vs `usable(at)=true`) and that the adapted probe refreshes instead of replacing.

## Which keeps change behaviour (the brief's show-from-the-code ask)

Only `keepWith`'s duplicate arm (`prompt_cache.zig:1151-1162`), and only when `replace_stale` is on:
`e.point == null` and `usable` says the state is dead at every prompt length. Concretely: a re-kept state the family
let go (stale rows / dropped snap / loading / planner-held) is dropped and its fresh copy saved in the same keep;
everything else — including a valid state near the prompt end and any entry with a durable point — takes today's
refresh path (`e.used = s.clock`, shared promotion, return true) with no copy. The pre-existing
`full_end and !e.full_end` displacement is untouched (harness `end-over-notend-*`, both switch states). The decode
path is untouched: this is cache policy only — it changes *which* prefix restores, never a value.

**A replaced entry cannot be in use:**
- Replacement needs the family to say dead-at-any-length. A state a pass is standing on or copying from cannot be
  there: placement never takes rows a live stream holds (`placeKept`/`staleable`: `!clashes(live, ...)`), and the
  in-place resume source is excepted when a stream stales overlapping snaps (`markStale`'s `except`; "the resume in
  place ... stales nothing", `lanes.zig:1955-1959`). `loading` snaps are restore destinations whose entries carry a
  point (the guard skips them).
- The one held-pointer pattern is `Job.entry` (`lane_host.zig:238`); `Job.kept` calls `job.reported()` first — "before
  a keep can evict the entry the pass restored" — releasing the hold and nulling `stream.reuse.saved` before
  `keepWith` runs, and `Store.resumed` re-checks membership (`indexOfScalar ... orelse return`) before touching an
  entry. A job's planned keep marks stand strictly above its resume point (`Store.marks`: `at <= from` skipped).
  One corner (independent review's finding 4): the endstate end keep (`lanes.zig:1154`, `at = done.len`) is not
  mark-gated, so `lanes_main`'s `keepHook` — which holds `j.entry` across its own `keepWith` — can duplicate its
  resumed entry on an identical resend (resume at `prompt.len`, zero rows). Safe: the arm requires the state to be
  dead at its own length (the resume is already doomed), `keepHook` reads `j.entry` before `keepWith`, and
  `st.resumed` (`lanes_main.zig:409`) only compares the pointer (membership re-check) — "a replaced entry cannot be
  *validly* in use" is the precise claim.
- Cross-job: `keepWith` may free an entry another job looked up but has not restored yet. That exposure is unchanged
  from today: the same function already removes entries there (`full_end` displacement, `victimBut(null, ...)` room
  evictions that prefer exactly the superseded duplicate, `end_limit`) under the documented invariant at
  `lane_host.zig:673` ("The entry stays alive until the backend restores it: nothing keeps between here and this
  stream's own pass"), and `adopt` already removes stale twins the same way. The arm is one more caller of the
  existing `remove` within that invariant.

## Launch counts (served shape: TF_DS_DRAFT_BLOCK=6/_KEPT=2, graphs on, TP2, up to 4 streams)

Before/after: **identical** — the patch touches no kernels and no launch sites (0 launches added or removed at the
served shape). When the arm engages, a *later* turn's prefill may resume from the replaced state instead of missing,
so that pass runs fewer chunk launches (fewer prefill rows) — a cache-hit effect only the hardware gate can measure;
every individual launch shape is unchanged. No host syncs, no pageable transfers, nothing added to any
graph-captured body.

## Gate results

1. `zig build test`: **exit 0** (first run hit the documented `lane_host.test.prefill` interleave flake; two
   subsequent runs clean). `zig build` (all exes incl. `tf-dsv41-lanes` and the native server) also compiles clean —
   `zig build test` alone does not compile `lanes_main.zig`.
2. CUDA syntax check: base `27baa93` and this tree both report the **same 9-error stub set** (normalized diff equal;
   the patch touches no file under `zig/kernels/`).
3. CPU harness `h0276/` (`build.sh` + `build.zig` + `harness.zig` + `model.zig` + `run.log`): the real
   `prompt_cache.zig` (imported from the worktree by path) run against an **independent model of the intended
   policy** (written from the brief's policy sentence, `model.zig`). 10 cases / 22 steps, whole-state + outcome
   comparison each step: keep→refresh baseline; keep→mark stale→keep (replaced) →keep (refreshed); the same with the
   switch off (refreshed, still dead); with a tier (point guard: refreshed); full-end stale replaced then refreshed;
   live end state never copied again (odd at + tail batch off — the re-derived probe case, with both probe verdicts
   printed); live end refreshed (even at + tail batch on); full end displacing a non-full-end duplicate with the
   switch off and on (pre-existing rule); a `loading` state treated as let-go. **All match the model.** The
   engagement banner appears exactly once in `run.log` (checked by `build.sh`).

## Engagement banner

`info: dsv41: TF_DS_CACHE_REPLACE_STALE: a stale kept state is replaced by its fresh copy (first at 5 tokens)` —
one `std.log.info` behind an atomic `swap(true)` guard in `keepWith`, exactly 0265's `head_scored_noted` idiom,
suppressed under `zig build test` (the build runner fails tests that write stderr; `note()`'s idiom). The hardware
gate can count the line to prove the arm ran.

## Design decisions

- **Keep side only** (the brief): the source's `usable`-wiring half (`native.zig`/`cuda.zig` moving
  `snaps.usable = L.usableFn` out of the tier branch) is not ported — production runs `SPILL_GIB=180`, so the tier
  is on and `usable` is wired (`native.zig:443-444`, `cuda.zig:494`). The arm is additionally inert where `usable`
  is absent (its `if (s.family.vtable.usable)` guard), i.e. off-tier deployments without
  `TF_DS_EXACT_RESUME`/`TF_DS_PROMPT_ENDSTATE` keep today's behaviour (see Gaps).
- **Store field, not a Rules field**: `replace_stale` sits beside `shared_plan`/`metrics` (store-policy switches set
  after `init`); the parse runs at both dsv41 store-construction sites at startup. The switch module is a core
  module (`zig/src/core/cache_replace_stale.zig`) because the change lives in core (`prompt_cache.zig`), per the
  brief; the commit subject keeps the brief's `core:` prefix. GLM/FlashNext stores are not wired: their snapshot
  vtables have no `usable`, so the arm cannot fire there (dead code otherwise).
- **Counts**: a replacement counts as a kept state (`counts.kept`) — it is a real fresh save; the removal is
  uncounted, like `adopt`'s stale-twin removal and the `full_end` branch. The engagement banner logs only rule-3
  replacements (not the pre-existing `full_end` displacement).
- **Sticky flags are not carried over on replace** (independent review's finding 2): the fresh entry gets the
  current keep's `shared`/`group`, while a refresh ORs them into the old entry. A dead shared state re-kept after
  the note ring rotated its key can lose its shared placement priority / tier share flag. The source patch behaves
  the same, the brief is silent, and the harness model matches the implemented semantics; recorded as a gap.
- **Both ranks**: the replace is `remove` + fresh `keep`, both already mirrored to rank 1 by the family wire
  (`snapDropFn`/`snapSaveFn`); no rank-conditional code. Under graph capture: no GPU work at all.
- **Deviation from the source**: the per-entry probe length (`probeLen`, above). The source's single
  `maxInt(u32)` probe does not survive our `Lanes.usable`'s `is_end` capability gate; the adaptation preserves the
  source comment's stated intent and is demonstrated in the harness.
- **Process conventions**: per the round's export convention (0265's commit touches patch files only), no
  `decisions.md`/`todo.md`/`continue.md` are added to the worktree — this report is the decision record and its
  "Known gaps" section is the follow-up list.

## Known gaps (hardware gate / follow-ups)

- The prefill-launch reduction when the arm engages (the hit-rate effect) is only measurable on hardware; locally
  only the policy outcomes are verified. Bit-identity of any computed value is not at risk by construction (no
  arithmetic or decode path touched), but the resumed-position shift should be watched at the served shape.
- `probeLen` is exercised against a modelled `Lanes.usable` shape (harness), not the real one on GPU pools. The
  model covers residency/stale/loading/planner/is_end/tail-batch/grid/window; tail-carrying (TF_DS_EXACT_RESUME)
  snaps pass both probes identically and are outside the matrix. If the family's `usable` grows new gates, revisit
  `probeLen`'s "vacuous at maxInt" claim (it is written in its doc comment).
- Off-tier deployments with KV-shard staleness but no spill tier and no `TF_DS_EXACT_RESUME`/`TF_DS_PROMPT_ENDSTATE`
  have `usable == null`, so the arm never engages there: that is the source patch's "usable half", deliberately out
  of scope per the brief (production always runs the spill tier). If such a deployment matters, port the
  `native.zig:448-449`/`cuda.zig:484-486` wiring as a follow-up — it would make the arm work there unchanged.
- A replacement whose fresh save is refused or fails keeps nothing (the dead copy is already dropped; the keep
  returns false, counted `refused`/`failed`) — same as the source (review finding 3); modelled in `model.zig`'s
  header but not in the case matrix (the tree tests cover the failure path's neighbours). Nothing valid is lost —
  the removed copy was dead — but the cache changed under a `false` return.
- The engagement banner is process-wide once (atomic); a test-runner process that replaces before the served one
  cannot happen (separate processes), but a hypothetical multi-store process would log only its first replacement.
- Independent review (subagent, source-trace; verdict **approve**): confirmed the off-path gate, the arm semantics,
  `probeLen` against `Lanes.usable` (incl. tail snaps passing maxInt), the wiring and rank parity, and the lifetime
  analysis beyond the corner above (finding 4). Its finding 1 (arm silently inert without tier/exact/endstate — the
  out-of-scope "usable half") is the first gap on this list. The reviewer had no shell, so the gate commands in this
  report were run by the implementer (and re-run once more at the commit).
