# REPORT — lane tier0 consolidation (0297/0298/0299 + review fixes + k280 window)

Consolidation of lane tier0, run 2026-11-23 (recovery finish; the first pass hit the 55-min cutoff after the
full-suite capture and patch export, before this report). Worktree `~/Workspace/tfref-tier0` (branch
`tier0-0297`), now clean at **`c82ff51`**. Review source: this run's `REVIEW.md` (read-only, verdict
"findings: OK with notes"). Nothing touched hardware; no shared docs edited; no `sync-head`.

## 0. TL;DR

- **Commits**: `c4e227a` 0297, `f863532` 0298, `a4db035` 0299 (from the lane's earlier runs) + **`c82ff51`
  0299z** (this run: REVIEW.md F3's defect fix, committed after focused tests passed).
- **REVIEW.md findings disposition**: F3 fixed (0299z), F4 fixed (`window.sh` comment), F7 fixed (`work.py`
  docstring); F5 = no code change (the claimed `table_gpu.py` bug is not real — drop the follow-up);
  F1/F2/F6 are judgment calls → **deferred to the parent** (recorded in §5/§6).
- **Full `zig build test`**: run once with `2>&1 | tee` + tail polling; log at `buildtest.log` in this
  output dir (744 lines, seed `0xda5023b1`). Three `failed command:` binaries (the known load-flake class),
  all green standalone at the same seed; **no new failing name/module vs the known-flake baseline**. Exact
  per-test failing names are unrecoverable from the capture (see §2 — wrapper limitation, same one the 0a
  closeout hit).
- **Patches**: `patches/` holds four exports of `1421187..HEAD`, renamed to the `zig/patches/` convention
  (0297/0298/0299 + the 0299z fix).
- **Scoring edits** (`runs/k280/work.py`, `runs/k280/window.sh`, `runs/overnight-20261122/arms.sh`,
  `runs/overnight-20261122/table_gpu.py`) remain **uncommitted for the parent** (§5, open decision 4).

## 1. Per-item summary

### 0a — the TF_DS_IDX_FUSED selection divergence, localized (0297 + 0298)

| commit | files | what |
|---|---|---|
| `c4e227a` **0297** | `zig/src/families/dsv41/key_prof.zig` (new), `dsv41.zig`, `model.zig`, `round.zig`, `lanes_main.zig`, `idx_fused.zig` | `TF_DS_KEYS_PROFILE` (off by default): the pruned selection site records the keys' width and key-encoding class (`kit`/`fold`) per class on a `key_profile` line beside `prefill_profile_ms`; host-side localization tests. |
| `f863532` **0298** | `zig/src/families/dsv41/idx_fused.zig` | `key_kit_flip` + `kitKey` (the kit's `_score_keys` encoding) + pin tests. No runtime caller in this tree — the fold lands with parked 0206 and must write these. |

Mechanism (proven host-side in `idx_fused.zig`'s tests): the fold writes `scoreKey`'s encoding (orders
unsigned) where the kit's `_score_keys` writes keys that order as **signed** i64 (torch.topk). The pruned
selection reads them signed, reversing every negative-vs-nonnegative pair at the selection boundary — the
observed 4/39 signature (len_20000/30000/50000/64000 = width inside 16,384 < width ≤ 65,536; len_100000
outside). Evidence: `runs/fused-followup-20261121/RESULTS.md:171-190` (only the sign bit differs, all 64
columns), REVIEW.md re-derived the orders independently and confirmed.

### 0a closeout — the full-suite question (no tree edits)

The previous run's full-suite failures (lanes core/stream, tokenizer, dsv41 binaries) were confirmed the
**known load-flake class**: all pass standalone (75/75, 23/23, 527/527) including under CPU load; the three
binaries are import-disjoint from the lane's changes. This run's suite re-confirms (§2).

### 0b — the clock-fold profiler + per-round distributions (0299)

| commit | files | what |
|---|---|---|
| `a4db035` **0299** | `zig/src/families/dsv41/clock_fold.zig` (new), `model.zig` | The round-clock buffer fold extracted verbatim from `Model.verify` into pure host code; `verify` calls it (behavior-identical); three host tests (nonzero phases 1.5/0.4 µs a round from a constructed profiler state, the cap/fill bound, and the `Model.open`/`Round.init` clk-init source pins — the bug class that served `0xAAAA…` rounds and 0 µs phases when the pinned-image rebuild dropped the init hunk). |

Scoring-pipeline piece (working-tree edits in the main repo, **uncommitted**, parent consolidates):
`runs/k280/work.py` sampler thread → `per_round.samples` per cell (forward/pass/wait per-round samples,
~1 s scrape cadence; attribution-only — this is why its docstring no longer claims "byte-for-byte the
original workload", REVIEW.md F7); `runs/overnight-20261122/table_gpu.py` "Per-round distributions" table
(mean/median/p95/p99, nearest-rank; additive, backward-compat proven by dry-run diff on k272/k267: 0 lines
changed, 22 added); `runs/overnight-20261122/arms.sh` `*.frames.txt` full-series capture beside the
pre-existing `pass_counts` edit (preserved intact).

### 0c — the k280 hardware window packaged (runs/k280/window.sh)

`runs/k280/window.sh` (new, workstation-side): rsyncs the README's exact two file lists to the head (never
`sync-head`; `.env` never synced), then runs `passprof.sh → run.sh → l2-run.sh → topk-run.sh` one at a
time, each gated on its `run.log` showing `restored: validate rc=0` and its `finished` line. No arms added
or changed. Evidence in the 0c note: `bash -n` on all five scripts, `py_compile` on the entry points,
`gate_drafts.py` reproduced on saved suites (k272: 19 rows IDENTICAL exit 0; k267: 12 rows IDENTICAL exit
0), verdict-gate logic checked on three synthetic logs.

### This run — REVIEW.md evidence-backed fixes

| finding | disposition | where |
|---|---|---|
| **F3** — `clock_fold.fold`'s `h.len < 3` guard skipped the round count the pre-image performed (doc says "every call counts one round") | **fixed + committed `c82ff51` (0299z)**: `clk_rounds.* += 1` moved above the guard (restores the pre-image's unconditional per-call count; unreachable on the verify path where `h.len` is 1025, so no behavior change in production) + a pin in the cap-bound test (a 2-entry buffer counts a round). Focused tests 3/3 green. | `zig/src/families/dsv41/clock_fold.zig` |
| **F4** — `window.sh` header mis-attributed its hazards to the README (the README has no `sync-head`/stale-`.env` hazard) | **fixed**: comment now reads "(per standing rules, echoed at start)" (the hazard itself is real and kept). | `runs/k280/window.sh:8` (main repo, uncommitted) |
| **F7** — `work.py` docstring "byte-for-byte the original workload" oversells (the sampler's ~1 Hz `/metrics` + `docker logs` scrapes run in every cell regardless of `WORKLOAD_SAMPLED`) | **fixed**: docstring now says "the original request stream (the per-round sampler's ~1 Hz /metrics + docker-logs scrapes run in every cell either way; attribution-only)". | `runs/k280/work.py:8` (main repo, uncommitted) |
| **F5** — 0b's claim that `table_gpu.py`'s wall-seconds loop reads the previous iteration's `v` | **not a bug** (REVIEW.md verified the loop is correct as written); the follow-up item is **dropped**, no code change. | — |
| **F1/F2/F6** | judgment calls → deferred to the parent (§5 open decisions, §6 follow-ups). | — |

## 2. Full-suite record (before/after vs the known-flake baseline)

Command (run once, with `tee` + tail polling per the task):

```
cd ~/Workspace/tfref-tier0/zig && /home/colum/.local/zig-0.17.0/zig build test 2>&1 | tee <out>/buildtest.log
```

Log: `buildtest.log` in this output dir (744 lines; complete to the wrapper's `BUILDTEST_EXIT=0` marker;
seed `0xda5023b1`).

| | 0a run (baseline, `/tmp/tier0-0a-buildtest.log`, seed `0x1c6da9fb`) | this run (`buildtest.log`, seed `0xda5023b1`) |
|---|---|---|
| `failed command:` binaries | 3: `cc6450…` lanes core, `6ab41a…` tokenizer, `81c16b…` dsv41 | 3: `93f023c6…` lanes core, `4c9912e6…` tokenizer, `af2d3e36…` dsv41 |
| standalone re-run (same seed) | 75/75, 23/23, 527/527 pass | 75/75, 23/23, **530/530** pass (527 + 0299's 3 clock_fold tests) |
| named-error lines (`error: '…' failed:`) | none (capture truncated at the error summary) | **none** |
| exit | n/a (timeout-killed capture) | `BUILDTEST_EXIT=0` |
| delta vs known-flake baseline | same three modules as the standing list | **same three modules; no new failing name or module** |

**Exact failing test names: unrecoverable — wrapper limitation.** The three `failed command:` lines name only
the test *binaries* (hash paths). Per-test names are not in the capture: with `--listen=-` the build protocol
does not stream per-test failure names into the console output, and this run's log shows no
`error: '<test name>' failed:` lines at all (the same limitation the 0a closeout documented). The wrapper
itself is sound: a scratch deliberately-failing build through the identical `2>&1 | tee` + `BUILDTEST_EXIT`
wrapper produced the full named error, `Build Summary`, and `BUILDTEST_EXIT=1` — so the exit-0 / no-summary /
no-named-error shape of this run is the known load-flake signature (fail under full-build parallel load,
clean standalone; `failed command:` present, build summary absent), **not** a deterministic test failure and
not a wrapper artifact.

The known-flake baseline names (standing list: `round.test.TF_DS_EXPERT_HOT_CACHE`, `prompt_cache.test.0178`,
`native.test` follower-killed, `lane_host.test` prefill interleave, core/ stream+tokenizer binaries) are
consistent with this capture; the log's warning noise (Engram read failures, KV transfer `InjectedFailure`,
`TF_DS_*: expected …` lines) is the tests' own expected-failure output, not failures.

## 3. Patch export

`patches/` in this output dir — `git format-patch 1421187..HEAD -o patches/`, renamed per the
`zig/patches/` `NNNN-<slug>.patch` convention (slug = subject minus the leading patch number, lowercased,
non-alnum → `-`, cut at 48 — `scripts/export-fnext-series.sh`'s `slug()`):

```
0297-dsv41-tf-ds-keys-profile-off-by-default-the.patch        (c4e227a, +6 files: key_prof plumbing + localization tests)
0298-dsv41-kitkey-key-kit-flip-the-key-fold-writes-th.patch   (f863532, idx_fused.zig: kitKey spec + pins)
0299-dsv41-clock-fold-the-round-clock-profile-s-host.patch    (a4db035, clock_fold.zig + model.zig)
0299z-dsv41-clock-fold-the-round-count-moves-above-the.patch  (c82ff51, REVIEW.md F3 fix + pin)
```

Note the export now covers four commits: the task's three names plus the review-fix commit. The 0299z patch
is intentionally separate (a one-hunk fixup to 0299); the parent may squash it into 0299's patch at landing
if preferred. Patch files in `zig/patches/**` were **not** touched (outside this lane's edit allowlist).

## 4. Consolidated staged hardware plan (operator runs all of it; nothing below was run here)

Fresh image = this tree's source + the 0200-0212 bundle (parked 0206/0207/0208 folded on) + 0297 + 0298's
fix applied to 0206 (exact edit in the 0a note: the kernel's two stores flip the top bit
(`keys[...] = u ^ 0x8000000000000000ULL`, `tmax[...] = top ^ 0x8000000000000000ULL`); the zig `foldKeys`
line writes `kitKey(x, i)`). Mirror `runs/fused-followup-20261121/campaign.sh` (`arm.sh` +
`tools/exact.py record/compare` + `probe.py`); record the live production panel first (no restart);
`campaign.sh`'s `restore` trap returns production at the end. One hardware window can carry Stage A→C and
the k280 window back-to-back, or split into two — the operator's call on idle time.

**Stage A — localize** (unfixed bundle + 0297; `TF_DS_KEYS_PROFILE=1`):
1. `bash runs/fused-followup-20261121/arm.sh <tag> <image> TF_DS_IDX_FUSED=1 TF_DS_KEYS_PROFILE=1`
2. `python3 -I tools/exact.py record/compare` → expect **DIFFERENT 4 of 39** (len_20000/30000/50000/64000).
3. Collect `key_profile` from both ranks → `fold` records with `width_min > 16384`, `width_max ≤ 65536`,
   `kit` records below. REVIEW.md F1 (deferred): as specced, a literal re-roll of 0206 will record `.kit`
   at every width (the parked patch predates `key_prof`); the re-roll spec should add one line at
   `round.zig`'s selection site — `key_prof.record(nb, if (keys_done) .fold else .kit)` — or Stage A step 3
   drops the class claim. Parent decides.

**Stage B — the fix's byte parity** (same image, 0298's fix folded into 0206; probe.py keys+tmax vs the kit):
1. Same arms → expect **IDENTICAL 39 of 39**, `key_profile` unchanged.
2. `python3 runs/fused-followup-20261121/probe.py <kit cubins> <new bin>` comparing `keys` **and** `tmax`
   byte-for-byte against the kit's `_score_keys` on identical score rows (mixed-sign tiles, all-negative
   with −inf/ties, non-negative with ±0/+inf, NaN):
   - **`keys`: 0 differing bytes** at widths 64/128/16,384/32,768/65,536 **plus ≥1 partial-tile width**
     (e.g. 20,000 = 312×64+32 — REVIEW.md F2: the failing widths have partial tiles and the fold's
     partial-tile rule vs the kit's is part of the inferred `tmax` rule).
   - **`tmax`: 0 differing bytes at every tile index incl. the partial last tile** — the decisive check of
     the **inferred** rule ("the kit's tmax = largest kit-encoded key in the consumer's signed order" =
     `top ^ flip`). Mixed-sign tiles discriminate: any differing byte refutes the rule → re-derive only the
     `tmax` store (e.g. signed max over kitKey); the `keys` fix stands regardless.
   - **Pre-fix sanity**: the same probe on the unfixed fold must differ by exactly the top bit on `keys` —
     proof the harness sees differences at all.
   - REVIEW.md F2 (deferred): reconcile `runs/fused-followup-20261121/RESULTS.md:194-196` first — it says
     the kit faults at `width % 4 != 0` yet lists 65,533 (1 mod 4); likely a typo for 65,532, but the probe's
     width filter depends on it.

**Stage C — the bundle gate** (the item's blocker): arms `all` (all four fused switches) and `allpdl`
(+ `TF_DS_PDL=1`) vs the same 39-case suite → expect **IDENTICAL**; `base`/`base2` controls as in the
follow-up run.

**Then the k280 window** via `runs/k280/window.sh` (0c's packaging): rsync the README's exact file lists
(including the updated `work.py`/`table_gpu.py`/`arms.sh`) to the head, then `passprof.sh → run.sh →
l2-run.sh → topk-run.sh` one at a time, each gated on `restored: validate rc=0` + `finished`. Production
idle before step 1; never `just sync-head` (this checkout's `.env` is three deploys stale). ≈3.5 h total.
Score with `table_gpu.py` / `gate_drafts.py` per the README; speed claims stay unverified until this runs.

## 5. Proposed Justfile recipe (NOT applied — `just` file untouched)

```just
# Run the k280 hardware window (runs/k280/README.md) end to end from the workstation: rsync the README's exact
# file list to the head (never sync-head: this checkout's .env is three deploys stale), then run passprof.sh ->
# run.sh -> l2-run.sh -> topk-run.sh on the head one at a time, each gated on its run.log's "restored: validate
# rc=0" verdict. Production must be idle before this starts. About 3.5 h.
k280-window:
    head={{head}} dir={{dir}} bash runs/k280/window.sh
```

(`head`/`dir` are the Justfile's existing globals; `window.sh` takes the same names as env overrides.)

## 6. Open decisions (parent)

1. **0298 landing vs re-rolled 0206.** 0298 commits the encoding spec (`kitKey`/`key_kit_flip` + pins) as
   dead code; the runtime half (the two kernel stores + the `foldKeys` line) rides the re-rolled 0206. If
   the parent prefers the fix wholly inside 0206's patch, fold `kitKey` in and drop 0298's commit (same
   content). Related: REVIEW.md F1 — the re-roll spec needs the one-line `key_prof` class fix
   (`if (keys_done) .fold else .kit`) or Stage A's class expectation changes; and the re-roll must fit the
   newer `fused_trace.idxArm` switch (`idx_small`/`idx_wide` write no keys: `keys_done` must stay false
   there, `scoreKeys` keeps running).
2. **The inferred `tmax` rule** (largest kit-encoded key in the consumer's signed order = fold's unsigned
   max column, flipped bit). Stage B's probe byte check on `tmax` confirms or refutes; if refuted, adjust
   only the `tmax` store. The `keys` fix stands regardless.
3. **Comparator orientation.** The alternative fix (make the consumer compare unsigned on the folded keys)
   also restores selection parity but changes the kit contract; 0298 fixes the encoding instead (reproduces
   the kit byte for byte). Say so and 0298 is replaced if the parent wants the comparator fix.
4. **The scoring edits left uncommitted** (`runs/k280/work.py`, `runs/k280/window.sh`,
   `runs/overnight-20261122/arms.sh`, `runs/overnight-20261122/table_gpu.py`) — commit or fold as the
   parent sees fit. Suggested subject if committed: `0299 runs: per-round distributions (work.py samples,
   table_gpu.py mean/median/p95/p99, arms.sh frames.txt)` — note committing `arms.sh` necessarily commits
   its pre-existing `pass_counts` edit (content preserved either way). `table_gpu.py` is untracked in the
   main repo's git (that is why earlier git diffs showed "no content diff").
5. **0299z placement** — keep as its own patch/commit, or squash into 0299 at landing (§3).
6. **REVIEW.md F6 provenance check**: `runs/suite-fused-next2-20261121/query128/bench.py` carries an
   unattributed +2/−2 working-tree edit (`embedded_fatbin(binary, name="fatbin_dsv41_torch")` + symbol
   match) that appears in none of the lane notes — plausibly the k280 build lane's. Confirm provenance
   before consolidating so it is not lost when staging `arms.sh`. Not touched here.

## 7. Follow-ups (record only; todo.md untouched per constraints)

- Re-roll parked 0206 with 0298's fix (+ F1's one-line `key_prof` class fix); run Stage B's probe
  key/tmax parity rows (incl. a partial-tile width per F2) before the window gate.
- Reconcile `RESULTS.md:194-196`'s `width % 4` fault rule (65,533 vs 65,532) before trusting the probe's
  width filter (F2).
- With 0217 in the series the fixed fold covers width ≤ 131,072: the exact suite should gain a case in
  (65,536, 131,072] (len_100000 exercises it once the fold engages there).
- 0297 could record `prunes()` (pruned vs direct consumer) per width if the window gate wants it.
- Any future patch-stack image rebuild (the `234f9080b326` recipe) must apply `0190z`'s model.zig clk-init
  hunk; 0299's pin test fails loudly if a build's source drops the init lines (but only if the suite runs
  on it).
- `window.sh`: optional stall timeout + idle pre-check if k280 windows ever run unattended (0c open items).
- Dropped (per REVIEW.md F5): the `table_gpu.py` "wall seconds reads the previous loop's `v`" fix — not a
  bug.

## 8. Constraints compliance / evidence index

- No hardware touched, no SSH, no docker against the pair, no `sync-head`, no long benchmarks, no GPU
  claims. Short local tests only (focused `zig test` runs + the one full suite + scratch wrapper checks).
- Not edited (per standing rules): `todo.md`, `continue.md`, `decisions.md`, `to-benchmark.md`,
  `best-runtime.md`, `zig/patches/**`, `start.sh`, `.env`. (The main repo's pre-existing modifications to
  `todo.md`/`continue.md`/`decisions.md`/`bench.py` were left exactly as found.)
- Never touched `~/Workspace/tfref-sched`.
- Evidence in this output dir: `buildtest.log` (full-suite capture), `patches/` (four exports),
  `REVIEW.md` (the review). Focused-test evidence: `clock_fold.zig` 3/3 (post-fix), `idx_fused.zig` 10/10,
  `key_prof.zig` 2/2 (transcript runs this session). Wrapper sanity: scratch failing build through the
  identical tee wrapper → named error + `Build Summary` + `BUILDTEST_EXIT=1`.
- No staged files: worktree `git diff --cached` empty at `c82ff51`; main repo staging area empty.
