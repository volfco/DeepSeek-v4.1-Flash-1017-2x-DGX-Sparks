# 0a closeout — TF_DS_IDX_FUSED selection divergence (lane tier0, run 2026-11-23)

Closeout of item 0a. Worktree `~/Workspace/tfref-tier0`, clean at `f863532` on top of `1421187`. Commits
**0297** `c4e227a` and **0298** `f863532` (from the earlier item-0a run; full analysis in
`…/outputs/0e41eea5-…/runs/lane-tier0-20261123/0a.md`) are unchanged — this closeout only resolves the open
test question the previous run's timeout cut off, re-runs the focused tests, and records the final test state.
No tree edits in this closeout; no commits made.

## 1. The open test question: the core/ stream+tokenizer failures in the full build run

**Conclusion: known flake class, confirmed. Not a deterministic failure. No defect for the review stage.**

The previous run's `zig build test` (seed `0x1c6da9fb`, log preserved at `/tmp/tier0-0a-buildtest.log`) had
three failing test binaries (the `failed command:` lines, binaries still in the worktree's `.zig-cache/o/`):

| binary | module | standalone result here |
|---|---|---|
| `cc6450482fb7000cd21ba2e94230e2cb/test` | lanes core (`zig/src/core/lanes/lanes.zig`; its `stream.test.*` names are `zig/src/core/lanes/stream.zig`) | **75/75 pass**, 7 runs |
| `6ab41acafe2ea0e7327fd3c6cc36f6ed/test` | tokenizer (`zig/src/core/tokenizer/tokenizer.zig`) | **23/23 pass**, 7 runs |
| `81c16bea24456b436b507951f25fa9b9/test` | dsv41 (`zig/src/families/dsv41/`) — the tree carrying 0297/0298 | **527/527 pass**, 1 run (delta check, see below) |

**Standalone re-run protocol (per-binary):**
- Run 1 — the full run's exact order seed, `--seed=0x1c6da9fb`: stream 75/75 OK, tokenizer 23/23 OK
  (`/tmp/closeout-stream-run1.log`, `/tmp/closeout-tok-run1.log`).
- Run 2 — fresh random seed: stream 75/75 OK, tokenizer 23/23 OK (`…-run2.log`).
- Timing: ~1 s (stream) / ~3 s (tokenizer) per run.
- Load re-runs — both binaries in parallel with `nproc+4` CPU-burner workers, 4 iterations (8 runs):
  all pass (75/75, 23/23 every time; `/tmp/closeout-*-load{1..4}.log`).

**Exact failing test names from the full build run: unrecoverable; none observed standalone.**
- The full-run log is truncated at the source: the build's error summary is cut mid-line
  (`+- run test w`, log lines 2/8/13) because the previous run's timeout killed the capture mid-write, and
  the test binaries ran under zig's `--listen=-` build protocol, which never streamed per-test failure names
  into the captured output. No fuller copy exists: the previous run's artifacts dir holds only `0a.md`, and
  all worker transcripts (`03ec300f…`, `97d5fcdd…`, `d96ee50d…`) contain the same truncated log.
- Across the 14 standalone runs above there were **zero failures, hence zero failing names to record**.
- The failure mode matches the standing known-flake description ("flaky core/ stream+tokenizer test binaries
  under a full build run") exactly: failure only under full-build parallel load, clean standalone, and both
  binaries are import-disjoint from the dsv41 changes (verified: zero `@import` of `families/dsv41` in
  `zig/src/core/lanes/**` and `zig/src/core/tokenizer/**`; 0297/0298 touch only `zig/src/families/dsv41/*`).
  The 4-pass-under-load runs are consistent with a load/timing flake rather than a load-reproducible defect,
  but note the loop only completed 4 of 10 planned iterations before the earlier run was interrupted.

**Delta check on the third binary (the one carrying 0297/0298):** the dsv41 module binary passes **527/527**
standalone (`/tmp/closeout-dsv41-run1.log`), including the 0297/0298 tests (`idx_fused.test.kitKey…`,
`idx_fused.test.the fold's flipped keys…` are inside it) and the standing known-flake names
(`round.test.TF_DS_EXPERT_HOT_CACHE: hotPick…` passes standalone here too — consistent with that being a
full-suite-load flake on this tree, per the standing list). So the full-run dsv41-binary failure is the known
pre-existing load-flake set, **no delta attributable to c4e227a/f863532**.

## 2. Focused tests for 0297/0298 (re-run this closeout, zig 0.17.0 `/home/colum/.local/zig-0.17.0/zig`)

```
$ cd ~/Workspace/tfref-tier0/zig && zig test src/families/dsv41/idx_fused.zig
… (10/10)
4/10 idx_fused.test.scoreKey matches the -0/+0 rule and orders scores like signed integers...OK
5/10 idx_fused.test.kitKey is _score_keys' encoding: 0.0 at column 0 is 0x00000000ffffffff, and the order is signed...OK
6/10 idx_fused.test.blockMaxima: NaN wins, the newest block is pinned, short blocks are padded -inf...OK
7/10 idx_fused.test.topCols picks the k largest by key, ties to the lower block, every masks the tail...OK
8/10 idx_fused.test.poolPick zeroes the mask, sets picked blocks above -inf, lists -1 otherwise...OK
9/10 idx_fused.test.key negative-column rule: a column at or past every is -1 in the list and not in the pool...OK
10/10 idx_fused.test.the fold's flipped keys change the pruned picks across the sign bit (the failing window's widths)...OK
All 10 tests passed.

$ zig test src/families/dsv41/key_prof.zig
2/2 key_prof.test.the record counts each class's selections and widths; the line prints one JSON object…OK
All 2 tests passed.
```
(key_prof's three `TF_DS_KEYS_PROFILE: expected 1 or 0` lines are the test's own env-parse error-path prints.)

## Final test state of c4e227a (0297) and f863532 (0298)

- Focused: `idx_fused.zig` 10/10 (incl. 0298's `kitKey` pin + 0297's flip-divergence localization test),
  `key_prof.zig` 2/2. Both green, re-run this closeout.
- Full-suite context (previous run's `zig build test`): 3 failing test binaries — lanes/stream, tokenizer,
  dsv41 — all **pass standalone** (75/75, 23/23, 527/527); all three match the standing known-flake class
  (fail under full-build load only). No new failure introduced by 0297/0298 (import-disjoint from the core/
  binaries; the dsv41 binary green standalone). Baseline before/after deltas: **none**.
- Both patches remain off by default (`TF_DS_KEYS_PROFILE` unset; 0298 is dead code until parked 0206 adopts
  `kitKey`). No reply-bit or kit-contract change in this tree.

## What changed (from item 0a; unchanged by this closeout)

| commit | files | what |
|---|---|---|
| `c4e227a` **0297** | `zig/src/families/dsv41/key_prof.zig` (new, 98 lines), `dsv41.zig` (+2), `model.zig` (+3), `round.zig` (+4), `lanes_main.zig` (+2), `idx_fused.zig` (+156) | TF_DS_KEYS_PROFILE (off by default): the pruned selection site records width + key-encoding class (`kit`/`fold`) per class on a `key_profile` line beside `prefill_profile_ms`; host-side localization tests. |
| `f863532` **0298** | `zig/src/families/dsv41/idx_fused.zig` (+36/−7) | `key_kit_flip` + `kitKey` (the kit's `_score_keys` encoding) + pin tests. No runtime caller in this tree — the fold lands with parked 0206 and must write these. |

## Evidence (this closeout)

- `/tmp/closeout-stream-run{1,2}.log`, `/tmp/closeout-tok-run{1,2}.log` — standalone runs 1–2 (75/75, 23/23 ×4).
- `/tmp/closeout-stream-load{1..4}.log`, `/tmp/closeout-tok-load{1..4}.log` — 4 load iterations (8 runs, all green).
- `/tmp/closeout-dsv41-run1.log` — dsv41 module binary 527/527 standalone (delta check).
- Focused test output above (re-run in this closeout).
- `/tmp/tier0-0a-buildtest.log` — the previous run's full `zig build test` output (truncated at the error
  summary by the timeout kill; the three `failed command:` lines identify the binaries by hash).
- Import-disjoint proof: `grep -rn '@import("…dsv41\|families/dsv41' zig/src/core/lanes/ zig/src/core/tokenizer/`
  → zero matches (only comment strings mention dsv41).

## Staged GPU plan for the operator's window (NOT run here; the pair serves live traffic)

Fresh image = this tree's source + the 0200-0212 bundle (parked 0206/0207/0208 folded on) + 0297 + 0298's
fix applied to 0206's two stores (`keys[...] = u ^ 0x8000000000000000ULL`, `tmax[...] = top ^ 0x8000…ULL`;
the zig `foldKeys` line writes `kitKey(x, i)` instead of `scoreKey(x, i)`; exact edit in 0a.md). Mirror
`runs/fused-followup-20261121/campaign.sh` (`arm.sh` + `tools/exact.py record/compare` + `probe.py`); record
the live production panel first (no restart); `campaign.sh`'s `restore` trap returns production at the end.

- **Stage A — localize** (unfixed bundle + 0297):
  1. `bash runs/fused-followup-20261121/arm.sh <tag> <image> TF_DS_IDX_FUSED=1 TF_DS_KEYS_PROFILE=1`
  2. `python3 -I tools/exact.py record --url $url --like <prod panel> --label idx --out …` + `compare` →
     expect **DIFFERENT 4 of 39** (len_20000/30000/50000/64000).
  3. Collect `key_profile` from both ranks → expect `fold` records with `width_min > 16384`, `width_max ≤ 65536`,
     `kit` records at the shorter widths.
- **Stage B — the fix's bit identity** (same image, 0298's fix folded into 0206): same arms → expect
  **IDENTICAL 39 of 39**, `key_profile` unchanged; `python3 runs/fused-followup-20261121/probe.py <kit cubins>
  <new bin>` with key-fold rows at widths 64/128/16384/32768/65536 comparing `keys` **and** `tmax`
  byte-for-byte vs the kit's `_score_keys` (multiples of 4 only; the kit faults at `width % 4 != 0`) →
  0 differing bytes.
- **Stage C — bundle gate** (the item's blocker): arms `all` (all four fused switches) and `allpdl`
  (+ `TF_DS_PDL=1`) vs the same 39-case suite → expect **IDENTICAL**; `base`/`base2` controls as in the follow-up run.

## Open decisions (unchanged from 0a.md; parent consolidates)

1. **Where 0298's runtime half lands** — the exact two-store edit for re-rolled 0206 is in 0a.md; alternatively
   fold `kitKey` into 0206's patch and drop 0298's commit (same content).
2. **The `tmax` rule is inferred** (largest kit-encoded key in the consumer's signed order = fold's unsigned
   max column, flipped bit); Stage B's probe.py byte check on `tmax` confirms or refutes. The `keys` fix stands
   regardless.
3. **Comparator orientation** — the alternative fix (consumer compares unsigned) also restores parity but
   changes the kit contract; 0298 fixes the encoding instead (reproduces the kit byte for byte).
4. **len→width map** (`width == len`) not statically pinned; 0297's `key_profile` width ranges confirm on the window.

## Follow-ups (todo.md candidates; parent consolidates)

- Re-roll parked 0206 with 0298's fix; run probe.py key/tmax parity rows before the window gate.
- With 0217 in the series the fixed fold covers width ≤ 131,072: the exact suite should gain a case in
  (65,536, 131,072] (len_100000 already exercises it once the fold engages there).
- 0297 could also record `prunes()` (pruned vs direct consumer) per width if the window gate wants it.
- For the next full-suite run: capture failing test names immediately (`zig build test 2>&1 | tee` with the
  tail polled), since the build protocol hides per-test names and a timeout kill truncates the error summary.

## File:line map (this tree, for later lane stages)

- `zig/src/families/dsv41/idx_fused.zig` — `max_k`=2048 / `max_nbk`=16384 (:47) / `small_rows`=7 /
  `small_nbk`=6144 (:56); `scoreKey` (:73, the flip is its `^ 0x8000…` return); `key_kit_flip` (:89) +
  `kitKey` (:93) [0298]; `topCols` (:120); localization helpers/tests at file end [0297].
- `zig/src/families/dsv41/round.zig` — candidate-source arm: `indexScore` (:1326), pool gate
  `nb > c.candidate_blocks * cb` (:1328), `fused_trace.idxArm` switch (:1334), `.idx_fused` call (:1352),
  `key_prof.record(nb, .kit)` (:1366) [0297], kit keys + pruned selection (:1368-1372). The other
  key-consuming arm: `indexKeys`/`topkSelectPruned` (:1317-1319).
- `zig/src/families/dsv41/tri_index.zig` — `tile`=64 (:14), `topk_fused_max`=4096 (:16), `TopK` (:138),
  `prunes` (:144), `topkSelectPruned` (:153), `scoreKeys`=`_score_keys` (:165).
- `zig/src/families/dsv41/fused_trace.zig` — `idxArm` (:120).
- `zig/src/families/dsv41/config.zig` — `index_topk`=512, `candidate_blocks`=2048, `candidate_block`=8 (:266).
- `zig/patches-parked/0206-dsv41-tf-ds-0195-key-fold.patch` (main repo) — the fold: kernel stores
  (lines 56-57/73), `foldKeys`, `ops.zig` `idxFused` keys/tmax args, `round.zig` `keys_done` (lines 197-222).
- Key-profile plumbing: `key_prof.zig` (new), `model.zig` enable (after the TF_DS_PREFILL_PROFILE line),
  `lanes_main.zig` printers (both profile sites, beside `prefill_prof.print`).
