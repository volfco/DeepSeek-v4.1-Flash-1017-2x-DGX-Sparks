## Harness review — h0270 (TF_DS_PROMPT_PICKS_ONCE)

Scope reviewed: `h0270/build.sh`, `h0270/src/harness.zig`, `h0270/src/prompt_picks_once.zig`, `run.log`, `run-negative-control.log`, cross-checked line-by-line against the worktree's `zig/src/families/dsv41/prompt.zig`, `prompt_picks_once.zig`, `model.zig`, `config.zig`, `tri.zig`. Read-only; no files modified. Note: this reviewer role has no shell tool, so I could not re-execute `build.sh`; verification is static analysis plus the two logs.

### Transcription cross-check (harness → real call sites) — all match

| Seam | harness.zig | prompt.zig (real) | Verdict |
|---|---|---|---|
| Pad predicate | `:127` `kk % 16 != 0 and (s.n > decode_rows or one_split)` | `:1133` identical | ✅ |
| ring_mode / one_split | `:125-126` | `:1076-1078` identical (constants verified: `ring_extra=16` prompt.zig:59, `window=128` config.zig:286, `decode_rows=16` tri.zig:10) | ✅ |
| fill args | `:138` `@memset(bytes[0 .. n*kp*8], 0xff)` | `:1141` `fill(e, ch.cidxp, 0xff, n * kp * 8)` (byte fill, prompt.zig:715) | ✅ |
| copyRows args | `:139-140` per-row `kk` i64, strides `kk`/`kp` | `:1142` `copyRows(e.s, ch.cidx, kk*8, ch.cidxp, kp*8, kk*8, n)` | ✅ |
| Reuse arm | `:130-136` (`held` only in else, both arms read cidxp) | `:1138-1145` (`padded = true` both arms; `.idx` at `:1162`) | ✅ |
| begin hook | `:81-88` `stale()` | `:975` | ✅ |
| indexer hook | `:107` `stale()` first, `sh_kk=kk`, kk==0 writes nothing | `:1499` first stmt; `sh.kk` written **only** in `indexer` (`:1506,:1522,:1624`) | ✅ |
| replayCut hook | `:93-105` conditional `stale()` on `first>0 or pre>0`, `first = max(start,replay)-start`, `n = n-first+pre`, `floor=replay` | `:2658-2680` exact same shape (`stale()` at `:2679` inside the same guard) | ✅ |
| Cut point & order | `runChunk`: at `cut_layer=20`, cut → indexer → attention | model.zig:1625 `li == c.layers / 2` before the layer's `attention` (which runs `indexer`, prompt.zig:1129); `rs.floor = replay` model.zig:1637 | ✅ |
| kk derivation | `runChunk`: `min(512, (start+n)/ratio(li))` | `:1123` `n_comp_end=(start+n)/r`, `:1544` `kk = min(index_topk, n_comp_end)` | ✅ |
| Config map | 40 layers, ratios 2@2..19 / 1@20..39, sources {2,8,14,20,24,28,32,36}, topk 512 | config.zig:287-290 | ✅ |

Slice fidelity verified: `Resume.shared: prompt.Shared` persists across `TF_DS_PREFILL_SLICE` slices (model.zig:79-86, reset only at chunk end/error), and decode rounds between slices use round.zig's own `rd.cidx` (round.zig:474) — never `ch.cidx/cidxp/picks`. The harness's "state crosses slice boundaries" claim (harness.zig:248-253) is faithful. Copied `h0270/src/prompt_picks_once.zig` is content-identical to the worktree's (compared in full).

### Hand-constant re-derivation (check d) — all re-computed and correct
38 compressed reads (layers 2–39), 8 indexer stale()s → 8 pads / 30 reuses / off 38+38; `want_pads {0,6,12,18,22,26,30,34}` = `li-2` for the 8 sources; banner (127, 63) at layer 3; replay=64 → n=63; case t: 3 fills, `events[18]` (layer 20 = 19th read); case u: n=135; case v: 2 fills, reuse at `[18]`; test 2's chunks (n=127/40/17) each give 8/30 (all kk ≢ 0 mod 16, n>16). Every constant checked out against config.zig's map.

### Findings

**S2-1 — build.sh's drift greps miss the pad-predicate seam.** `build.sh:14-19` guards the reuse condition (prompt.zig:1138), fill (:1141), copyRows (:1142), held (:1143) and the 3×`stale()` count, but **not** prompt.zig:1133 (`kk % 16 != 0 and (n > tri.decode_rows or one_split)`), the ring_mode/one_split formulas (:1076-1078), or model.zig:1625's cut point. The entire case matrix rests on the predicate; if it drifted, the harness would keep passing while no longer modeling the real pad block. Smallest fix: add greps for `if (kk % 16 != 0 and (n > tri.decode_rows or one_split))`, `const ring_mode = n <= ring_extra and !(replay_floor and below);`, `const one_split = !ring_mode and n <= tri.decode_rows;` (and optionally model.zig's `if (li == c.layers / 2)`).

**S3-1 — refPadRead inherits the tile width from the code under test.** `harness.zig:177-181` pads to `kp` columns, but `kp` is passed in from `picks.padWidth(kk)` (the module under test). If `padWidth` rounded to the wrong tile size, the byte-compare and the `-1` loop (harness.zig:163) would both still agree. The 16-column rule is pinned only by the module's own unit test (prompt_picks_once.zig:81-90, hand constants — correct). Smallest fix: compute `(kk + 15) / 16 * 16` inside refPadRead or assert `kp == (kk+15)/16*16` in checkPadded. (Formulation itself is independent: elementwise `if (c < kk) pick else -1` vs. the emulated memset + row-memcpy; no self-comparison.)

**S3-2 — test 4's `want_pad` is derived from the transcription.** `harness.zig:343-344` recomputes the same ring_mode/one_split formula as the emulated predicate (:125-127), so the sweep cannot detect a wrong pad predicate (tests 1–3's counts are hand constants and do cover the served map). Smallest fix: a hand corner table, e.g. (kk=16,n=17)→compact, (kk=17,n=17)→pad, (kk=17,n=16 plain)→compact, (kk=17,n=16 post-cut)→pad, (kk=64,n=127)→compact.

**S3-3 — negative control 2 shows the case matrix is insensitive to the shape check.** `run-negative-control.log`: with `Tracker.reusable` reduced to `t.holds`, only the module's tracker test fails; all four harness tests pass. Acceptable: I verified every writer of `sh.kk` (prompt.zig:1506/1522/1624, all inside `indexer` which stale()s at :1499) and of `ch.n`/`ch.start` (:973-975 begin, :2677-2679 replayCut) accompanies a `stale()`, so a shape change without an event is unreachable in the real flow — the shape check is defense-in-depth guarded by its unit test only.

**S3-4 — "committed tree" is the worktree as-is.** `build.sh:21-22` copies at run time (good — no stale vendored copy; verified identical today), but watchdog_diff shows uncommitted changes (prompt.zig +21/−3, ops.zig +4, dsv41.zig +2, prompt_picks_once.zig untracked). Fine for an in-flight patch, but the guard pins the worktree, not a commit.

**S3-5 — `replay_floor=false` untested.** `replay_floor_on = true` (harness.zig:27) models the served `TF_DS_REPLAY_FLOOR=1`; the transcription formula is flag-faithful, but the replay_floor=off arm (small below-floor chunk reads compact) has no case.

### Verified good (evidence)
- **(a)** refPadRead is a third, independent formulation (harness.zig:177-181); -1 semantics match BRIEF (modulo S3-1's width seam, pinned by hand-constant module tests).
- **(b)** No self-comparison: `checkPadded` (harness.zig:159-166) compares refPadRead output vs `cidxp[0..n*kp]` byte-for-byte at **every** padded read in both arms, plus a constant `-1` check over `[kk, kp)`.
- **(c)** Case matrix complete: cold 127 (:234), TF_DS_PREFILL_SLICE=5 + 3-chunk multi-chunk (:248-253, :269), replay cut mid-prompt (:288-296), cut-alone-forces-re-pad (same-count shift, synthetic map, :298-308), no-op cut keeps pad (:317-325), pick counts 1..64 exhaustive + 65/80/100/127/200/511/512 (:218-226, :329), kk=0 read-none arm (:366-369), both chunk classes (ring vs post-cut one_split).
- **(e)** `run.log`: "All 9 tests passed" (4 harness + 5 module tests — `zig test` runs the imported module's tests). `run-negative-control.log`: cut hook removed → `expected 3, found 2` at harness.zig:306 (the hook-only re-pad case); shape check removed → tracker test fails.
- **(f)** Module copied at run time from `$wt` before `zig test`; greps would catch drift at the guarded seams (see S2-1 for the gap).

Merge verdict: no S1; the harness demonstrably covers BRIEF-0270's matrix with an independent byte-for-byte reference and correct hand constants. The S2 is a robustness gap in the drift guard, not a defect in the run as recorded.

VERDICT: OK

*Skipped/not checked: could not re-execute `build.sh` (no shell in this role — logs used as evidence); graph-capture question from the Contract is out of harness scope. Risk to know: if prompt.zig's pad predicate or ring/one_split lines drift after this review, nothing in build.sh will fail (S2-1) — add those greps before relying on the harness long-term.*