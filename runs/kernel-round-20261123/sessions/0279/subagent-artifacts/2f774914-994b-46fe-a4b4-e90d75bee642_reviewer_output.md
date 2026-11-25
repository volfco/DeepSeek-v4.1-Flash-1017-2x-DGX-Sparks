All verification complete. Here is the review.

## Review

### Correct (verified from source)

**1. Off-path parity — clean.**
- `idxArm` (zig/src/families/dsv41/fused_trace.zig:120–123) with `wide=false`: `wideOk` short-circuits on its first conjunct `wide and` (idx_pick_wide.zig:37), so arm selection is bit-identical to base in every case.
- Function lookup is gated: `idx_pick_wide_fn = if (idx_pick_wide_on) try fm.function(...)` (ops.zig:224); `fromText(null)` returns false silently (idx_pick_wide.zig:21–22). No new launch, log, or lookup when off.
- round.zig/prompt.zig pass `e.ops.idx_pick_wide` (default `false`, ops.zig:140); the `.idx_wide` prong (round.zig:1340) is unreachable when off; the banner var has no side effects at rest.
- The only base-code edit is `tf_ds_idx_select`'s `float*` → `const float*` (dsv41_ops.cu:1081). Verified safe: `tf_ds_idx_select` only reads `maxima`, and its only callee `tf_ds_pool_cut` already takes `const float*` (dsv41_ops.cu:974). Both existing callers (fused :1170, pick :1183) pass mutable pointers — implicit promotion, same codegen for loads, no launch/arg change.

**2. Wide-arm correctness — structurally sound.**
- `tf_ds_idx_pick_wide_kernel` (dsv41_ops.cu:1189–1197) calls the *same* `tf_ds_idx_select(bmax + (long long)r*bs, nbk, k, nullptr, 0, cblk, r, every[r])` as `tf_ds_idx_pick_kernel` (:1183) after verbatim shared staging — staging changes no values, so outputs are identical at any nbk, including ≤ 16,384 (where wideOk refuses it anyway).
- The nbk-dependent bound checks out: `tf_ds_score_key`'s low word is `0xFFFFFFFF - j` (dsv41_ops.cu:136); columns 0–65,535 have inverted-high-16 `0xFFFF`, column 65,536 has `0xFFFE`, so the `0xffff0000` radix prefix (:1090) covers exactly nbk ≤ 65,536. The new test (idx_pick_wide.zig:63–70) verifies this against `idx_fused.scoreKey`, which matches the CUDA key formula exactly (idx_fused.zig:73–77).
- Unfused-parity at nbk ∈ (16384, 65536]: same key function → same k-set (unique keys, ties to lower column); ascending compaction with in-place `-1` for `col ≥ every || value ≤ ninf` matches `tf_ds_topk_indices`' vis-masking + `tf_ds_pool_pick_kernel`'s `> ninf` rule (dsv41_ops.cu:940–956). This inherits the pre-existing idx_fused-vs-unfused equivalence claim; the diff adds no new divergence.
- `rd.bmax` sizing is not a new risk: the wide arm's `blockMax(..., rd.bmax, nbk, R)` demand is identical to the unfused arm's (round.zig:1350), and the allocation `R * (nb_max/candidate_block + 1) * 4` (round.zig:471) already covers it.

**3. Gate completeness — consistent; no wideOk-true shape can hit `BadIdxPickWideArgs`.**
- `wideOk` (idx_pick_wide.zig:36–41) implies every `idxPickWide` check (ops.zig:569–571): `nbk > 16384 ∧ ≤ 65536` matches the wrapper's refusal window; `k ≤ max_k ∧ ≤ nbk` both; `width ≤ maxInt(c_int)` both; `nbk == 1+(width-1)/bsize` with `bsize ≥ 1` implies `nbk ≤ width`; at the only call site (round.zig:1345) `bs == nbk` so `bs < nbk` never fires. The wrapper is deliberately laxer (no rows/bsize checks) and its doc says so.
- `.idx_wide` is impossible in the prompt arm (`allow_small=false` short-circuits wideOk) — prompt.zig:1581 unaffected.

**4. CUDA conventions — match.** `pdl_enter(pdl);` first statement (:1194); `extern "C" __global__ __launch_bounds__(1024)`; `r = blockIdx.x`; arg order `(bmax, bs:longlong, nbk:int, k:int, every, cblk, pdl:int)` matches the host adds (ops.zig:573–579) exactly and mirrors idxPick↔tf_ds_idx_pick_kernel; launch is grid=`rows`, block=1024, no `.shared`, and the kernel uses no dynamic shared (only `tf_ds_idx_select`'s static `__shared__`, legal).

**5. Zig call sites — all consistent.** All 13 `idxArm` call sites take the new 8-arg form (fused_trace.zig tests :193, 238, 244–245, 257–269; prompt.zig:1581; round.zig:1332). The round.zig switch has `else => unreachable`, so the new `Arm` variant compiles; no other exhaustive `Arm` switch exists (`note()` uses `@tagName`); `Ops` fields have defaults so no other struct literal breaks.

### Findings (all P2, report-only)

- **P2 — doc-comment misattachment, round.zig:1115–1120.** The new `// banner` comment + `var idx_pick_wide_noted` were inserted *between* `attention`'s `///` doc comment and `fn attention`. The `///` now documents the atomic var; `attention` loses its doc. Compiles; smallest fix: move the var (and its `//` comment) above the `///` block.
- **P2 — module-registration convention gap, dsv41.zig:56/125.** Every sibling switch module (`idx_fused`, `compress2_norm`, `rope_store`, `conf_pack`, `expert_prune`) is `pub const`-exported and `refAll`'d; `idx_pick_wide.zig` is neither. Its tests still run (transitively imported via ops/fused_trace) and its decls are analyzed on use, so this is convention/visibility only. Smallest fix: add the export + `refAll(idx_pick_wide)`.
- **P2 — stale comment contradicts the new bound, dsv41_ops.cu:1090.** "(inverted columns below 8192 have two constant high bytes)" is wrong — the constant-high-byte range is 65,536 columns (per :136 and the new kernel's own comment :1185–1187). Pre-existing text, but the diff makes it load-bearing documentation; one-word fix avoids a future misread of the bound.
- **P2 — scope interpretation worth a decisions.md entry.** With `TF_DS_IDX_FUSED` off and `TF_DS_IDX_PICK_WIDE` on, the wide arm still engages (asserted at fused_trace.zig:265). The brief frames the change as "the small-row form's" second launch — the small-row form being TF_DS_IDX_FUSED's. The independent-switch reading is defensible and mechanically sound, but it is a broader behavior change than the literal brief text; log the decision.

**No P0/P1.** Note: `dsv41_ops.cu` must be recompiled and `zig build test` run to confirm compile/test passage (I cannot run commands); the tests to run are `fused_trace.zig`'s arm-matrix tests and `idx_pick_wide.zig`'s three tests.

**Merge verdict: OK with notes.**

*Skipped/unchecked: GPU runtime behavior (radix timing, L2 residency claims), byte-level SASS comparison of the const-qualified rebuild, and the brief's served-config reachability report (parent scope). Known risk: the unfused-equivalence at nbk ∈ (16384, 65536] is verified structurally, not by execution — the brief's CPU-emulation harness is the intended proof.*