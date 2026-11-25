All verification is complete. Here is the review.

## Review

### Correct (with evidence)

**1. Off-path launch/byte identity — verified for every changed call.** The full arm (`draft.zig:700-703`) is verbatim the pre-diff code. Each changed call reduces to the old value when `scored == false` (`lrow = n = dr.bn`, `draft.zig:692`):

| Call | Old | New | Off-path value |
|---|---|---|---|
| `keyedNoise` stride (NoiseTap) `draft.zig:160` | `dr.bn * lw` | `rs * lw` | `rs = lrow = n = dr.bn` — identical (bn is set in `pass()` `draft.zig:580` and unchanged during `body`) |
| `specClean` stride (SpecTap) `draft.zig:182` | `dr.n * lw` | `rs * lw` | `rs = dr.bn`; SpecTap only runs with `dr.sq != 0` ⇒ `TF_DS_SPEC_SAMPLE` on, and `Drafter.init` refuses spec with `n_most != n` (`draft.zig:359-362` `error.DraftBlockWithSpecSample`), so `steps ≤ n_most = dr.n` ⇒ `bn = max(dr.n, steps) = dr.n` — identical |
| `specClean` clean stride `draft.zig:182` | `dr.n * dr.sq_w` | unchanged | sq keeps its full layout; lanes reads it at `i * d.n + j` (`lanes.zig:1637` `plan.add(i * d.n + j, …)`) — correct under both layouts since spec ⇔ `dr.bn == dr.n` |
| `keyedNoise` (SpecTap) `draft.zig:183` | `dr.n * lw` | `rs * lw` | same as specClean |
| `stepsTapped` `block` `draft.zig:716` | `n` | `lrow` | identical (`n = dr.bn`) |
| `confPack` `n` `draft.zig:725` | `n` | `lrow` | identical |
| `toF32` xc offset `draft.zig:729` | `i * n * d * 2` | `i * lrow * d * 2` | identical |

No other launch arg changed; `head_scored_noted`/`scored()` are host-only. A malformed `TF_DS_DRAFT_HEAD_SCORED` refuses startup — that matches the established switch convention, not a deviation.

**2. Reader completeness — every reader found and checked (list with row indexing):**
- **Markov `_score`** (`tri_markov.zig:128-133`): `LG = local + 4*step*lw`, `l_row = block*lw` with `block = lrow` → row `step + stream*lrow` = `readRow(steps, i, s)` packed / `i*n+s` full. ✓
- **NoiseTap** (`draft.zig:160`, kernel `dsv41_ops.cu:1421-1435`): `logits + blockIdx.y * l_row` with `l_row = rs*lw` and base row `step` → row `step + stream*rs`. ✓
- **SpecTap** (`draft.zig:182-183`, kernel `dsv41_ops.cu:1446-1458`): same logits stride; `clean` rows keep `dr.n * dr.sq_w` matching lanes' `SpecPlan` rows (`lanes.zig:489-503` `base = rows[k]*sq_w`, rows added as `stream*d.n+step` at `lanes.zig:1637`). `tok`/`stage` indexing uses `ts`/`mk.cols` — unchanged layouts. ✓
- **Confidence, `TF_DS_CONF_PACK`** (`draft.zig:725`; kernel `dsv41_ops.cu:1837-1859`): `n` is used *only* as the xc stream stride `xc + (i*n + s)*d` (`dsv41_ops.cu:1846`); `ids` uses `steps+1` and `embs` is packed `j = i*steps+s` — passing `lrow` is correct and the ids/embs rows are untouched. ✓
- **Confidence, else-chain** (`draft.zig:727-732`): `toF32` reads `steps*d` contiguous from `dr.xc + i*lrow*d*2` — the first `steps` rows of a stream are contiguous in *both* layouts; `xcf`/`conf_in`/`conf` all use packed `N*steps` row counts that were already packed before the diff. ✓
- **Probe `.local`** (`draft.zig:705`, checker at `layer_main.zig:637`): `probe != null` forces `scored = false` (`draft_head_scored.zig:51`), so a checked pass sees the recording's exact buffer. ✓
- `dr.xn` has exactly one reader (the head GEMM). `MarkovGather`, `dr.out`, `dr.embs`, `dr.msc.stage` use layouts the diff does not touch. The pre-head probe points (`.stages_h`/`.stages_pre`) read `h`/`pre`, which the scored path does not repack. No missing readers found.

**3. Bit-identity of the head GEMM — claim holds at every level reachable from this repo.** The head is EXL3: `weights.zig:727-729` calls `L.linear("head", vr, …)` with `cols` non-null (`plan.zig:82-84` returns `[2]usize`), skipping the plain-BF16 shortcut (`weights.zig:476-482`) — so it never reaches the cuBLAS `plainRows` path, whose own comment (`prompt.zig:734-737`) warns cuBLAS rounding *does* follow the GEMM shape. For glinear: kernel template `glinear[k2][codebook][wk]` is chosen by layer shape only (`exl3_linear.zig:181, 206-207`), grid `Σ(n/128)*sk` (`:213`) and block `wk*32` (`:215`) are M-independent, `plan(k, planN)` is M-independent (`:23-42`, "a layer keeps one reduction for every row count", `:15-16`). The only M-dependent parameters are `smem = wk*min(m,8)*128*4` (`:214`), `rotMany`'s grid (`:172`), and Z partial *addresses* (`:201`) — these vary row tiling/scheduling, not a row's K-reduction order. Nothing per-row varies with M at the dispatch level; the per-row invariance *inside* the served cubin rests on the module docs ("K ranges and reduction order" per layer, `exl3_linear.zig:1-3`) since `linear_grouped.cu` is not in the repo — that's as far as this codebase can prove.

**4. Graph capture.** `PassGraphs.Key = {streams, steps, keyed}` (`draft.zig:222`); `scored = f(switch, probe==null, e.two==null, steps, n=blockRows(steps))` with `dr.n`/`dr.n_most` fixed at init — constant per key. `body` is reached only via `pass()`/`PassGraphs.run` with `dr.bn = blockRows(steps)` set first (`draft.zig:580`) and asserted (`draft.zig:652`). The N collapse launches are a function of `streams` (in the key). No new host interaction.

**5. Zig correctness.** No compile issues found: `u64 + usize` offsets match pre-existing patterns; `i*n*c.hc*d*2` (i<16, n≤8) can't overflow `usize`. `scored()`: `steps=0` → false (and `pass()` rejects it, `draft.zig:576`); `steps>n` unreachable (`n = max(dr.n, steps)`, asserted `draft.zig:652`); `steps==n` → full path, incl. the widened `TF_DS_DRAFT_BLOCK` case (test covers `(6,6)`). `draft_head_scored.zig`'s asserts (`srcRow`/`packRow`) are only reachable from its own test, whose loops respect them — cannot fire in valid use. Test math verified: 72 cases / 60 shallow is what `N 1..4, n 5..7, steps 1..n` yields.

**6. Switch convention.** `env_name`/`parse → ?bool`/`fromText → error{BadDraftHeadScored}` is character-for-character the `idx_fused.zig:12-25` pattern (also `conf_pack.zig`), and the `ops.zig` wiring mirrors `compress2_norm`/`rope_store` (bool, read at load). No deviation.

**Report items the brief asked for** (for the parent's report):
- **Served policy engagement (`TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2`)**: `passDrafts(asked, base=5, kept, wide_kept=2, room)` (`lanes.zig:2203-2206`) runs `asked` when `asked ≤ 5`, widens to 6 only when `kept ≥ 2 and room > 5`. Block rows `n = max(5, steps)` (`draft.zig:558-560`). The on path engages iff `steps < n` iff **steps ≤ 4**; steps = 5 runs the full arm at n=5 and the widened steps=6 pass runs the full arm at n=6.
- **Brief's own arithmetic checks out**: 4 streams × 3 steps = 12 rows vs 20; 8 rows × 64,640 cols × 4 B = 2,068,480 B = 1.97 MiB.

### Findings

- **Finding (P1, brief compliance): the `h0265/` harness is missing.** BRIEF-0265's "Harness" section requires a host emulation of the row selection/packing *against an independent reference* over ≥114 layouts including 64 shallow passes. There is no `h0265/` anywhere in the worktree, and the only check is `draft_head_scored.zig:82-102`, which (a) covers 72 layouts / 60 shallow (below the brief's floor), and (b) compares the module's *own* formulas to each other (`packRow` vs `readRow` is `i*steps+s` vs `s+i*steps`; `srcRow` vs `i*n+s` is its own definition) — it never emulates the per-stream collapse gather (the `h`/`pre` byte offsets and packed `xc` writes at `draft.zig:696`) nor a value round-trip through each reader's stride against an independent reference. Smallest fix: extend the test to gather values at `srcRow` from a filled buffer via the exact byte-offset formulas the call site uses, pack, and read back through `readRow` with each reader's stride — or add the `h0265/` harness and note the deviation in the report.

- **Finding (P2, note): `_score`'s `l_seg` changes value on the on path** (`N*steps*nc` vs `N*n*nc`, `tri_markov.zig:128`). It is inert in the served geometry because `tri_markov.init` always yields `segs == 1` (`tri_markov.zig:71`: `segs = if (split) 1 else world`, `split = world > 1` → segs is 1 for world 1 or 2), so the segment-offset term is multiplied by `blockIdx.y == 0`; the 2D case (`lg_w != 0`) is excluded from the on path. If Markov geometry ever gains `segs > 1`, this must be re-derived. The `_score` kernel is a served-build binary pinned by `tri.conform` fixtures (not in the repo), so this is a structural argument, not a source proof.

- **Finding (P2, note): SpecTap's off-path identity is only structural via a mutual exclusion.** `draft.zig:182` now passes `dr.bn` (through `rs`) where it passed `dr.n`. Byte-identical today only because `Drafter.init` (`draft.zig:359-362`) rejects spec with `TF_DS_DRAFT_BLOCK > base`, forcing `dr.bn == dr.n` whenever SpecTap runs. If that exclusion is ever relaxed, the off path silently changes. One line (`rs = if (scored) steps else dr.n`) or a comment at the init check would make the dependency explicit.

### Merge verdict: **OK with notes**

No correctness blockers: off-path identity, reader indexing, capture constancy, and the M-invariance argument all hold with the evidence above. Before release I'd want the P1 harness gap closed (or explicitly waived in the report with the 72/60-vs-114/64 deviation recorded); the two P2s are report-only.

Skipped/not checked: internals of the precompiled `_collapse`/`_rmsnorm`/`_score` kernels and the `glinear` cubin (verified via launch args, wrapper docs, and conformance fixtures only); no builds or tests were run — the supervisor should run the dsv41 test block (`zig build test` or the `dsv41` suite incl. `refAll(draft_head_scored)`) to confirm compilation.