# Review: fused-next 0213-0251 (read-only audit)

Round under review: `git log --oneline 3c54a9d..fused-next` in `~/Workspace/tfref-fused-next` (13 commits,
0213-0251). Evidence: `runs/fused-next-20261121/wp{0..4}/`, harness `tools/fused_bench/`. Rules: `BRIEF.md`.
Nothing was edited, committed, rebuilt into an image, or run on the GPU. Read-only git, file reads and greps only.

Method: the coordinator read the full diff and the evidence, then ran six parallel read-only reviewer
subagents (bit-exactness split three ways over 0213/0214/0215, 0240/0241/0242 and 0230/0231; concurrency and
memory safety; host-side call sites; claims versus evidence). Every blocker and should-fix below was
re-checked by the coordinator against the branch before it was kept; findings a subagent asserted but the
code did not support were dropped or demoted, and are noted where that happened.

Severity: **blocker** = can change a served byte, deadlock, race, OOB read or crash on a served shape.
**should-fix** = wrong on an unserved/edge shape, a broken fallback, or a claim the evidence does not
support. **note** = cosmetic or maintenance.

---

## Part 1 — coordinator-verified findings

These are the coordinator's own results, confirmed at the cited lines. Part 2 (appended) carries the
reviewer fan-out's findings.

### Should-fix

**S1. 0251's "prove engagement" test proves a copy of the dispatch, not the dispatch.** Confidence: high.
`zig/src/families/dsv41/fused_trace.zig:100-124` defines `idxArm`/`compressArm`/`keyArm`/`latentArm`/
`queryArm`, which re-derive each arm from the gate functions. `grep -rn "idxArm\|compressArm\|keyArm\|latentArm\|queryArm"`
matches **only** `fused_trace.zig` (lines 130-135, 174-182) — every match is inside the file's own tests.
The production call sites (`round.zig:1281`, `prompt.zig:1411`, `ops.zig:466`, `ops.zig:487`, `exact.zig`
launchers) call `fused_trace.note(...)` directly with their own inline conditions. So
`test "the served matrix selects the recorded arms"` asserts that the *model* agrees with a hard-coded table;
it cannot fail if a call site and the model diverge. The claim in wp4/RESULTS.md:143-166 ("walks the served
matrix ... through the same gate functions the call sites use and asserts the arm each shape takes") is
therefore too strong: it asserts the arm the *model* takes.
The divergence is not hypothetical — it already exists:
- `queryArm` (fused_trace.zig:118-121) returns `.query_pair` for every `rows >= 32`, but `exact.zig:291`
  additionally requires `heads % 8 == 0 and hd % 64 == 0`. For `heads = 12`, `rows = 32` the model says
  `query_pair` and the engine runs `query_fused`.
- `idxArm` (fused_trace.zig:106-109) returns `.idx_small` for `rows <= 7 and nbk >= 6144`, but
  `prompt.zig:1476-1483` (the pooled source layer) has **no** small-row branch: it calls `idxFused` or the
  unfused chain. At a prompt chunk with `m <= 7` and `nbk >= 6144` the model says `idx_small` and the engine
  runs `idx_fused`.
Fix: have the call sites call `idxArm`/`queryArm`/... themselves (one decision point), or assert on a
recorded trace from a real launch. As it stands, 0251's stated purpose — showing which arm ran — is served
by the `note()` calls, but the test that is supposed to guard them does not guard them.
(The trace itself is sound and off by default: `note()` is called behind `enabled()`, writes no bytes, and
`enabled()` caches one bool. `engagement-trace.txt` is the host test's own stdout, not a live round; wp4
says so honestly at RESULTS.md:263.)

**S2. 0216's "pooled is false" is a config-dependent argument, not a checked invariant.** Confidence: high
that it is unproven, medium that it can bite. `prompt.zig:1397-1407` returns early when
`0 < n_comp_end <= c.index_topk`, leaving `sh.pool` unset. The comment (prompt.zig:1401-1403) justifies this
with "a pool needs nbk > candidate_blocks", i.e. `n_comp_end > candidate_block * candidate_blocks`. For this
checkpoint that is `8 * 2048 = 16,384 > index_topk = 512` (`config.zig:266`, `tri_index.zig:176`), so
`nbk <= 64 < 2048` and `pooled` is false. But nothing asserts `candidate_block * candidate_blocks >=
index_topk`. On a checkpoint where the pool is smaller than `index_topk` (e.g. `candidate_blocks = 16`),
the bypass fires with `pooled == true`, `sh.pool` stays 0, and the next layer's `candidate_source < li` branch
hits `if (!sh.pool) return error.NoCandidatePool` (`prompt.zig:1494`). That is a hard failure, not a silent
wrong byte, but it is a config-dependent landmine introduced by a shortcut whose justification is arithmetic
on the served config only. Cheap fix: include `not (nbk > c.candidate_blocks)` in the bypass condition, or
assert the invariant in `config.zig`.

### Notes

**N1. The bypass's justification cites the wrong producer for the keyed layers.** Confidence: high
(doc-level). `prompt.zig:1399-1401` and wp1/RESULTS.md:255-258 justify the identity selection by pointing at
`tf_ds_topk_indices_kernel`'s ascending bitonic sort (`dsv41_ops.cu:190-211`) and its `cols < vis ? cols : -1`
masking (`dsv41_ops.cu:209-211`). That kernel is what the **non-keyed** arm runs. The **keyed** layers (before
the candidate source) run the kit's `topk_select_pruned` instead (`prompt.zig:1466-1468`). The bypass sits
*before* the `keyed` branch, so it replaces a kit-produced `cidx` on those layers too, and the ascending-order
argument is not stated for the kit producer. The claim is probably true, but it is the same class of statement
that 0206 got wrong. Verified: `iotaVis` (`dsv41_ops.cu:285-292`) does produce `j < vis[r] ? j : -1`, which
matches the engine producer exactly; the kit producer's order for `k == n` is asserted, not shown.

**N2. `index_topk == 512` makes the bypass bound and the decode bound agree — by coincidence of the config.**
Confidence: high. `prompt.zig:1404` uses `c.index_topk`; `round.zig:1225-1229` refuses `nb > 512` with
`notPorted` for the same identity. These agree only because `index_topk == 512`. If `index_topk` were raised,
the decode arm's 512 refusal would remain and the two identity paths would disagree. Worth a shared constant.

**N3. Graph capture keys are adequate for the new dispatch.** Confidence: high. `round.zig:1759` keys a
captured stretch on `{rows, bucket, stretch, mode}`. Every parameter the new dispatch reads is either in that
key (`rows` for 0215's `small_rows` and 0241's `pair_rows`), derived from it (`nbk` from `bucket`), or a
config constant (`d`, `heads`, `hd`, `rd`). `pdl` is process-constant (`ops.load`). No missing key found.

### Verified sound (recorded so the next round does not redo it)

- **0214's cutoff equivalence.** `tf_ds_pool_cut` scans only digits 7-4; the removed `tf_ds_pool_kth` scanned
  7,6,5,4,1,0. The two pick the same set for every input class. When the old scan broke early it was only
  because `want_s == 0` (the bin completed exactly), and its partial `kth` then admitted the whole bin — the
  same set the new `want == 0` path admits via `u >= prefix`. When `want_s != 0` the old scan narrowed the
  bin by column digits, i.e. kept the *lowest* columns; the new compaction keeps the lowest `want` tied
  columns in ascending scan order. Same set, same slots. `prefix_s/mask_s = 0xffff0000` is safe to 65,535
  columns, and `max_nbk` is 16,384. `k == nb` yields `prefix = mask = 0`, `want = 0`, `u >= 0` → every
  column picked, matching the old `kth = 0` path.
- **0213's vectorized phase 1.** `(ss & 3) == 0` plus `b * 8` a multiple of 8 makes every `float4` load
  16-byte aligned for all `r`; the `b * 8 + 8 <= width` guard keeps the partial last block on the scalar
  path; the comparison order (NaN wins, then greater) and the `last`-block `+inf` pin are unchanged.
- **0230's arithmetic.** The inline body's reduction `(w0+w2)+(w1+w3)` and `tf_ds_kit_squares4` are identical
  to the replaced `tf_ds_compress2_decode` (`dsv41_torch.cu:555-583`), and dropping the `tid < 128` guard is
  valid only because `exact.zig:228` launches the 128-wide kernel with exactly `d = 128` threads.
- **0240's byte arithmetic.** `tf_ds_scale_byte_bits` reproduces `max(amax, 6*2^-126)/6`'s biased exponent at
  the boundaries traced (mantissa `0x40`, bf16 subnormal and zero → 1, inf → 255); `tf_ds_recip_bits(255) =
  0.0` preserves the sign of a zero result; byte 254 is unreachable so the `(254 - byte) << 23` encoding
  never falls into the subnormal range.
- **0241's pair geometry.** `warpShapesOk` (`rope_store.zig:69-71`) already forces `d % 64 == 0`, `rd` even
  and `(d - rd)` even, so `i0 = chunk + 2*lane` is a genuine rotary pair inside one lane; the half-warp
  `__reduce_max_sync` masks (`0x0000ffff` / `0xffff0000`) are each internally consistent and cover exactly
  the kit's two 32-element scale groups; the `h >= H` early return happens before any warp primitive.
- **0231's ring mapping.** `(raw_pos0 + j - raw_first) % ring_len` equals `(start + j) % raw_rows`, the removed
  scatter's mapping; `ch.gpos`/`ch.groups` have no readers outside `prompt.zig`, so skipping their upload when
  both fused stores are taken is safe.
- **Kernel renames.** No stale `tf_ds_compress2x_norm_kernel` / `tf_ds_fused_rope_qd_kernel` references remain
  in the engine; the harness looks up `compress2x_norm128/512` and `fused_rope_qd_pair` (`tools/fused_bench/
  bench.py:711,844`, `chain.py:113,124`, `sass.py:22`).

---

## Part 2 — reviewer fan-out findings (verified by the coordinator before inclusion)

Six reviewers ran read-only. Where a reviewer asserted something the code did not support, it is demoted or
dropped and said so. The numeric check behind B1 was reproduced by the coordinator.

### Blockers

**B1. 0241's pair kernel rounds the odd rotary element in the opposite fma order from the kernel it replaces.
Confidence: high (proved in code and by numeric reproduction).** `zig/kernels/cuda/dsv41_torch.cu:975`
(`tf_ds_qd_pair_chunk`) computes `r1 = __fmaf_rn(v1, cs, __fmul_rn(v0, sn))`, i.e. `fma(xo, cs, xe*sn)`.
The one-element kernel at `:946` computes `r1 = __fmaf_rn(other, sn, __fmul_rn(v, cs))`, i.e.
`fma(xe, sn, xo*cs)`, and the file's own contract at `:699-700` states the kit's `_rope_heads` form is
`im = fma(xe, sn, xo*cs)` while the torch `ops.rope_` form is `im = fma(xo, cs, xe*sn)`. So the pair form
implements the *torch* contraction and the one-element form the *kit* contraction. They are not the same
function: `round(xo*cs + round(xe*sn))` versus `round(xe*sn + round(xo*cs))`. The comment "Both write the same
bytes" (`:931`) is false.
Reproduced: over 2,000,000 random bf16 `xe, xo` and fp32 `cs, sn`, the two forms differ in the final fp32 in
33.6% of cases and in the bf16-rounded value in 1.9e-5. A constructed group shows the served output byte
differing: `xe = -0.38867188, xo = -0.034179688, cs = 0.48039034, sn = -0.44551173` gives kit `0.125` and pair
`0.1875` for `out` (rd.iq4, read by the indexer at `round.zig:1252-1269`).
Affected served shapes: every launch that takes the pair geometry — decode R=32/48 and prompt chunks with
`n >= 32` (`exact.zig:291`). This is a served-byte divergence under BRIEF rule 4, so it is a blocker even
though the harness reported 0 differing bytes: the per-case probability of an *observable* difference is
roughly 0.25% (≈0.6 differing bf16 values per case, ×~0.4% chance each changes a code or a scale byte), so
~4% over the whole parity sweep. Production runs 40 such launches a round; it will fire.
Fix: `r1 = __fmaf_rn(v0, sn, __fmul_rn(v1, cs))`. Then re-run the query parity cases with a targeted input
(a group whose odd element is the maximum and sits on a bf16 tie point) rather than relying on random rows.

**B2. Two served-shape byte differences were observed and never explained. Confidence: high that it is
unexplained, medium that it is a kernel defect.** `wp2/SUPERVISOR.md:5-12`: `idx-small R=1 nb=8192` differed
from the reference by 3,984 bytes and `compress-decode R=32 D=512 mixed` by 3,707 bytes, new arm only
(`old_vs_ref: 0`) in one run where two harness processes ran concurrently while production served. Both are
*served decode shapes*, both are new arms (0215 and 0230). Investigation: 40/40 isolated repeats pass,
`compute-sanitizer` initcheck/racecheck/memcheck clean, six concurrent full runs pass. Cause unknown; wp2
declines to call it a harness flake but has no mechanism. Under rule 4 an unexplained new-only byte
difference on a served shape is a blocker until it has a mechanism. Cheapest closure: one run that hashes
each arm's inputs and output buffer addresses before launch (the signature — new-only, one arm, two
processes — is consistent with an address-dependent uninitialised read on the new arm, not with kernel
nondeterminism, which `new_vs_newdup: 0` already rules out).

### Should-fix

**S1. 0216's keyed-layer bypass replaces a kit kernel's output on an argument about a different kernel.
Confidence: high that it is unverified; medium that it is wrong.** The bypass (`prompt.zig:1397-1407`) sits
*before* the `keyed` branch, so it also applies to the layers before `candidate_source`. Those layers run
`tri_index.topkSelectPruned`; with `n_comp_end <= 512`, `prunes(512,512)` is false (`tri_index.zig:144-147`,
`n > topk_fused_max = 4096` required), so the full path routes to the kit's `_topk_sel` (`tri_index.zig:136`)
— not the engine's `tf_ds_topk_indices_kernel` that the comment (`prompt.zig:1399-1401`) and wp1/RESULTS.md:255
cite. The bypass's ascending-order argument therefore does not cover the producer it actually replaces. The
equivalence rests on the decode precedent (`round.zig:1225-1229` substitutes `iotaVis` for the engine's
selection at `nb <= 512`), which is served and pair-validated, but that precedent is for the *engine*
producer. This is the same class of statement that 0206 got wrong. Fix: gate the bypass to `li >=
c.candidate_source` (the non-keyed layers), or byte-check `_topk_sel`'s output against `iotaVis` at
`n_comp_end <= 512` on a keyed layer.

**S2. 0216 is default-on with no switch. Confidence: high.** The bypass condition is
`n_comp_end <= c.index_topk` and nothing else; there is no way to run the old path (`prompt.zig:1397`). BRIEF
rule 6 says "no new default-on behaviour ... every change rides an existing fused switch ... or a new opt-in
one; unfused fallbacks stay". Every other item in this series is behind an off-by-default switch; this one is
not. Fix: `and e.ops.idx_fused` (or a new opt-in) so the previous path stays reachable.

**S3. 0216's "pooled is false" is arithmetic on one config, not an invariant. Confidence: high that it is
unproven.** `pooled` requires `nbk > candidate_blocks`, i.e. `n_comp_end > candidate_block *
candidate_blocks = 8 * 2048 = 16,384 > index_topk = 512` (`config.zig:266`). On this checkpoint the bypass
therefore never coexists with a pool. Nothing asserts it: with `candidate_blocks = 16` the bypass fires with
`sh.pool` unset and the next layer's `candidate_source < li` branch hits `error.NoCandidatePool`
(`prompt.zig:1485`). Fix: add `not (nbk > c.candidate_blocks)` to the condition or assert the invariant.

**S4. 0251's trace statics are unsynchronised. Confidence: high (code), conditional on multi-threaded
callers.** `fused_trace.zig:42-44,55-63`: `seen`, `count`, `enabled_cache` are plain globals. Two threads that
both see `count == 127` both pass `if (count == capacity)`, both write `seen[127]`, both increment → `count ==
129`; `entries()` then returns `seen[0..129]` and the next `note()` writes `seen[129]`, one past the array.
Off by default, so no served byte changes. The pair's two ranks are separate processes, so the cross-rank case
does not arise; the exposure is a process that serves more than one lane host thread. Fix: a mutex, or an
atomic CAS with `if (idx >= capacity) return`.

**S5. 0251's "prove engagement" test does not exercise the call sites (coordinator S1, restated with the
reviewer's line evidence). Confidence: high.** `queryArm` (`fused_trace.zig:118-121`) returns `.query_pair`
for every `rows >= 32`, while `exact.zig:291` also requires `heads % 8 == 0 and hd % 64 == 0`; `heads = 12,
hd = 128, rows = 32` records `query_pair` but runs `query_fused`. `idxArm` returns `.idx_small` for `rows <= 7,
nbk >= 6144`, but the prompt pooled layer (`prompt.zig:1476-1483`) has no small-row branch. The model
functions are called only from the test (`fused_trace.zig:130-135,174-182`). Fix: have the call sites call the
model functions (one decision point) instead of duplicating the condition.

**S6. 0250's pre-wait weight read is correct only because of an unstated invariant. Confidence: high that it is
currently correct.** `dsv41_torch.cu:667-669` and `:858-860` read `w[threadIdx.x]` before
`griddepcontrol.wait`; the safety argument is that `lay.comp_norm` and `lay.idx_k_norm` are `L.plain` uploads
(`weights.zig:661,679`) that no kernel in the round writes. Nothing in code enforces it. If a future path
passes a kernel-produced norm weight, with `TF_DS_PDL=1` the kernel silently reads the previous round's bytes.
Fix: state "w must be a load-time weight, never a kernel output" on `exact.compress2xNorm` and
`exact.idxKeyFused`. (PDL is off in production, so this is latent.)

**S7. 0217's timing claim has no artifact, and wp4's `timing_clean` claim is false for the chain files.
Confidence: high.** The nb=8193/16384/16385 timings in `wp1/RESULTS.md` item 7 exist only in prose — no
`wp1/*.json` contains 16384 or 8193. And `wp4/RESULTS.md:3-5` states "`timing_clean: true, timing_retries: 0`
in each JSON", but of the twelve wp4 JSONs only the five parity/i2 files contain those keys; `wp4-chain.json`,
`wp4-a/b/d/e.json` — the files carrying the chain numbers — do not. Fix: attach the JSONs, or reword.

**S8. 0213 regresses the prompt rows outside its own IQR and does not dispatch per shape. Confidence: high on
the number.** `wp1/wp1-idx2b.json`: R=1024 old 580.572 (iqr 0.588) → new 583.426 (iqr 0.816); R=2048 old
1127.808 (iqr 3.246) → new 1132.619 (iqr 2.076). The deltas (+2.85, +4.81 µs) are 1.5-4.8× the IQR, so by
BRIEF's own standard they are resolvable, but wp1/RESULTS.md calls them run-to-run spread. 0213 has no
per-shape dispatch, so the keep/drop rule reads "drop". Small (0.4-0.5%, DRAM-bound rows), but the claim
should say "a 0.4-0.5% regression at the DRAM-bound prompt rows, accepted", not "not resolvable".

**S9. 0215's dispatch threshold sits in an unmeasured interval, and its comment is stale. Confidence: high.**
`idx_fused.zig:44 small_nbk = 6144`; the measurements behind it are nbk 2049, 4096 and 8192 only
(`wp1/wp1-small.json`, `wp1-small2.json`). At nbk 4096 the small form already loses at R=7 (8.668 vs fused
8.464), and nothing in [6144, 8191] was timed. nbk 16384 — reachable since 0217 raised `max_nbk` — was never
timed in the two-phase form. The comment at `idx_fused.zig:38-43` still calls 8192 "the pool's cap", which
stopped being true at 0217.

**S10. 0231's host half is mirrored in the harness, not executed. Confidence: high.** `bench.py:1116-1129`
re-implements the engine's branch, so `prompt.zig:1304` (`!fused or full == 0`), `:1319-1320` (the
`raw_first`/`raw_pos0`/`n` arguments) and `:1329-1343` (the conditional `ch.groups`/`ch.gpos` staging) are
never exercised end-to-end, while `wp2/RESULTS.md:213-227` says "Parity: exact, every live output" for the
source step. The coordinator read those three paths and they are correct, but the claim is stronger than the
evidence.

**S11. Prompt rows above 1024 are not covered for 0197/0198. Confidence: medium on servedness.** Every
prompt-source/key/latent parity case tops out at R=1024, while `prompt.zig:1098-1099` allows `n` up to 4096
(`vis_host: [4096]`). If a 2048- or 4096-row chunk is served, those kernels' bytes are unchecked.

**S12. `tf_ds_tile_prefix`'s `warp_end` zeroing is wrong for a block size that is not a multiple of 32.
Confidence: high (code), unreachable today.** `dsv41_ops.cu:749`: `threadIdx.x >= (blockDim.x >> 5)` uses the
floor warp count, so for `blockDim = 1023` the last partial warp's index is 31 and a lane of warp 0 zeroes
`warp_end[31]` concurrently with warp 31's lane-0 write. Both launchers use 1024 (`ops.zig:467,486`), where
`blockDim.x >> 5 == 32` makes the guard a no-op. Fix: `(blockDim.x + 31) >> 5`.

### Notes

- `dsv41_torch.cu:925-927`: the comment still says "grid.x = rows * H, block = 128" three lines above the text
  describing `grid = (rows, H)`; a flat launch would now be OOB at `r >= rows`. Stale, misleading.
- `ops.zig:487` records `width = nbk` for `idx_small` while `fused_trace.zig:24-26` documents `width` as
  columns and `idxArm` records columns, so a trace-enabled run reports one arm with two widths.
- `prompt.zig:1338`: the `full > 4096` `ChunkTooLong` guard now runs only when the host arrays are staged, so a
  chunk with `full > 4096` taking both fused stores no longer errors. Benign (nothing else reads those arrays),
  but it is a behaviour change.
- `dsv41_torch.cu:600-602` and `:15-19`: the "ptxas then issues both 64-bit slot loads ahead" and "+0.6 us"
  comments are SASS observations and chain measurements, not guarantees; a future CUDA version may schedule
  differently and cost the 0.14 µs, never the bits.
- `dsv41_torch.cu:667,669`: the literal `pdl_launch(1)`/`pdl_wait(1)` is safe only because of the enclosing
  `if (pdl)`; pass `pdl` for readability.
- 0217's "unchanged occupancy" is derived, not queried: a 1024-thread CTA is capped at 1 block/SM by the
  1536-thread/SM limit at every nbk, so the raise changes no residency — the right conclusion from the wrong
  reason ("shared doesn't bind"). A 512-thread CTA would reach 3 blocks/SM below nbk≈5,000.
- 0213's summary says "−6% to −20%" and "9.6-33%" for the small-row form; the measured small-row deltas are
  −33.7% (R=1) and −26.7% (R=7) vs old, so the lower bound is the nb=2049 case.
- `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101,376` is taken from wp1/RESULTS.md, not re-queried; the opt-in
  attribute is set at load on both kernels (`ops.zig:204-205`), so a smaller limit would fail at load even for
  runs that never exceed 48 KiB. One `cuDeviceGetAttribute` line in the harness log would close it.

### Not verified in this review

- No build, no GPU run: `zig build test` (794/796 baseline) was not re-run; the two pre-existing failures and
  the new 0251 tests are taken from the evidence.
- SASS and occupancy numbers were not re-derived; only the arithmetic and the dispatch logic were checked.
- Whether the pair's 39-case exact gate contains a short prompt that takes 0216's bypass is not established
  (the gate lives in the *previous* campaign, `runs/fused-followup-20261121/prod.exact.json`, 39 cases; no
  39-case artifact exists under `runs/fused-next-20261121/`).

---

## Verdict

**Not safe to benchmark on production as it stands.** One commit breaks bit-exactness on served shapes and
must be fixed first: 0241's pair kernel rounds the odd rotary element with the torch contraction instead of
the kit's, so `rd.iq4` can differ from the path it replaces at decode R=32/48 and on every prompt chunk with
`n >= 32` — a one-line fix (`r1 = __fmaf_rn(v0, sn, __fmul_rn(v1, cs))`) followed by a targeted parity case,
because the existing sweep's 0-differing-bytes result is not evidence of equality here, only of rarity
(~0.25% per launch, and production runs forty a round). Second, the previous round's two unreproduced
new-only byte differences on served decode shapes (`idx-small R=1 nb=8192`, `compress-decode R=32 D=512`) are
still unexplained and sit inside this series' 0215 and 0230; they need a mechanism or a hashed-input rerun
before the pair benchmark, not a "not a harness flake" note. With those two closed, the rest of the series is
sound: 0213/0214/0215's selection and compaction equivalence is proved rather than sampled, 0230/0231 are
algebraically identical to the paths they replace, 0240's E8M1/E8M0 byte arithmetic reproduces the kit
formula at every boundary traced, and 0250/0217/0215 pass an adversarial concurrency read with no race, OOB
or UB on a served shape. The one thing I would fix before calling the round finished even if it does not
block the benchmark is 0216: it is a default-on host shortcut with no switch, no harness byte check, and a
justification that cites the engine's `topk_indices` while the code path it actually replaces is the kit's
`_topk_sel` — gate it behind an existing switch and to the non-keyed layers, or byte-check `iotaVis` against
`_topk_sel` at `n_comp_end <= 512`, and the residual risk of a second 0206 drops to near zero. Benchmark with
the production switch set (PDL off, as deployed), and re-run the pair's 39-case gate — which no artifact in
this round's evidence shows was run — before trusting the reply hashes.
