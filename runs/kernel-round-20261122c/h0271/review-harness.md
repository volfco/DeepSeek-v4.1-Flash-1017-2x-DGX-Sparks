## Review — h0271 CPU-emulation harness (kernel patch 0271, `tf_ds_resume_assemble_kernel`)

**Verdict: OK with notes** (no S1 blockers; one S2 fill-state fragility). Note: I have no write-capable tool in this session, so the review is returned here for the runtime to persist to `/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/kernel-round-20261122c/h0271/review-harness.md`.

### 1. Extraction fidelity — OK
- `splice.py` extracts `k = src[start:end]` verbatim (no transformation; mutations are explicit single-string replacements with `assert k.count(want) == 1`).
- Worktree `zig/kernels/cuda/dsv41_ops.cu` lines 1196–1220 (start marker `// TF_DS_RESUME_ASSEMBLE (resume_assemble.zig)` → end marker `// A greedy row's token as sampling.argmax takes it on the host`) read and compared side-by-side with `fused_kernel_current.inc`: identical comment block, signature, and body, including `p % window`, `first + i - pre`, `pos[i] = (long long)p`.
- Marker uniqueness verified by grep: start marker occurs exactly once (line 1196), end marker exactly once (line 1221). `extern "C"` count-1 assert guards a truncated window.

### 2. Reference independence — OK
- `ref_assemble`/`ref_restore` (`fused_run.cpp:35–52`) implement the brief's *copy sequence*: one bulk memcpy of fresh rows to out rows `[pre, n)` from `fresh[first…]`, tail rows restored by a two-segment ring copy (`slot = p0 % window`, `n1 = min(rows, window − slot)`, two memcpys) per field, positions `replay + i` filled on the host — i.e., replayCut + tailRowsCopy's segment split, structurally different from the kernel's per-row `(replay + i) % window` slot selection. An independently plausible wrong kernel (wrong slot, wrong boundary at `i < pre`, wrong fresh index, wrong positions) diverges from this formulation; the three mutants confirm sensitivity.

### 3. Emulation correctness — OK
- Kernel body (verified by read) has no `__shared__` (also asserted in `splice.py`), no barriers, no atomics. Each block writes one row (`blockIdx.x = i`); threads write disjoint words `w = threadIdx.x + k·blockDim.x`; `pos[i]` written only by thread 0; reads come only from tail/fresh buffers the kernel never writes. Every interleaving is therefore equivalent to any other — sequential per-thread execution is exact, no race is missed and none is falsely flagged. blockDim=256 in the harness is semantics-neutral (the loop strides `blockDim.x`).

### 4. Matrix coverage — OK
Enumerated from `main()`:
- Group 1 (window 128): 16 pre × 9 kept × 3 first × 5 mod = **2160**
- Group 2 (long appends, window 128): 3 × 2 × {256, 2048-kept} × 2 = **24** (2048 = served prompt chunk)
- Group 3 (16-byte rows, window 128): 4 × 2 × 2 × 2 = **32**
- Group 4 (ring stress): window 4: 4×3×2×3 = 72; window 8: 8×3×2×3 = **144**
- Total **2432**, served-ring (window 128) **2216 ≥ 150** — both match `run.log` exactly.
- Wrapped **and** unwrapped restored prefixes: `replay = 7·window + mod` so `replay%window = mod`; wraps when `mod + pre > window` (mod=127 & pre≥2, mod=7 & pre≥122, mod=1 & pre=128, mod=0 & pre=128 fills the ring), unwrapped for mod=0 and mod=128−pre (boundary-exact). Short appends kept 1..7 ✓ (plus 8, 32). Long appends 256/2048 ✓. Served ring window 128 ✓ (plus 4/8 stress).
- Byte-for-byte: `memcmp` over `out_h` (n·row_h), `out_p` (n·row_p), **and** `pos` (n·8) per case ✓. Served shape row_h=40960, row_p=16 (4×fp32) matches the served config.

### 5. Mutation checks — OK (non-vacuous), with S2/S3 notes
- Mutations are real and targeted at asserted-unique kernel strings. Output changes:
  - `fresh_off_by_one`: fresh rows shift by one; row patterns distinct → fails all 2432 (run.log: 2432) ✓.
  - `pos_plus_one`: positions wrong on every case → 2432 ✓.
  - `bad_slot` (`p % (window−1)`): I predicted passes analytically — window 128: never equal (`m%128` vs `(m+7)%127`); window 8: equal iff `m<7` → 42 (mod 0, pre≤7) + 36 (mod 1, pre≤6) = 78 passes; window 4: never equal → predicted **2354** failing = 2432 − 78. run.log shows **2354** exactly ✓.
- Not vacuous: `run_kernel` executes the compiled kernel into `out_h_k/out_p_k/pos_k`; `check()` compares against `ref_assemble` outputs in separate buffers (`out_h_r/out_p_r/pos_r`); `out_h_k/out_p_k` poisoned 0xAA per case. `and`-chaining only short-circuits *after* a failure — it cannot hide one.

### 6. Run log authenticity — statically authenticated; live re-run **not verified**
- `run.log`: 0 mismatches, all three mutants caught (2432/2432/2354 failing checks), kernel restored, `h0271: ok`. `mut_run.log` on disk is the last mutant's (bad_slot) output, consistent with 2354.
- **Not verified:** I did not re-run `build.sh` (no shell/exec tool available to this subagent; per instructions I report rather than execute). The log's counts match my independent enumeration (2432/2216) and my independent analytic prediction of bad_slot's exact pass count (78) — strong static authentication. Supervisor should run `bash …/h0271/build.sh` once to confirm reproducibility.

### Findings
- **S2 — fill-state invalidation across reallocation and row_h changes** (`fused_run.cpp:75–107`): `Buf::need` frees+reallocs on growth but the `tail_filled`/`fresh_filled` statics persist, so rows below the fill frontier are never patterned in the *current* allocation (e.g., the fresh buffer reallocates when `kept=1, first=5` needs 7 rows; the next `kept=2, first=0` case reads rows 0–1 that were only patterned in the previous allocation). Those rows' contents are allocator-dependent (mmap zeros vs recycled heap), so mutant sensitivity on those cases is luck — it held this run (fresh_off_by_one caught 2432/2432) but is fragile. Relatedly, groups 3/4 read tail/fresh bytes written under a different `row_h` (fill functions skip refilling), so row patterns collide (group-3 tail rows `s` and `s+16` are byte-identical: byte `j = (16s+j)·7+1`, and `112·16 ≡ 0 mod 256`). Clean-run equality is unaffected (kernel and reference read the same bytes). Smallest fix: reset `tail_filled`/`fresh_filled` whenever `need` reallocates, and refill when `row_h`/`window` change (or drop the fill-cache and rewrite all rows per case).
- **S3 — `pos_k` not poisoned** (`fused_run.cpp` `run_case` memsets only `out_h_k`/`out_p_k`): a mutant dropping the `pos` write would compare uninitialized memory for positions. Poison `pos_k` (0xAA) alongside the row buffers.
- **S3 — end-marker uniqueness not asserted** (`splice.py`): `src.index("// A greedy row's token as sampling.argmax takes it on the host")` takes the first occurrence; the start marker has `assert count == 1` but the end marker does not (grep confirms it is unique today). Add `assert src.count(end) == 1`.
- **S3 — duplicate mod values** for pre ∈ {1, 127, 128} (128−pre collides with 127/1/0): re-runs identical cases; harmless. And `pre=0`/`kept=0` layouts are not covered (the brief mandates appends 1..7, so within brief; the kernel handles both trivially).

**Merge verdict: OK with notes.** Skipped/not checked: live `build.sh` re-run (no exec tool — supervisor must run it), and the real GPU launch config (irrelevant to emulation semantics).

*Ponytail: skipped the build.sh re-run and byte-count verification of the .inc (no shell tool); the two remaining risks are the S2 allocator-dependent fill gap and unauthenticated live repro.*