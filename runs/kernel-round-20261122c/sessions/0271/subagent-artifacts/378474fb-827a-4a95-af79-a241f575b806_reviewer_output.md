All verification is complete. Note: no write-capable tool is available to me in this session, so per my operating rules I return the complete review artifact below for the runtime to persist to `/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/runs/kernel-round-20261122c/h0271/review-code.md`.

---

# review-code.md — kernel patch 0271 (TF_DS_RESUME_ASSEMBLE)

**Scope reviewed:** uncommitted worktree diff in `/home/colum/Workspace/tfref-k0271` (base 27baa93) + new file `zig/src/families/dsv41/resume_assemble.zig` + surrounding code (`prompt.zig` replayCut/tailRowsCopy/tailStash/Arena/upload, `ops.zig`, `dsv41_ops.cu`, `model.zig` tails, `idx_fused.zig`, k0265 `draft.zig` banner reference).

## Verdict: OK with notes (no S1/S2; four S3 notes)

---

## 1. OFF-PATH IDENTITY — no issues found.

- The removed base lines (diff `-` hunks) reappear textually verbatim inside the `else` arm at `prompt.zig:2669-2676` — same two `copyRows`, same two `tailRowsCopy` calls, same args, same order, same `std.mem.swap` sequence.
- The gate at `prompt.zig:2663` (`const one = pre > 0 and resume_assemble.ok(...)`) is pure host computation with no observable effect; Zig `and` short-circuits, so `ch.tail.?` is only evaluated when `pre > 0`, and `tailPre` (`prompt.zig:191-195`) already guarantees `tail != null` whenever `pre > 0`.
- With the switch unset/0: `ok()` returns false at its first clause (`resume_assemble.zig:28`, `if (!on ...) return false`) → `one == false` → the fallback arm and `if (!one) try upload(...)` (`prompt.zig:2689`) reproduce base behavior exactly, including the position upload and `ch.pos = ch.pos1` (`prompt.zig:2690`).
- With `pre == 0` (even switch on): `one == false`; the `else if ((first * 8) % 16 != 0)` branch (`prompt.zig:2691-2694`) is untouched.
- The banner log (`prompt.zig:2668`) fires only under `one`. `fromText` of an unset env prints nothing (`resume_assemble.zig:13-14`). Load-time change is one `fromText` + one conditional `fm.function` (`ops.zig:181, 226-227`), matching the conf_pack/topk_merge pattern; when the switch is off, no kernel lookup occurs.

## 2. ON-PATH EQUIVALENCE — no issues found (one test-coverage note, S3-1).

- **Ring/wrap math:** `tailRowsCopy` (`prompt.zig:2609-2621`) resolves prepend row `j` to ring slot `(replay + j) % window` via `slot = p0 % slots`, `n1 = min(rows, slots - slot)`, second segment from `ring[0]`. The kernel (new block in `zig/kernels/cuda/dsv41_ops.cu`, after line 1193) computes `r = p % window` with `p = replay + i` per row — mathematically identical for every wrap split. `tailPre` guarantees `pre ≤ t.end - t.start ≤ window`, so slots are distinct and in range.
- **Fresh rows:** base `copyRows(ch.h + first*row_h → ch.h_alt + pre*row_h, rows = ch.n - first)`; kernel output rows `i ∈ [pre, n)` take fresh row `first + i - pre`, i.e. `first .. ch.n-1` over `n - pre` rows — identical coverage and offsets (same for `row_p`/`ch.pre`/`ch.pre_f`).
- **Positions:** base host fill `host_pos[i] = @intCast(replay + i)` + upload `n*8` bytes to `ch.pos1` (`prompt.zig:2688-2689`); kernel writes `pos[i] = (long long)(replay + i)` for `i < n` — same bytes, and the host fill still runs on the on arm so `host_pos` readers see the same values.
- **Cases:** the new test (`resume_assemble.zig:72-170`) compares the kernel's host reference (`assemble`/`rowSource`) against an independent transcription of today's sequence (`Today.run`, incl. the wrap split) across 2,250 cases: window 4 and 128; `pre ∈ 1..window` incl. `pre == window`; `first ∈ {0,1,4}` (first==0 and first>0); `replay % window ∈ {0,1,window/2,window-pre,window-1}` (wrap boundary and dense wrap); `kept 1..300`. The CUDA kernel transcribes `assemble` line-for-line (verified by inspection: tail/fresh select, row strides `row_h16`/`row_p16`, word loops, `pos[i]`).
- Prepend rows in `[0, pre)` and fresh rows in `[pre, n)` are disjoint outputs with disjoint sources (h/h_alt are distinct Arena allocations, `prompt.zig:456`), so one-launch write ordering cannot differ from the base's copy-then-overwrite.

## 3. ABI — no issues found.

- Kernel (14 params): 7 pointers (`tail_h, tail_p, fresh_h, fresh_p, out_h, out_p, pos`), `unsigned long long row_h16, row_p16, window`, `int pre, int first`, `long long replay`, `int pdl` (last).
- `Ops.resumeAssemble` (`ops.zig:621-639`) adds, in order: 7× `u64` pointers, `u64 row_h/16`, `u64 row_p/16`, `u64 window`, `c_int pre`, `c_int first`, `c_longlong replay`, `c_int pdl`. Types/widths match exactly (u64-as-pointer is the file's established convention — `copyRows`, `ops.zig:591-610`; `launch.zig:11` "each must have exactly the C type of its kernel parameter"). 14 args / 108 bytes, well within `Args` limits (`launch.zig:13-14`).
- Launch geometry is right for the kernel's indexing: `go(o, f, s, n, 256, &a)` gives `grid.x = n` (a block a row, `blockIdx.x = i`) — same "a row a block" convention as `copyRows`' strided path (`ops.zig:610`). Not `blocks(n, 256)`.

## 4. SAFETY — no issues found (two latent notes: S3-2, S3-3).

- `ch.tail.?.buf` unwrap (`prompt.zig:2663,2665`): guarded by `pre > 0 and` short-circuit, and `pre > 0` already implies `ch.tail != null` (`prompt.zig:2653`).
- Casts: `pre ≤ window`, `first < ch.n ≤ cap` → `c_int` safe; `replay → c_longlong`; `@intCast` traps loudly on violation. `row_h/16`, `row_p/16` are exact because `ok()` requires both `% 16 == 0` (`resume_assemble.zig:28-29`).
- Alignment: `ok()` checks all six word-copied buffers `% 16 == 0` (`resume_assemble.zig:30`); `Arena.take` is 256-aligned (`prompt.zig:66-115`) so `ch.h/pre/h_alt/pre_f/pos1` are fine; `tail_p = buf + window*row_h` is 16-aligned whenever `row_h % 16 == 0`; tail slot offsets `slot*row_h`/`slot*row_p` likewise. `tailBytes` (`model.zig:993-995`) confirms the h-then-p ring layout the kernel assumes (`Tail.buf` comment, `prompt.zig:182`), and a non-16-aligned per-slot `tail.buf` (possible when `row_p % 16 != 0`) is correctly caught by `ok()`'s address check.
- Extents: tail halves are exactly `window` rows; fresh rows read `[first, ch.n) ⊂ cap`; out rows `n ≤ cap` (checked, `prompt.zig:2657`). `n == 0` early-return in `resumeAssemble` is unreachable at this call site.
- **`hc` not a multiple of 4:** `row_p = hc*4` fails `row_p % 16` → `ok()` false → the fallback arm runs, whose `copyRows` strided path handles unaligned rows (`ops.zig:603-610`) — byte-identical to base. Graceful, exactly as intended.
- `pos1` is absent from `ok()`'s address list; it only receives 8-byte `long long` stores and is 256-aligned from `Arena.take` (`prompt.zig:453`) — safe today (S3-3).

## 5. ORDERING — no issues found.

All arm launches and `tailStash`'s copies run on `e.s` (`prompt.zig:2665/2669-2674` and `tailStash → tailRowsCopy`, `prompt.zig:2626-2639, 2609-2621` — all `e.ops.copyRows(e.s, ...)`). The assemble kernel's reads of the tail ring precede `tailStash`'s writes of the ring, and `tailStash`'s reads of the assembled `ch.h/ch.pre` (post-swap = old `h_alt/pre_f`) follow the kernel's writes — same-stream program order serializes both directions exactly as the base's multi-launch sequence did. `attnMixes`/`kvSourceUpdate` (`prompt.zig:2649-2651`) also precede on `e.s`. No new cross-stream exposure; PDL is launch-timing only (`dsv41_ops.cu:11-13`).

## 6. CONVENTIONS — no issues found.

- **Switch module:** `resume_assemble.zig:9-22` verbatim-matches `idx_fused.zig:12-25` (and compress2_norm/rope_store/conf_pack/topk_merge): `env_name`, `parse` (unset/""/"0" off, "1" on, else `std.debug.print("{s}: expected 1 or 0\n")` + null), `fromText` → `error{BadResumeAssemble}`. Loaded identically to conf_pack/topk_merge (`ops.zig:181, 226-227`).
- **Kernel style:** `extern "C" __global__`, `pdl_enter(pdl);` first statement, `pdl` last parameter, prose comment naming the call site — matches `dsv41_ops.cu:11-25` and neighbors (`tf_ds_f16_to_f32_kernel` etc.).
- **Banner idiom:** `prompt.zig:48-49, 2668` matches k0265 `draft.zig:32-33, 694` (`head_scored_noted`) exactly: module-level `std.atomic.Value(bool).init(false)`, `swap(true, .monotonic)`, one `std.log.info("dsv41: ...")`, informative first-fire message.
- **CUDA-graph capture:** the on arm is one launch plus a host-side `host_pos` fill — no `synchronize`, no pageable `cuMemcpyHtoDAsync` (upload skipped at `prompt.zig:2689`; base comment at 2685-2687 documents this). `replayCut` runs eagerly (`model.zig:1626`, `layer_main.zig:839`), outside the draft graphs (`draft.zig:224`), so capture safety is moot-but-satisfied; the on arm is strictly friendlier to capture than the base.

---

## Findings

- **S3-1** — Tests never execute the CUDA kernel. `zig/src/families/dsv41/resume_assemble.zig:72-170` validates `assemble` (host transcription) against `Today.run` (host transcription of the fallback); the kernel `tf_ds_resume_assemble_kernel` is a third transcription verified here only by inspection. `zig build test` is host-only ("no GPU work", `build.zig:130`). *Smallest fix:* none in code — the supervisor must run a device check: the layer gate (`layer_main.zig:839`) or a served exact-resume scenario (`TF_DS_EXACT_RESUME=1`, replay < chunk start) with `TF_DS_RESUME_ASSEMBLE=1` compared against a recording/off-path run.
- **S3-2** — `Ops.resumeAssemble` (`zig/src/families/dsv41/ops.zig:621`) has no shape guard: a future caller bypassing `ok()` with `row_h % 16 != 0` silently truncates `row_h/16` and makes misaligned `uint4` accesses. *Smallest fix:* two clauses (`if (row_h % 16 != 0 or row_p % 16 != 0) return error.BadResumeShape;`) or fold `ok()` in. The sole call site is safe today.
- **S3-3** — `ok()` (`zig/src/families/dsv41/resume_assemble.zig:27-31`) omits `ch.pos1` from its alignment list. Safe today (8-byte `long long` stores; `pos1` is 256-aligned `Arena.take`, `prompt.zig:453`) — latent only if `pos1` ever comes from a non-Arena allocation.
- **S3-4** (style) — The kernel lacks the `if (i >= n) return;` bounds guard its neighbor `tf_ds_scatter_rows_kernel` carries (`zig/kernels/cuda/dsv41_ops.cu:117-118`); safe because the launcher sends exactly `grid.x = n` blocks.

## Commands for the supervisor (I cannot run shell)

1. `cd /home/colum/Workspace/tfref-k0271 && zig build test` — host unit tests (includes the new resume_assemble tests via `refAll`, `dsv41.zig:128`).
2. Build the CUDA kernels (existing cuda build step) to confirm `tf_ds_resume_assemble_kernel` compiles and is looked up under the switch.
3. Device validation per S3-1: layer-gate or served resume run with `TF_DS_RESUME_ASSEMBLE=1` vs off path.

**Merge verdict: OK with notes.**

---
*Skipped/not checked: no compile or GPU run was executed (no shell in this review session); equivalence of the compiled kernel to its host reference rests on line-by-line inspection. Risk to know: S3-1 means a transcription slip inside the kernel body itself would pass `zig build test` — the device-side run in the commands above is the real gate.*