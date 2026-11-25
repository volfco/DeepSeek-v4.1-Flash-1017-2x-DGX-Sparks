# report-0271 — TF_DS_RESUME_ASSEMBLE (an exact-resume prepend assembles its rows and positions in one kernel)

**Status:** implemented; premise holds. Commit **`8078e03fa45e4f9b1bf35187b68b47167fb1e04a`** (`8078e03`) on
`kernel-0271` (`/home/colum/Workspace/tfref-k0271`, parent/base `27baa93` = recipe-0264): exactly one commit, subject
as the brief's export format, working tree clean. Switch: **`TF_DS_RESUME_ASSEMBLE`** (off by default; unset/`""`/`"0"`
off, `"1"` on, anything else prints `TF_DS_RESUME_ASSEMBLE: expected 1 or 0` and refuses startup via the `fromText`
error).

## What it does

On an exact-resume prepend (`prompt.zig` `replayCut`, the `pre > 0` case: a resumed pass whose replay reaches before
its start, taking rows `[replay, start)` from its tail ring) today's path copies the fresh residual rows and their
four mixing weights separately (two `copyRows`), restores each field's prepended rows from one or two ring segments
(`tailRowsCopy`, two when the prefix wraps), and uploads a fresh position vector (`host_pos` fill +
`cuMemcpyHtoDAsync` to `ch.pos1`). With the switch on and `resume_assemble.ok`'s shape, **one kernel**
(`tf_ds_resume_assemble_kernel`, `zig/kernels/cuda/dsv41_ops.cu`) does all of it: output row `i` copies its
`row_h`-byte residual and `row_p`-byte mixing-weight row from the tail ring's slot `(replay + i) % window`
(`i < pre`, the wrap included in the slot) or from fresh row `first + i - pre`, and writes `pos[i] = replay + i` on
device. The old sequence stays verbatim as the fallback arm.

## Files touched

- `zig/src/families/dsv41/resume_assemble.zig` (new, pure): `env_name`/`parse`/`fromText`; `ok(on, row_h, row_p,
  window, addrs)` (the one-launch shape check: whole 16-byte words, six word-copied buffers aligned); `rowSource`
  (the row rule) and `assemble` (the host reference over byte slices); 3 tests — the switch's forms, the shape check,
  and a **2,250-layout sweep** of `assemble` vs an independent transcription of today's copy sequence (fresh copy +
  `tailRowsCopy`'s segment split + host position fill), windows 4 and 128, `pre` 1..window, `first` 0/1/4, replay
  mods {0, 1, w/2, w−pre, w−1}, kept 1..300.
- `zig/kernels/cuda/dsv41_ops.cu`: `tf_ds_resume_assemble_kernel` (after `tf_ds_copy_rows16_kernel`; `extern "C"`,
  `pdl_enter(pdl)` first statement, `pdl` last argument, a block an output row, a 16-byte word a thread, positions by
  thread 0).
- `zig/src/families/dsv41/ops.zig`: `resume_assemble: bool` + `resume_assemble_fn: ?cuda.Function` parsed at
  `Ops.load` (`fromText(envSpan(...))`), the kernel looked up only when the switch is on; `Ops.resumeAssemble`
  launcher (`grid.x = n`, block 256).
- `zig/src/families/dsv41/prompt.zig`: `replayCut`'s one-launch arm + the unchanged else arm; the engagement banner;
  import + `resume_assemble_noted`.
- `zig/src/families/dsv41/dsv41.zig`: export + `refAll(resume_assemble)`.

## Launch counts at the served shape (an exact-resume prepend, `pre > 0`)

Served config: `hc 4`, `hidden 5120` → `row_h` 40,960 B, `row_p` 16 B (the four fp32 mixing weights), tail-ring
window 128 (`sliding_window`). Per prepend (one per resumed pass's first chunk; up to 128 restored rows, the kept
fresh rows up to the 2,048-row prompt chunk):

| | before | after | saved |
| --- | --- | --- | --- |
| prefix unwrapped | 4 copy launches + 1 pageable HtoD upload | 1 kernel | 4 launches + the upload |
| prefix wrapped | 6 copy launches + 1 pageable HtoD upload | 1 kernel | 6 launches + the upload |

(The two fresh `copyRows` + two-or-four `tailRowsCopy` segments + the `n*8`-byte position upload become one launch.)
The `pre == 0` replay cut (positions as a `pos0` view or one DtoD copy) is a different case and is unchanged.

## Premise findings

1. **The saving exists at the served shape** (the count above matches the todo item exactly: "four copy launches, or
   six when the restored prefix wraps, plus a host upload down to one launch"). The prepend is real work: this path
   matters for multi-turn chat resume; pricing is measure-only on hardware.
2. **The item's line hints drifted** (written against TensorFold `ffdce65`): found by name on this base —
   `tailPre`/`tailRows` (`prompt.zig:184–197`), `tailRowsCopy` (~2602), `tailStash` (~2618), `replayCut`'s copy
   block (~2658–2695).
3. **Bit-identity is by construction**: pure data movement + integer positions. The harness compares the kernel's
   exact source byte for byte against today's sequence over 2,432 layouts (below); the module tests do the same over
   2,250 host layouts.

## Contract items

- **Off: launch- and byte-identical.** The else arm is today's four calls textually verbatim (code reviewer confirmed
  against the base hunks); `if (!one) try upload(...)` keeps the upload; the banner fires only on the assembled arm;
  the only load-time delta is one `fromText` + a conditional kernel lookup (conf_pack/topk_merge's pattern).
- **On outputs:** same bytes as the off sequence — identical source rows/ring slots, and `pos[i] = replay + i` is the
  upload's content. The arm swaps `h/h_alt` and `pre/pre_f` exactly as the off path, so downstream buffer semantics
  (including `tailStash` reading the assembled rows) are unchanged. Both ranks identical (no rank-conditional code).
- **Tail-stash ordering:** `tailStash` (this chunk's rows into the ring) stays after the restore in `replayCut`, both
  on stream `e.s` — same-stream program order serializes the kernel's ring reads before the stash's ring writes
  (exactly the guarantee the copyRows sequence relied on). No events or cross-stream work added.
- **`host_pos` consumers:** `host_pos` is per-chunk scratch (written by `begin`/`replayCut`, read only for its
  uploads at the `model.zig`/`layer_main.zig` call sites — grep-verified). The host fill loop is kept on both arms,
  so the host copy is unchanged; only the device upload is dropped on the on arm.
- **Banner:** `dsv41: TF_DS_RESUME_ASSEMBLE: the prepend's rows and positions assemble in one kernel (first at {d}
  rows, {d} from the tail)` behind `resume_assemble_noted.swap(true, .monotonic)` — 0265's `head_scored_noted`
  idiom, at the arm's first run.
- **Graph capture:** the on arm is one kernel launch + a host-side fill: no syncs, and it *removes* the pageable
  transfer, so it is strictly capture-friendlier than the off path (the code reviewer notes `replayCut` itself runs
  eagerly today — `model.zig:1626`, `layer_main.zig:839` — so capture safety here is moot-but-satisfied).
- **Fallback:** when the switch is off or `ok()` fails (window 0, `row_h`/`row_p` not whole words, any word-copied
  buffer unaligned — e.g. `hc` not a multiple of 4 makes `row_p % 16 != 0`), today's path runs, whose `copyRows`
  strided arm already handles unaligned rows.

## Gate results (all pass)

1. **`zig build test`** (worktree root, zig 0.17.0): **exit 0**, at baseline (`27baa93`, twice) and on the committed
   tree (twice). The stray `failed command:` line some runs end with is the known pristine-baseline artifact (test
   binaries' own injected-failure output; exit 0 either way — same note as report-0268). The new module's 3 tests
   pass (`zig test resume_assemble.zig`: "All 3 tests passed").
2. **CUDA syntax check:** the shared `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails on the pristine base
   itself** (stubs older than the tree: `__nv_bfloat16` from int, `__nv_bfloat162`, `float4`, `__clz` — the same
   finding reports 0260/0268 recorded) and its error set is **identical before and after** this patch (no new
   errors; the kernel adds none). The round-local fixed stubs
   (`runs/kernel-round-20261122/stubs/check.sh`, left untouched) print **`cuda syntax check: ok`** on the committed
   tree. *Deviation from the letter of gate 2:* the shared tool cannot pass at this base at all; the fixed-stub run is
   the meaningful check, as in the previous two rounds.
3. **CPU harness** `runs/kernel-round-20261122c/h0271/` (`build.sh`, `splice.py`, `fused_run.cpp`,
   `fused_kernel_current.inc`, `run.log`, plus the two reviews): **2,432 layouts (2,216 at the served ring), 0
   mismatches.** `splice.py` extracts the kernel's exact source from the worktree `.cu` (verbatim, marker asserts,
   no transformation); the harness emulates blocks sequentially and each block's 256 thread bodies in turn (the
   kernel has no barriers/shared state, so any thread order is its semantics) against an **independent** reference of
   today's copy sequence (bulk fresh copy + `tailRowsCopy`'s two-segment wrap split per field + host position fill),
   comparing residual rows, mixing weights and positions **byte for byte**. Matrix: windows 128 (served) + 4/8
   wrap-density stress; `pre` 1..128 incl. boundary (`mod = 128−pre`) and wrapping (`mod = 127`) prefixes; short
   appends 1..7 fresh rows (+8, 32); long appends 256 and 2,048 rows; `first` 0/1/5. Three seeded mutants
   (`fresh_off_by_one`, `pos_plus_one`, `bad_slot`) are all caught (2,432/2,432/2,354 failing checks; the `bad_slot`
   count matches the harness reviewer's independent analytic prediction exactly). `run.log` is a live re-run from the
   final harness sources.

## Reviews (two independent reviewer subagents; `code-review-graph` unavailable — MCP search: no match, same as 0268)

Full texts in `h0271/review-code.md` and `h0271/review-harness.md`. Both verdicts: **OK with notes, no blockers.**
Findings and dispositions:

- **Harness S2** fill-frontier staleness across reallocation/row-size change (mutant sensitivity could be
  allocator-luck): **fixed** — frontiers reset on reallocation and row-size change; re-run reproduces 0 mismatches
  and the exact mutant counts.
- **Harness S3** `pos_k` not poisoned (a dropped position write would compare garbage): **fixed** (poisoned per
  case).
- **Harness S3** splice end-marker uniqueness not asserted: **fixed** (assert added).
- **Harness S3** duplicate `mod` values for `pre ∈ {1, 127, 128}` re-run identical cases: **left** (harmless repeats;
  `pre = 0`/`kept = 0` are not this path — the call site requires `pre > 0` and returns false when `first >= ch.n`).
- **Code S3-1** tests never execute the CUDA kernel: **mitigated** — the harness executes the kernel's exact source
  against the independent reference with mutation checks; remaining codegen-level evidence is GPU-gate only (gaps).
- **Code S3-2** `Ops.resumeAssemble` has no internal shape guard: **left** — the sole call site gates through `ok()`
  and the doc comment states the precondition (`confPack`'s convention).
- **Code S3-3** `pos1` missing from `ok()`'s address list: **left** — `pos1` only takes 8-byte `long long` stores
  and is `Arena.take`'s 256-byte-aligned (`prompt.zig` `Chunk.init`).
- **Code S3-4** kernel lacks an `i >= n` bounds guard: **left** — the launcher sends exactly `grid.x = n` blocks
  (`copy_rows16` relies on its geometry the same way).

## Design decisions

- **A block an output row, a 16-byte word a thread** (`copy_rows16`'s idiom), positions by thread 0: a pure
  thread-loop kernel — the shape the brief prescribes for barrier-free CPU emulation — and one grid dimension, no
  div/mod per word.
- **Word copies behind `ok()`:** the kernel uses `uint4` loads like `copy_rows16`; unsupported shapes (unaligned
  buffers, `row_p % 16 != 0`) fall back to today's `copyRows` sequence rather than growing a byte-copy kernel.
- **The arm keeps the h/h_alt and pre/pre_f swaps** so every downstream reader (including `tailStash`) sees the same
  buffer layout as today.
- **The host `host_pos` fill is kept** on the on arm (only the upload goes): the host copy stays what it has always
  been for any consumer, at ~zero cost.
- **`ok()` takes the switch itself** (one predicate at the call site, `idx_fused.fuseOk`'s shape).
- **Harness ring sizes:** the served 128 is the brief's "ring sizes from the served config" (the tail ring is sized
  `c.window = sliding_window = 128`; there is no second served ring size in this path). Windows 4/8 are extra
  wrap-density stress beyond the brief, on the same slot rule.
- **No `decisions.md`/`continue.md` writes:** the write scope is the worktree commit + `h0271/` + this report
  (brief + sole-writer rules, 0268's precedent); this section is the decision record.

## Known gaps (hardware gate only)

1. **Codegen-level evidence.** The kernel is pure data movement + integer stores, so bit-identity is by
   construction, but the compiled cubin has never run on a GPU here. The hardware gate should byte-compare switch on
   vs off at a real exact resume (`ch.h`, `ch.pre`, `ch.pos1`), confirm the tail ring is intact for the stash, and
   count the banner (`TF_DS_RESUME_ASSEMBLE=1`, exactly one line) to prove the arm ran.
2. **Timing is measure-only.** Price by launch count (4–6 copies + one pageable upload → 1 launch per prepend); the
   wall-clock win depends on how often multi-turn chat resumes prepend and at what `pre`. A null timing result with
   the banner present is still a valid outcome.
3. **PDL interplay** (`pdl_enter` first statement like its neighbors) is untested on hardware; launch-timing only,
   bits unaffected.
4. **Fallback shapes** (`hc` not a multiple of 4, unaligned buffers) are covered by the module's `ok()` tests and are
   exactly today's code; one negative hardware test at most, if the gate wants it.

## Follow-ups (todo.md)

None from this change. The brief's remaining item in this area — keeping the tail stash ordered — is satisfied by
same-stream program order (above), not deferred.
