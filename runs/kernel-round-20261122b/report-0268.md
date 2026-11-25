# report-0268 — TF_DS_DRAFT_QKV_NORM (the drafter's query and KV normalization as one qKvNorm launch)

**Status:** implemented; premise holds. Commit **`f4d53bc87f3123577dc84c98ac72320581f833a6`** (`f4d53bc`) on
`kernel-0268` (`/home/colum/Workspace/tfref-k0268`, base `27baa93` = recipe-0264): exactly one commit, subject as the
brief's export format, working tree clean. Switch: **`TF_DS_DRAFT_QKV_NORM`** (off by default).

## What it does

In `Drafter.attention` (`zig/src/families/dsv41/draft.zig`, the block loop's per-stage attention) the query
RMSNorm + wq_b input rotation (`tri_norm.rmsnormRot`, kit `_rmsnorm_rot`) and the KV RMSNorm + RoPE + FP8 round trip
(`tri_norm.kvNormRope`, kit `_kv_norm_rope`) run as **one** `tri_norm.qKvNorm` launch (kit `_q_kv_norm`, the launch
the decoder already makes at `round.zig:1209`) when the switch is on: one launch a DSpark block instead of two,
three a served pass. With the switch off (or `lay.wq_a.n > 2048`, qKvNorm's width bound) the two separate launches
run exactly as today.

## Files touched

- `zig/src/families/dsv41/draft_qkv_norm.zig` (new, pure): `env_name`/`parse`/`fromText` (unset/""/"0" off, "1" on,
  anything else prints `TF_DS_DRAFT_QKV_NORM: expected 1 or 0` and refuses startup via the `fromText` error),
  `max_d` (2048), `progs(nrot)`, `ringWrite(slot)` (the −1 ring rule), `arg_source` (the joint-vs-separate argument
  map: WK/KOUT the only renames, eps shared), 4 tests.
- `zig/src/families/dsv41/draft.zig`: the `attention()` swap (fallback arm byte-identical to today) + import.
- `zig/src/families/dsv41/ops.zig`: `draft_qkv_norm: bool` field parsed at `Ops.load` via
  `draft_qkv_norm.fromText(envSpan(draft_qkv_norm.env_name))`. **Deviation from the switch convention:** no
  `?cuda.Function` field — there is no new kernel (the joint launch is a kit Triton kernel dispatched through
  `aot.Set` like the two it replaces), so there is nothing to look up when the switch is on.
- `zig/src/families/dsv41/dsv41.zig`: export + `refAll(draft_qkv_norm)` so `zig build test` runs the module's tests.

## Launch counts per pass (served checkpoint: 3 DSpark blocks)

The norm launches are row-invariant (one program a row), so the counts do not vary with N or n; the row count only
sets `grid.x = N * n` on the one joint launch. Before/after is the same in every matrix cell (measured in
`h0268/run.log`):

| N (streams) | n (block rows) | rows R = N·n | norm launches/pass before | after | saved |
| --- | --- | --- | --- | --- | --- |
| 1 | 5 / 6 / 7 | 5 / 6 / 7 | 6 | 3 | 3 |
| 2 | 5 / 6 / 7 | 10 / 12 / 14 | 6 | 3 | 3 |
| 3 | 5 / 6 / 7 | 15 / 18 / 21 | 6 | 3 | 3 |
| 4 | 5 / 6 / 7 | 20 / 24 / 28 | 6 | 3 | 3 |

(2 launches × 3 blocks → 1 × 3; a 4-block checkpoint would go 8 → 4. `n = 5` is the checkpoint's block;
`TF_DS_DRAFT_BLOCK=6/7` adds `n = 6/7`. All other launches per pass are unchanged.)

## Premise findings (why this holds)

1. **`_q_kv_norm` is the two separate kernels fused, per program.** The public kit source
   (`bertholomus/TensorFold` @ `deepseek-v41-tp2`, `kernels.py`; excerpt + quotes in `h0268/kit-q-kv-norm.md`):
   `_q_kv_norm`'s docstring says *"the two independent norms after attn_in, each program the code of the kernel it
   replaces"* — `(r, 0)` is `_rmsnorm_rot`'s body, `(r, 1)` is `_kv_norm_rope`'s body verbatim (same even/odd-half
   reduction, same RoPE contraction, same FP8 round trip), `(r, 2 + i)` at the served `qkv_split` recomputes the
   norm ("the same code, the same bits") for rotation `i`. It is nevertheless a **different compiled cubin**; see
   gaps.
2. **The −1 sentinel leaves the ring untouched.** Both `_kv_norm_rope` and `_q_kv_norm`'s kv program guard the ring
   store with `if slot >= 0` (quoted in `h0268/kit-q-kv-norm.md`). The drafter's block rows pass `dr.row(5)`, the
   all−1 sentinel (`draft.zig:588`, `hv[5 * max_rows + r] = -1`), so no ring row is written — same as today's
   `_kv_norm_rope` launch. The harness checks the joint call receives the same `SLOT_OF`/`RING`/`ring_size` as the
   separate call and that the sentinel list writes nothing.
3. **Kit dispatch at draft row counts.** Variant resolution (`zig/src/cuda/aot.zig` `matches`/`fits`) is on the
   constexprs and the arguments' specialization only — **no grid**, so `grid.x = rows` does not participate. The
   drafter's joint launch is exactly the recorded `_q_kv_norm` variant class (`fixtures/launches.json` case
   `round:r16:b1024`, NROT 1): consts `BLOCK 2048, D 1280, DK 512, NROT 1, PDL 1, QUANT 1, RD 64, SPLIT 1`, the
   same dtypes, and scalars `xs/os_/ring_size = 1280/1280/576` (the served pool is 4 slots × 144 ring rows = 576 —
   identical to the decoder's recorded ring size). The harness asserts this class match at 28 rows against the
   embedded fixture. The published kit itself is not on this workstation, so this is established from the recording
   plus the matching code, not a `Set.has` against the kit's `aot.json`.
4. **The item's line hints are mispointed.** BRIEF-0268 cites `draft.zig:506/532` for the kvNormRope calls; those
   are `absorb`/`absorbMany`, which have **no query side** and cannot take `qKvNorm` (it always runs both
   programs). The only query+KV pair is `attention()`'s (lines ~747/754 in this tree, matching the brief's
   `draft.zig:729–749` region), and the saving statement — "one launch per draft block, three per pass" — matches
   `attention()` alone. The swap is there; `absorb`/`absorbMany` are untouched (that batching is todo.md's separate
   KV-ring item, 0263, parked).

## Off path / parity

- **Off: launch- and byte-identical.** The else arms are the pre-change calls byte for byte; the joint branch adds
  no launches and no host work (`plane_rows` just moved up — a pure expression). The switch is read once at
  `Ops.load` on every rank (both ranks identical; no rank-conditional path).
- **On:** every output of the two old launches is produced by the joint launch into the same buffers (`dr.qr`,
  `dr.xhq`, `dr.kvb`), and nothing between the joint's position (where `rmsnormRot` ran) and the old `kvNormRope`
  position reads `dr.kvb`/`dr.ykv` or writes `dr.qa`/`dr.ykv` (`groupedRotated` writes `dr.q`, `lora.apply(.wq_b)`
  writes `dr.q`). Readers after the joint: `wq_b` (dr.xhq), `lora.apply(.wq_b)` (dr.qr), `sparseAttn` (dr.kvb).
  The launch order changes on the on path (the KV norm moves before `wq_b`), which is unobservable: the kv outputs
  have no reader before `sparseAttn` and the ring is untouched at the sentinel.
- CUDA-graph safe: one plain kernel launch in `body`'s stream, no syncs, no transfers (identical in kind to the two
  launches it replaces; the decoder already captures `qKvNorm` in its graphs).

## Gate results (all three pass)

1. `zig build test` (worktree root, zig 0.17.0): **exit 0**, all steps (includes the new module's tests via
   `dsv41.zig`'s `refAll`). Note: some runs end with a stray `failed command:` line from a test binary's own output
   (present at the pristine baseline too, exit 0); a deliberate-failure probe confirmed real failures surface as
   exit 1 with `Build Summary ... (1 failed)`, so the gate is sound.
2. CUDA syntax check: the shared `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails on the pristine base
   itself** (its stubs are older than the tree: `__nv_bfloat16` from int, `__nv_bfloat162`, `float4`, `__clz`) —
   the same finding report-0260 recorded at `8d838cc`; this patch touches no `.cu`. The round-local fixed stubs the
   previous round built (`runs/kernel-round-20261122/stubs/check.sh`, left untouched) print
   **`cuda syntax check: ok`** on this tree.
3. CPU harness `runs/kernel-round-20261122b/h0268/` (`build.sh`, `src/harness.zig`, `src/` copies taken from the
   worktree at run time, `run.log`, `kit-q-kv-norm.md`, plus the two reviews): **20/20 tests, exit 0** (its 6 tests
   plus the copied `tri_norm` recorded-launch conformance and `draft_qkv_norm` tests). It runs the real `tri_norm`
   wrappers over N=1..4 × n=5..7 with and without LoRA and checks: all 19 joint arguments equal the separate
   launches' per `arg_source` (WK/KOUT the only renames; eps on both arms), grids (R, 3, 1), the const merge (kv's
   D as DK, SPLIT on; every recorded const compared), widths/strides (`xs = os_ = 1280`, `ring_size = 576`), the
   sentinel slot list (same buffer, −1 every row, zero ring writes), the fixture variant class at 28 rows, the
   read-after-write order of the swap (with and without LoRA), and the launch counts 6 → 3 per pass everywhere.
   `run.log` is the run of the reviewed sources (re-run after the review fixes, per the review's S1).
   **Kernel arithmetic parity is GPU-gate only** — the harness is a host check of the launches, exactly as
   BRIEF-0268 prescribes for a call-site swap to an existing kit kernel.

## Review findings

Two independent reviewer subagents ran after implementation; full texts in `h0268/review-code.md` and
`h0268/review-harness.md`. Both verdicts: **OK with notes, no blockers**.

- Code review (the diff): off-path identity, on-path argument equivalence (including the decoder's call
  field-for-field), dependency order, graph-capture safety, sentinel, style and switch conventions all verified.
  Findings: **S3** `draft_qkv_norm.max_d = 2048` duplicates `tri_norm.qKvNorm`'s `RowTooWide` literal (drift risk if
  the wrapper's bound moves) — *left as-is* (the comment names RowTooWide; hoisting a shared const would touch the
  already-qualified wrapper); **S3** `rot[0..1]` vs `&rot` form difference between the two arms — *left as-is*
  (behavior identical; minimal diff).
- Harness validation: no self-comparisons, const coverage complete, matrix complete. Findings and dispositions:
  **S1** fresh re-run to authenticate `run.log` — *done* (build.sh re-run after all fixes, exit 0, 20/20);
  **S2** the LoRA dimension of the argument test is a replay, not a variation (by construction: LoRA does not enter
  the norm arguments) — *fixed* with an explicit comment and the model-tracking note (the LoRA-dependent property
  is carried by the order test); **S3** the variant-class test could pass vacuously — *fixed* (`seen > 0`
  assertion); **S3** kv-side eps comparison missing from the arg loop — *fixed* (explicit `eps` check on the kv
  arm); **S3** the wide-row fallback arm (`lay.wq_a.n > 2048`) is unexercised — *noted here*: that arm is exactly
  the unchanged off-path calls, and no served checkpoint has `wq_a.n > 2048`.
- Process note: a reviewer created the first git commit despite its review-only brief (tree content was exactly the
  reviewed change); the commit message was amended in place to remove a duplicated paragraph, keeping **exactly one
  commit** with the brief's subject. The `code-review-graph` tool named in AGENTS.md is not available in this
  environment (MCP search: no match); the two reviewer subagents stand in for it.

## Design decisions

- **The joint call sits where `rmsnormRot` sat** (before `wq_b`), matching the decoder's `qKvNorm → groupedRotated`
  order; the KV norm therefore runs earlier than its old position, safe per the reader audit above, one launch.
- **The `max_d` (2048) guard** at the call site keeps the fallback arm real for the wrapper's `RowTooWide` bound
  instead of erroring mid graph-capture.
- **The pure module holds the argument-equivalence spec (`arg_source`) and the ring rule (`ringWrite`)** so both
  `zig build test` and the harness pin the same contract; the harness then executes it against the real wrappers.
- **No `decisions.md`/`continue.md` writes:** this task's write scope is the worktree commit + `h0268/` + this
  report (brief + sole-writer rules); this section is the decision record.

## Known gaps (GPU gate only)

`_q_kv_norm` is a different cubin from `_rmsnorm_rot` + `_kv_norm_rope`; source-level arithmetic is identical
("each program the code of the kernel it replaces", and the kv reduction tree follows `DK`, unchanged) but
compiler-level bit-identity cannot be proven without a GPU. The hardware gate must byte-compare, switch on vs off,
per stage and at N=1..4, n=5..7:
1. `dr.qr` (OUT, the query norm rows) and `dr.xhq` (H0, the rotated query input — its later reader is `wq_b`, and
   `lora.apply(.wq_b)` reads `dr.qr`);
2. `dr.kvb` (KOUT) and the ring plane (must be **unchanged** at the −1 sentinel);
3. then the standing suite: the 39-case exact panel, long-reply hashes, and a graphs-on pair A/B (three launches
   saved on a 4.5–7 ms pass — price by launch count; expect within-noise timing).

Also unsettled locally: resolution against the **real kit's** `aot.json` (fixture-class evidence only) and any
variant `num_warps`/PDL drift between the recorded and served kit builds.

## Follow-ups (todo.md)

None from this change. The absorb/absorbMany KV-ring batching (0263, parked) remains the separate todo.md item
"Drafter projections, prompt attention and replay".
