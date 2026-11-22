# report-0260 — TF_DS_CONF_PACK

**Commit:** `e06735c` on `kernel-0260-0263` (parent `8d838cc`). **Switch:** `TF_DS_CONF_PACK` (off by default; `1` on,
`0`/`""`/unset off, anything else refuses startup with `TF_DS_CONF_PACK: expected 1 or 0`).

## Files touched

- `zig/src/families/dsv41/conf_pack.zig` (new, pure): switch parse, host reference `pack`, tests.
- `zig/kernels/cuda/dsv41_ops.cu`: `tf_ds_conf_pack_kernel` appended (no `dsv41_torch.cu` helpers needed).
- `zig/src/families/dsv41/ops.zig`: `conf_pack_fn: ?cuda.Function` (looked up only when on) and `Ops.confPack`.
- `zig/src/families/dsv41/draft.zig`: `Drafter.body`'s confidence head: one `confPack` when on, the old chain verbatim
  in the `else` arm.
- `zig/src/families/dsv41/dsv41.zig`: module export + `refAll`.

Deviation: no separate `bool` field next to `conf_pack_fn`. The call site only needs to know whether the kernel is
loaded, so the optional serves as the switch (one source of truth).

## Launch counts (`Drafter.body` confidence head, excluding the unchanged memset and `blas.xv`)

| Streams N | Off (today) | On |
| ---: | ---: | ---: |
| 1 | 5 (gather, toF32, toF32, 2x copyRows) | 1 |
| 4 (served batch) | 11 | 1 |

Grid: `N*steps` blocks x 256 threads, one output row a block.

## Semantics and parity

- Bit-identical **by construction** (the only arithmetic is the exact bf16->fp32 widen, plus byte copies of bf16
  values). The chain's `tf_bf16_to_f32_kernel` (`torch_ops/pointwise.cu:18`) also widens with `__bfloat162float`, the
  same intrinsic the new kernel uses. Identical: `conf_in` bytes, the `invalid` flag, and `dr.embs` bytes. A valid id copies the embedding row in. An invalid
  id leaves the row untouched and widens its stale bytes, exactly as the chain's gather-then-widen did. The off path is
  the unchanged source in the `else` arm.
- `dr.xcf` / `dr.ef`: grep over `zig/src` finds only the allocation (`draft.zig` init) and the chain itself (now the
  `else` arm). They have no other reader, so the fused path leaves them unwritten. The allocations stay.
- `dr.embs`: this chain is its only reader and writer. The fused path still writes it, which keeps the stale-row
  history.
- Graph capture: the kernel and wrapper have no host interaction. The wrapper is a plain launch on `e.s` with `o.pdl`.
- Both ranks: no rank-conditional code.

## Gate

1. `zig build test` (worktree root): **exit 0, 27/27 steps**. Baseline `8d838cc` was also clean in this agent's
   environment (an uncached run into a scratch cache dir, then a rerun with nothing to redo). The two failures the
   round driver named (`prompt_cache.test.0178 (TF_DS_PROMPT_ENDSTATE)` and
   `round.test.TF_DS_EXPERT_HOT_CACHE: hotPick`) did **not** fail here. This environment sets no `TF_*` variables, so
   those failures probably depend on the environment. No new failures. Running `zig test conf_pack.zig` directly
   passes 5/5.
2. CUDA syntax check: the shared `/home/colum/Workspace/cuda-syntax-stubs/check.sh` **fails on the 8d838cc base
   itself** with 24 errors (`__nv_bfloat16` constructed from int, `__nv_bfloat162`, `float4`, `__clz`, `max`) because
   its stubs are older than the tree. The shared stubs were left alone. A round-local copy with those additions, at
   `runs/kernel-round-20261122/stubs/check.sh`, prints `cuda syntax check: ok` on both the base and this commit.
3. CPU harness `runs/kernel-round-20261122/h0260/` (`build.sh`, `extract.py`, `conf_pack_run.cpp`, `run.log`): the
   kernel's exact source (extracted by marker plus balanced braces) runs under a per-thread loop and is compared with an
   independent C++ transcription of the chain, launch by launch. It covers 152 cases x 2 passes. The second pass reuses
   the same buffers with one newly invalid id (stale-row parity). The matrix:
   - N in {1, 2, 4} x steps in {1, 3, 5, 6} x n - steps in {0..3}.
   - Three invalid modes: none; one boundary id; every stream's pending token invalid (incl. `vocab`, `-1`,
     `INT64_MIN`, `INT64_MAX`).
   - Widths crossing the 256-thread stride.
   - The served shape: d=5120, rank=256, vocab=129,280, N=4 and N=1.

   Random u16 bit patterns cover NaN, inf and subnormal bf16. **All match.** Mutation check: of three injected
   index/rule bugs, the two non-equivalent ones fail 158–186 passes. The third (an unconditional embs write) is
   equivalent, because the invalid path reads and writes the same row.

Host tests (`conf_pack.zig`):
- A 48-case matrix against a chain transcription.
- Boundary ids (0, vocab-1, vocab, -1, minInt, maxInt) with asserts on the flag and the untouched row.
- The two-pass stale-row case.
- bf16 specials through both halves.

## Known gaps (hardware gate)

- Only launch overhead changes. The GPU gate should:
  - byte-compare `conf_in` (or the `conf` GEMV output) and `dr.embs`, on vs off, over a drafting run, including an
    invalid-id pass if one can be provoked;
  - confirm 11 -> 1 launches in the captured pass graph.
- Nothing here depends on GPU rounding.
