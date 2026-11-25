# 0292 · `TF_DS_BLOCK_MAX8` (off by default): the pool's block maxima take aligned 16-byte loads

**Commit** `c5c257f` on `kernel-sweep-0290` (`~/Workspace/tfref-ksweep`). Not pushed. Not timed: no GPU here.

**Pattern** P5 (`PATTERNS.md`): 0213 gave the fused indexer's phase 1 two aligned 16-byte loads a block
(`tf_ds_block_max_t<8>`) and 0255 did the same for the candidate mask; the grid kernel that computes the same maxima
outside the fused launch kept eight scalar loads.

## Files

| File | Change |
|---|---|
| `zig/kernels/cuda/dsv41_ops.cu` | new `tf_ds_block_max8_kernel` after `tf_ds_block_max_kernel` (which is untouched) |
| `zig/src/families/dsv41/block_max8.zig` (new) | the switch, the launch gate `ok` (0255's `idx_fused.candMask8Ok`), the host reference `blockMaxima8`, 3 tests (9,720 rows against `idx_fused.blockMaxima`) |
| `zig/src/families/dsv41/ops.zig` | `Ops.load` looks the symbol up only when the switch is on; `Ops.blockMax` launches it when `ok` agrees; the one-time banner |
| `zig/src/families/dsv41/dsv41.zig` | import and `refAll` |

## What changes

`tf_ds_block_max8_kernel` is `tf_ds_block_max_t<8>`'s body over `tf_ds_block_max_kernel`'s 2D grid: a whole block's
eight scores in two `float4` loads, then the scalar loop's comparison order (a NaN wins; otherwise a greater score
replaces the maximum, so the first of two equal zeros keeps its sign exactly as before); the row's last partial block
keeps the guarded scalar loop; the newest block is pinned to +inf by the same floor division.

- **Gate** (`block_max8.ok`): the switch, block size 8, the score address a multiple of 16 bytes and the row stride a
  multiple of four fp32. Anything else launches `tf_ds_block_max_kernel` as today.
- **Off**: the old symbol, nothing new looked up. **On**: the same grid (`(ceil(nb / 256), rows)`), block (256) and
  arguments; only the function differs. Chosen per launch by the host, outside any kernel, so a captured graph holds
  the same node with a different function.
- Ranks need not agree.
- Banner (once): `dsv41: TF_DS_BLOCK_MAX8: the pool's block maxima take aligned 16-byte loads (first at … rows, … blocks)`.
- Lines in `ops.zig`'s struct and `load` sit beside `draw_cut`, not beside `l2_prefetch`, to keep clear of the
  0280–0284 branch.

## Work removed

Per block of eight scores: eight 4-byte global loads become two 16-byte loads; the trip count is static. The bytes
read are the same.

**Measured for the same pattern elsewhere:** 0255, 1.257 → 1.051 µs (7 rows, 128 blocks), 1.689 → 1.318 (7 × 2,048
blocks), 15–23% at all 12 aligned shapes, registers 31 → 22
(`runs/suite-fused-next-20261121/kernel-review/RESULTS.md`); 0213 (this very loop body inside the fused kernel),
−6% to −20% at the work-bound shapes (`runs/fused-next-20261121/wp1/RESULTS.md`, item 2). The grid kernel itself has
no timing on record.

**Where it runs:** `Ops.blockMax` is the first launch of the small-row selection (`round.zig:1337`: at most 7 rows and
6,144 blocks or more), of the wide selection (`:1344`, past 16,384 blocks, `IDX_PICK_WIDE`), and of the unfused chains
(`:1350`, `prompt.zig:1585`). So in production it is once a candidate-source layer a round **at deep context only**
(about 100K tokens and up at ratio 2); shallower rounds take the one-launch fused kernel, which already has this
body. All of `a_idx` is 0.8–2.8 ms of a round there.

**Does the gate engage at those shapes?** By the source, yes: the decode sites pass a row stride of
`rd.bucket / ratio` (`round.zig:1186`; buckets are powers of two from 1,024, or multiples of 32,768 under
`TF_DS_IDX_BUCKET_FINER`) and `rd.score` is an arena block (`Arena.take` aligns to 256 bytes). It is the same buffer
and the same rule 0255's mask kernel already engages on. Two cases fall back to the scalar kernel: a bucket capped at
a pool capacity that is not a multiple of 4 × ratio, and the unfused prompt site (`prompt.zig:1585`), whose stride is
a live key count. The banner is the proof on hardware; none has been seen.

## Harness: `h0292/` (`build.sh`, `block_max_run.cpp`, `old.inc`, `new.inc`, `run.log`)

```
old: tfref-kstack4 @ 2bb052f   new: tfref-ksweep @ c5c257f
h0292 TF_DS_BLOCK_MAX8: 890 cases, 2670 launches; 5155726 whole blocks through the two-load form, 28262 rows with a partial last block
compared 42187216 output bytes old-vs-off and old-vs-eight: 0 failures
PASS
```

- Three arms: the old kernel, the new tree's `tf_ds_block_max_kernel` (must be unchanged), the new kernel.
- Rows 1 / 7 / 16 / 128; every width from 1 to 80, then around a warp, a CTA and the pool sizes (248–264, 2,040–2,056,
  4,096, 8,191–8,193, 49,152, 65,536, 131,069, 131,072, 524,288); padded row and output strides; the launcher's grid and a
  clipped one (the stride loop).
- NaN (payloads, either sign, one at each position of a block, all-NaN), ±inf, signed zeros, subnormals, ties.
- `vis` at 0, 1, 7, 8, 9, inside, at and past the row, negative (floor division), and ±2^40.

Limits: the kernels have no barrier or atomic, so the emulation is a plain loop over threads; it checks indexing and
comparison order, not codegen.

## Gate (`gate-0292/gate.txt`)

```
tree: f26942d + working tree          (the tree committed as c5c257f)
zig build test: exit 0
zig build: exit 0
check.sh errors: 9 (base 9)           output identical to base apart from line numbers
check_all.sh: 338 error lines (base 331)
strict pass (stubs-extra/extra.h): identical to base
```

The plain set gains four lines from this patch, all `unknown type name 'float4'` on the new kernel's two load
statements: the stub gap the base already shows on the same two statements in `tf_ds_block_max_t` and in
`tf_ds_cand_fast8_kernel`. (The other three added lines and the one reworded line are 0291's.) With `float4` declared
(the strict pass) `dsv41_ops.cu` has no error at the base or after the patch.

## What only hardware can settle

1. **An nvcc build** (the new symbol), `LOCAL=0 STACK=0` (the `float x[8]` must stay in registers, as it does in
   0255's kernel), and that the loads are two 128-bit `LDG`s.
2. **Parity on the GPU** against `tf_ds_block_max_kernel` on the same rows, and the small-row chain's block list.
3. **Standalone timing** at 1 and 7 rows, 8,192 / 16,385 / 65,536 blocks, alone and in the chain with `idx_pick` /
   `idx_pick_wide` (0215's crossover at 6,144 blocks was set with the scalar kernel and may move).
4. A deep-context decode suite (≥ 128K tokens) on the GPU clock if 3 shows a win; the exact 39/39 panel.
