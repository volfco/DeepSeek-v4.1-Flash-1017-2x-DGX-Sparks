# 0291 · `TF_DS_QD_PAIR_SHFL` (off by default): the pair query's scale-group maximum by width-16 shuffles

**Commit** `c58b41e` on `kernel-sweep-0290` (`~/Workspace/tfref-ksweep`). Not pushed. Not timed: no GPU here.

**Pattern** A4 (`PATTERNS.md`): a half-warp masked `__reduce_max_sync`. The audit names this exact line as a
source-level variant to try ("0198 · Dispatch the 128-row prompt query separately", last paragraph).

## Files

| File | Change |
|---|---|
| `zig/kernels/cuda/dsv41_torch.cu` | `tf_ds_amax_bits_shfl<BLOCK>` (new helper); `tf_ds_qd_pair_chunk` and the pair kernel's body become templates on `SHFL`; the old symbol is the `<false>` instantiation; new symbol `tf_ds_fused_rope_qd_pair_shfl_kernel` |
| `zig/src/families/dsv41/qd_pair_shfl.zig` (new) | the switch, `kernel(shfl)`, host models of both reductions, 2 tests |
| `zig/src/families/dsv41/exact.zig` | `Exact.load` picks the symbol; the one-time banner at the pair launch in `ropeQd` |
| `zig/src/families/dsv41/dsv41.zig` | import and `refAll` |

## What changes

In the pair geometry (0241) a warp owns one head and each lane holds a rotary pair, so the chunk's two 32-element
FP4 scale groups are the warp's two halves. The group maximum of the sign-cleared bf16 bits was
`__reduce_max_sync(lane < 16 ? 0x0000ffff : 0xffff0000, …)`. Under `SHFL` it is four xor shuffles of width 16 (8, 4,
2, 1), each followed by an integer maximum. The value reduced is the same and an integer maximum is the same in any
order, so the scale byte, the reciprocal and every output byte are unchanged.

- **Off**: the old symbol, the same launch. **On**: the same launch (grid `(rows, heads / 8)`, 256 threads, same
  arguments) with the `_shfl` symbol. Chosen at load, so a captured graph has the same nodes.
- Ranks need not agree (the same bytes either way).
- Banner (once): `dsv41: TF_DS_QD_PAIR_SHFL: the pair query's scale groups take the shuffle maximum (first at …)`.

## Work removed: none at source level, and this is the honest reading

This patch **replaces one warp collective with four**: the emulation counted 38,301,696 masked reductions for the old
form and 153,206,784 shuffles for the new, none of the other. It is here because of one measurement:

- **Measured for the same construct elsewhere:** on the latent kernel (16-lane groups, the same half-warp masks) the
  masked `__reduce_max_sync` was 1.541 → 1.644 µs at R=1/7/32/48, +0.10 µs at every served row; ptxas duplicated the
  masked REDUX into three code paths; reverting to four shuffle stages restored the time
  (`runs/fused-next-20261121/wp3/RESULTS.md`, item 2). `dsv41_torch.cu`'s own note on `tf_ds_fp4_amax_bits` says the
  integer maximum stays "where a scale group is a whole warp".
- **Not measured:** the pair kernel in either form. The latent comparison was a masked integer reduction against a
  **float** shuffle maximum; this is integer against integer. It may be neutral or worse here (A9: a change on a
  latency-bound chain can cost more than it saves).

**Where it runs:** one launch per indexer query at 32 rows and up (`fused_trace.queryArm`): prompt chunks (128–2,048
rows; `prompt.zig:1514`) and decode windows of 32 rows or more (`round.zig:1290`). One-stream decode (about 6 rows)
takes the one-element form and never reaches it. The record's only timing of the pair form is 2.55 µs at R=32
(`exact.zig`'s note).

## Harness: `h0291/` (`build.sh`, `qd_pair_run.cpp`, `sum.py`, `old.inc`, `new.inc`, `run.log`)

```
old: tfref-kstack4 @ 2bb052f   new: tfref-ksweep @ c58b41e
h0291 TF_DS_QD_PAIR_SHFL: 498 cases, 5976 launches, each case under 3 lane orders (asc, desc, shuffle)
compared 459763776 output bytes (pair old vs pair off, pair old vs pair shfl, one-element old vs pair shfl): 0 failures
PASS
```

- Four arms: the old pair kernel, the new tree's pair kernel with the switch off, the `_shfl` kernel, and the old
  **one-element** kernel (which 0241 says writes the same bytes: an independent check of the grouping).
- Rows 1 / 7 / 16 / 32 / 48 / 128; shapes (heads, head dim, rotary dim): the served (32, 128, 64) and nine others that
  move each boundary (64- and 256-element heads, rotary 0 / 2 / 32 / 62 / 128, 24 heads).
- Nine content classes; **all 65,536 bf16 encodings** go through a scale group (signalling and quiet NaNs of either
  sign, both infinities, both zeros, subnormals).

Limits: the bf16 rounding and the fused multiply-add are host forms on every arm, so the comparison isolates the
reduction. It says nothing about the device's `cvt` instructions, which the patch does not touch.

## Gate (`gate-0291/gate.txt`)

```
tree: 6571e56 + working tree          (the tree committed as c58b41e)
zig build test: exit 0
zig build: exit 0
check.sh errors: 9 (base 9)           output identical to base apart from line numbers
check_all.sh: 334 error lines (base 331)
strict pass (stubs-extra/extra.h): identical to base
```

**The plain stub set is not identical, and why.** All nine errors `check.sh` prints for `dsv41_torch.cu` at the base
sit in `tf_ds_qd_pair_chunk`: the stock stubs lack `__nv_bfloat162` and `max`, so clang never really parses the
function this patch edits. Making it a template changes how clang reports the same gaps: one line reworded (the
statement the ambiguous `__reduce_max_sync` call is in) and three added (`max` undeclared, the missing-type error
twice in its template form). With the four missing declarations supplied (`stubs-extra/extra.h`, the strict pass) the
base and the patched tree print the same 225 lines, 2 of them in `dsv41_torch.cu` and neither in this function. That
strict pass is this patch's real syntax check.

## What only hardware can settle

1. **An nvcc build** (the new symbol), `LOCAL=0 STACK=0`, and the SASS: the old kernel's REDUX count (is the masked
   form duplicated here as it was in the latent?) against the new one's SHFL count.
2. **Standalone timing of both forms** at 32 / 48 / 128 / 1,024 rows, the shapes where the pair geometry is chosen.
   This is the whole question; adopt only on a win there.
3. Parity on the GPU against the kit chain (`_rope_heads` + `_fp4qd_p2`), including the stress rows.
4. The exact 39/39 panel.
