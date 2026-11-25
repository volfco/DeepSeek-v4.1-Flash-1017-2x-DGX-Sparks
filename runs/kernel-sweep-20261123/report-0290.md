# 0290 · `TF_DS_DRAW_CUT` (off by default): the sampled selectors stop at a complete radix bin

**Commit** `6571e56` on `kernel-sweep-0290` (`~/Workspace/tfref-ksweep`, base `2bb052f`). Not pushed. Not timed: no GPU
on this workstation.

**Pattern** P1 (`PATTERNS.md`), applied to the two selectors `docs/fused-kernel-rebenchmark-audit.md` names first
("0261 · Sampled verification top-k", items 2–4, and the amendment that the fallback draw runs the same eight passes).

## Files

| File | Change |
|---|---|
| `zig/kernels/cuda/dsv41_ops.cu` | `tf_ds_draw_rows_kernel` and `tf_ds_topk_select_kernel` become templates on `Cut`; the old symbols are the `<false>` instantiations; new symbols `tf_ds_draw_rows_cut_kernel`, `tf_ds_topk_select_cut_kernel` |
| `zig/src/families/dsv41/draw_cut.zig` (new) | the switch (`parse` / `fromText`, a bad value refuses startup), `selectKernel` / `drawKernel`, the host reference `select` (the kernels' radix, digit for digit, with and without the stop), 2 tests (134 rows) |
| `zig/src/families/dsv41/ops.zig` | `Ops.load` looks up the `_cut` symbol of whichever selector its own switch enables; the one-time banner in `drawRows` / `topkSelect` |
| `zig/src/families/dsv41/dsv41.zig` | import and `refAll` (the tests run under `zig build test`) |

## What changes

Thread 0 already walks the histogram to find the bin the k-th key falls in. Under `Cut` it also notes whether that bin
holds **exactly** the keys still wanted (`hist[b] == want - above`, `want != 0`), in bit 8 of the shared `digit`. Every
thread reads it after the barrier and leaves the digit loop. The partial prefix (lower digits zero) then admits the
keys of the higher bins plus the whole cutoff bin: exactly k keys, the same set the remaining digits would have
reached, because draw keys are unique (the id is in the low 32 bits). The compaction (`key >= prefix`, slots by
`atomicAdd`) and the rank-count sort are untouched and see the same candidates.

Kept as the audit asked: `tf_ds_draw_key`, global ids, the output value bits, the order, rank 0's row[0] NaN flag. **No
digit is skipped on a constant byte**: a rank's global ids straddle 65,536 (rank 1's start at 64,640), so 0254's column
shortcut is not ported.

- **Off** (unset, `""`, `0`): the old symbols, the same launches, grids and arguments; the `<false>` instantiation's
  source differs from `2bb052f` only by branches that are constant-false and a `const unsigned dg = digit;` copy.
- **On** (`1`): the same launches, grids and arguments with the `_cut` symbols. Nothing is added to or removed from a
  captured graph; the symbol is chosen at load.
- **Ranks need not agree**: either setting produces the same bytes, so a pair with one rank on is still exact.
- Banner (once): `dsv41: TF_DS_DRAW_CUT: the sampled selection stops at a complete radix bin (first at …)`.

## Work removed

Per sampled row, per rank: each skipped digit is one pass over the row (a load, a key and a shared-memory atomic per
column) and three block barriers. Old: 8 passes + the compaction, 9 reads of 64,640 fp32 (the audit's "about 2.22 MiB
a row a rank"). The emulation counted the passes the source executes at 64,640 columns:

| Row class | Passes (of 8) with the stop |
|---|---:|
| Gaussian, "served-like" (Gaussian with outliers), row[0] NaN | 2.36–2.49 |
| distinct values, ramps, subnormals | 1.85–3.70 |
| 1/8-grid values with many ties | 6.43 |
| ties straddling k | 7.00 |
| all equal, all NaN, all ±inf, two-valued | 8.00 (no change) |

These are synthetic rows. **How many passes real serving logits take is not known** (the audit lists it as open).

**Measured for the same pattern elsewhere:** 0254, 237.0 → 100.7 µs at 128 × 32,768, k=512; all −∞ 227.2 → 138.8
(`runs/suite-fused-next-20261121/kernel-review/RESULTS.md`). No timing of 0261's selector exists in any form.

**Where it runs:** once a device-drawn round (production sets `TF_DS_T27_DEVICE_DRAW=1` and `TOPK_MERGE`):
`topkSelect` on both ranks (`round.zig:2190`); `drawRows` on rank 0 only when the merged exchange does not engage
(`model.zig:1919`). Greedy rounds never reach either. The whole head stage is 1.16 ms of a 42.8 ms cycle, so the
ceiling is small (the audit's estimate: near 0.1%).

## Harness: `h0290/` (`build.sh`, `draw_cut_run.cpp`, `old.inc`, `new.inc`, `run.log`)

The old kernels are cut from `tfref-kstack4` at `2bb052f`, the new from the committed tree, and run on the CPU
(`emu/emu.h`: a fiber a thread, 1,024 threads a block). Three arms: old, new with the switch off, new `_cut`.

```
old: tfref-kstack4 @ 2bb052f   new: tfref-ksweep @ 6571e56
h0290 TF_DS_DRAW_CUT: 1310 cases (1173 select, 137 draw), 11790 launches, each case under 3 lane orders (asc, desc, shuffle)
compared 11485656 output bytes old-vs-off and old-vs-cut: 0 failures
PASS
```

- Rows 1 / 7 / 16 / 128; widths 56, 57, 63, 64, 65, 255, 256, 257, 1,023, 1,024, 1,025, 4,097, 64,640, and 129,280
  for the draw; k 1, 2, 7, 20, 55, 56; global id bases 0, 64,640 and 65,519; a padded row stride.
- 15 row classes: NaN (payloads, either sign, all-NaN, row[0]), ±inf, signed zeros, subnormals, ties straddling k,
  ties completing k, all equal.
- Three lane orders, because the slot an `atomicAdd` hands out follows the schedule.
- The select's list is also checked against an independent full sort by the draw key.
- The draw arm runs sampled and greedy rows with random temperature, top_p, min_p, seed and position.

Limits: it checks the C source in IEEE arithmetic with the host's `exp`/`log` on both sides. It is not codegen, not
the device's libm, and not a race check.

## Gate (`gate-0290/gate.txt`)

```
tree: 2bb052f + working tree          (the tree committed as 6571e56)
zig build test: exit 0
zig build: exit 0
check.sh errors: 9 (base 9)           output identical to base
check_all.sh: 331 error lines (base 331); diff against base: identical
```

## What only hardware can settle

1. **An nvcc build** for `sm_121`: the two new symbols must exist in the fatbin before the switch can be set (with it
   on, `Ops.load` fails on an old fatbin; with it off nothing new is looked up). `LOCAL=0 STACK=0` and the register
   count of both instantiations.
2. **Parity on the GPU**: the old and `_cut` symbols on the same rows, bytes compared (`tools/fused_bench` style), and
   Compute Sanitizer racecheck on the tied case.
3. **Selector-only timing**, then the audit's three arms on sampled requests (full-logit fallback, shipped 0261,
   0261 with the stop) at one and four streams, k=20 and mixed k to 56, scored on the GPU clock
   (`runs/overnight-20261122/table_gpu.py`). `runs/k280/topk-run.sh` is staged for the first two arms.
4. **The pass count on real logits**: a histogram of the digit the stop fires at.
5. The exact 39/39 panel and the seeded sampled panel (hashes, rounds, accepted counts).
