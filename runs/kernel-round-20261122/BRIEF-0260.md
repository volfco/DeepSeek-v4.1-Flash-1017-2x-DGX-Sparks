# BRIEF-0260 — DSpark confidence-input packer

**Switch:** `TF_DS_CONF_PACK` (off by default). **Module:** `zig/src/families/dsv41/conf_pack.zig` (pure).
**Commit subject:** `0260 dsv41: TF_DS_CONF_PACK (off by default): the DSpark confidence input packs in one kernel`

## What it replaces

`Drafter.body`'s confidence-head preparation in `zig/src/families/dsv41/draft.zig` (locate by content — this is
the tail of `body`, right after `dr.mk.stepsTapped` and the `Probe.check(probe, .markov, ...)`):

```zig
        // the confidence head: [xc (fp32), the embedding rows the steps read (fp32)] @ conf^T, a stream's first steps rows
        const rank = c.markov_rank;
        const w = d + rank;
        try e.d.check(e.d.api.cuMemsetD8Async(ch.invalid, 0, 4, e.s.handle), "cuMemsetD8Async");
        for (0..N) |i| {
            try e.ops.gatherRows(e.s, dr.ds.markov_embed, c.vocab, dr.out + i * ts * 8, dr.embs + i * steps * rank * 2, rank * 2, steps, ch.invalid);
            try e.ops.toF32(e.s, dr.xc + i * n * d * 2, dr.xcf + i * steps * d * 4, steps * d);
        }
        try e.ops.toF32(e.s, dr.embs, dr.ef, N * steps * rank);
        try e.ops.copyRows(e.s, dr.xcf, d * 4, dr.conf_in, w * 4, d * 4, N * steps);
        try e.ops.copyRows(e.s, dr.ef, rank * 4, dr.conf_in + d * 4, w * 4, rank * 4, N * steps);
        try e.blas.xv(dr.conf_in, dr.confw, dr.conf, N * steps, w);
```

At N=4 that is 11 device launches (`gatherRows` ×4 + `toF32` ×4 + `toF32` + 2 × `copyRows`). Keep the
`cuMemsetD8Async` and the `blas.xv` exactly as they are; the 11 launches become **one** packer kernel.

## Exact semantics of the chain (the bit-parity contract)

Notation: `N` streams, `steps` scored drafts per stream, `n = dr.bn = blockRows(steps)` (n ≥ steps), `d =
c.hidden`, `rank = c.markov_rank`, `w = d + rank`, `ts = steps + 1`. Output row `j = i*steps + s`.

- **ids:** stream i's gathered ids are `ho[i*ts + s]` for s = 0..steps-1 (contiguous i64s from `dr.out +
  i*ts*8`; `ho[i*ts]` is the stream's pending token, `ho[i*ts+1+j]` its draft j — so this is the token each
  scored block row *reads*).
- **embedding rows** (`markov_embed` is bf16 `[vocab, rank]`, row stride `rank*2` bytes): `dr.embs` row
  `j` = `markov_embed[id*rank ..]` — **but** `tf_gather_rows_kernel` (`zig/kernels/cuda/torch_ops/movement.cu`)
  semantics on an invalid id (`id < 0 || id >= vocab`): `atomicExch(invalid, 1u)` once, and the destination row
  is **left untouched** (it keeps whatever `dr.embs` row j held from an earlier pass).
- **hidden rows:** `dr.xcf` row j = exact bf16→fp32 widen of `dr.xc` row `i*n + s` (row stride d).
- **ef rows:** `dr.ef` row j = exact bf16→fp32 widen of `dr.embs` row j (after the gather).
- **conf_in** (fp32 `[N*steps, w]`, row stride `w*4` bytes): row j = `xcf` row j (d floats) followed by `ef`
  row j (rank floats).

## New kernel `tf_ds_conf_pack_kernel` (in `zig/kernels/cuda/dsv41_ops.cu`)

One launch over the `N*steps` output rows (e.g. one block a row, 256 threads striding `w` columns — tiny rows,
keep it simple), writing `conf_in` directly:

- For output row j with id = `ho[i*ts+s]`:
  - valid id: widen `markov_embed`'s row into conf_in's `[d, w)` half **and** byte-copy the bf16 row into
    `dr.embs` row j (the chain wrote it there; keeping the write preserves the buffer's history exactly, which
    is what makes an invalid id in a *later* pass byte-identical too).
  - invalid id: `atomicExch(invalid, 1u)` (thread 0 of the row; match the movement.cu kernel's flag
    semantics), and widen `dr.embs`'s **existing** (stale) row j into the `[d, w)` half instead. Do not touch
    `dr.embs`.
  - `[0, d)` half = widen `dr.xc` row `i*n + s`.
- Widen = `__bfloat162float` (exact). Do not clear `invalid` in the kernel (the `cuMemsetD8Async` before stays).
- `dr.xcf`/`dr.ef` keep their allocations; the fused path simply never writes them (they have no other reader —
  verify with grep and say so in the report).
- Follow the `ops.zig` conventions for the wrapper (`pdl` handling like neighbouring launches). Graph-capture
  safe: this runs inside `body`, which `BatchDraftGraph` captures; no host interaction.

## Host reference + tests (`conf_pack.zig`, pure Zig)

Reference functions describing the semantics above (row assembly, the gather's valid/invalid rule, the stale-row
rule). Tests must cover:

- A ≥48-case matrix of stream counts (1, 2, 4), block widths n and depths steps (the historical evidence was
  "48 stream, block and depth combinations") against a straightforward independent transcription of the chain.
- ids at boundaries: 0, vocab-1, vocab, -1, minInt(i64), maxInt(i64); assert the invalid flag and the untouched
  destination row.
- Stale-row parity: a pass with a valid id then a pass with an invalid id at the same row (the chain widens the
  previous pass's gathered bytes).
- bf16 specials (NaN, ±inf, ±0, subnormal, max) widened bit-exact to fp32.

## CPU-emulation harness (`runs/kernel-round-20261122/h0260/`)

Extract the kernel source (p0194 `splice.py` style; this kernel needs only a per-thread loop, no barriers) and
compare its conf_in bytes + invalid flag + resulting `dr.embs` bytes against an independent C++ reference
implementing the chain semantics over the same matrix (use small synthetic vocab/rank/d for speed, plus one
real-shape case: d=5120).

## Out of scope

The validity-flag clear, the cuBLAS `xv`, the drafter noise/probe paths, the 2D draft head, dead-buffer
cleanup (`dr.xcf`/`dr.ef` allocation removal is a separate future item).
