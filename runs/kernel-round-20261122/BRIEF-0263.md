# BRIEF-0263 — Batched drafter KV ring store

**Switch:** `TF_DS_KV_RING_BATCH` (off by default). **Module:** `zig/src/families/dsv41/kv_ring_batch.zig`
(pure: the switch + small arithmetic reference functions). 
**Commit subject:** `0263 dsv41: TF_DS_KV_RING_BATCH (off by default): the drafter's three KV ring stores are one grid`

## What it replaces

`Drafter.absorb` and `Drafter.absorbMany` (`zig/src/families/dsv41/draft.zig`, locate by content — the
`for (dr.ds.blocks, 0..) |lay, j|` loops calling `tri_norm.kvNormRope`) run **one `_kv_norm_rope` Triton launch
per stage** after `mainKeys`' grouped projections (3 stages on the served checkpoint). Each launch normalizes
its stage's `dr.ak[j]` rows with the stage's `lay.kv_norm` weights, applies RoPE over the last `c.rope_dim`
dims at positions `dr.apos`, applies the FP8 (E4M3) quant-dequant round trip, writes `dr.ao` and the stage's
ring rows at `slots[r]`. `dr.ao` has **no reader in either path** (verify by grep; the allocation stays).

New: **one grid over stages and rows** that reads each stage's weights and writes its ring directly. Three
launches become one and the `dr.ao` scratch writes go away (21 KiB per rank at seven rows).

## Exact semantics to reproduce (bit-level on the ring bytes)

Read `tri_norm.kvNormRope` (`tri_norm.zig`, its doc comment: "the window KV out [R, D] = RMSNorm(y), RoPE on
its last rd dims, FP8 quant-dequant, also into ring row slots[r]") and its call sites first. Per row r of
stage s:

- `y` = `dr.ak[s]` row r (bf16, `d = c.head_dim`), `w` = stage `lay.kv_norm` (bf16), `eps = c.eps`.
- RMSNorm with the reduction the Triton kernel uses. **Reuse `tf_ds_triton_rmsnorm_row`** in
  `kernels/cuda/dsv41_torch.cu` — it is the codebase's existing replication of the Triton `_rmsnorm`
  arithmetic (built for exactly this class of fusion). Do not use `sqrtf`+`__fdiv_rn` blindly; the 0197
  lesson in continue.md ("Bit-exactness rules that held") applies.
- RoPE over the last `rd = c.rope_dim` dims (pairwise, `cos`/`sin` at `pos[r]`): mirror the Triton contraction
  `re = fma(xe, cs, -(xo*sn))` exactly as `tf_ds_key_store_row` / `tf_ds_fused_rope_*` already do (continue.md:
  "Triton contracts `re = fma(xe, cs, -(xo*sn))`, and the odd element differs between torch and the kit").
  Whether `tf_ds_key_store_row`'s rotation phase is directly reusable is your call — report it.
- FP8 quant-dequant round trip (E4M3), then the bf16 store — follow the existing
  `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)` + `tf_ds_e4m3_to_float` pattern in
  `dsv41_torch.cu` (mind the documented NaN→MAXNORM behavior of the CUDA intrinsic; say in the report which
  rounding points you could pin and which only the hardware gate can settle).
- Ring write: `ring[slot_rows[r] * d + c]` where `slot_rows[r]` = the `slots` array's r-th i64, `ring` and
  `ring_rows` per call (`absorb`: `pool.view(j, slot)` base, `ring_rows = pool.ring`, slots = `(first + r) %
  pool.ring`; `absorbMany`: `pool.rings[j]`, `ring_rows = pool.slots * pool.ring`, slots =
  `slot*pool.ring + p % pool.ring`). Wraparound is just the slot array's modular arithmetic — the kernel must
  not assume contiguity.

## Implementation shape

- One kernel in `kernels/cuda/dsv41_torch.cu` (to reuse its `__device__` helpers directly), e.g. grid
  `(rows, stages)` or `rows*stages` blocks. Per-stage `{y, w, ring}` pointers can ride a small struct
  **by value** in the kernel parameters (the codebase already does this: `TfDsPrefetch` in `dsv41_ops.cu`) —
  max stages is a compile-time bound (`max_stages` in draft.zig).
- Wrapper in `ops.zig` following the house style; call site: replace the per-stage `kvNormRope` loops in
  `absorb` and `absorbMany` with one batched call when the switch is on, else keep the loop **unchanged**.
  `tri_norm.kvNormRope` itself stays (it is the fallback arm).
- Graph-capture safe (the absorb path runs under captured graphs; no host interaction in the kernel).

## Host reference + tests (`kv_ring_batch.zig`)

Pure reference functions for the arithmetic the harness shares: the E4M3 quant-dequant round trip (pin all
256 byte values and the NaN/±0/inf inputs), the RoPE pair contraction (`re = fma(xe, cs, -(xo*sn))` pinned
against a few known values), and the slot mapping (single slot, multiple slots, wraparound). The tests pin
the reference; the harness proves the kernel matches it.

## CPU-emulation harness (`runs/kernel-round-20261122/h0263/`)

Extract and execute the batched kernel's source on CPU against an independent C++ reference implementing the
documented `_kv_norm_rope` semantics. Cover: single-stream absorb (7 rows), batched absorb of several items
into **different slots**, ring **wraparound** (slots crossing `ring_rows`), 3 stages with distinct weights and
rings, `rd < d` rotation placement, odd row counts. Also run the reference as the **three-launch chain**
(three separate calls with the same reference arithmetic) and byte-compare all three rings — this is what
catches indexing/wraparound/stage-mixing bugs, which is what this harness can settle. Say explicitly in the
report that Triton bit-parity itself is hardware-gate only (the overnight plan byte-compares the rings after
single-stream absorb, batched absorb into different slots, and wraparound, then checks the first draft that
reads the updated rings).

## Out of scope

`dr.ao` allocation removal (the write is dropped, the buffer stays), the drafter's `ropeHeads`/`rotMany`
chains, `mainKeys`, the decoder's own norm/RoPE paths, and anything in the absorb **scheduling** (slots,
positions, `pool.absorbed`) — only the three ring stores change.
