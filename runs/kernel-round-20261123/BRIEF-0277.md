# BRIEF-0277 — TF_DS_DRAFT_RING_BATCH: the drafter's three KV ring stores in one launch

**Switch:** `TF_DS_DRAFT_RING_BATCH` (off by default; idx_fused.zig convention). **Module:** `zig/src/families/dsv41/draft_ring_batch.zig`.
**Commit subject:** `0277 dsv41: TF_DS_DRAFT_RING_BATCH (off by default): the drafter's per-stage KV norm, RoPE and ring stores in one launch`
**todo.md item:** "Batch the three drafter KV ring stores after projection."

- Today `Drafter.absorb` and `Drafter.absorbMany` (`draft.zig`) group the stage projections, then call
  `tri_norm.kvNormRope` once per stage `j` (three DSpark stages), each a launch that normalizes (`kv_norm`), applies RoPE
  and writes the stage's ring, and also writes a BF16 copy to `dr.ao`. Verify from the code that `dr.ao` has no reader
  on either path (todo says so; prove it or refute it).
- With the switch on, one kernel over (stage, row) does the same normalization, RoPE and ring write for all stages, and
  skips `dr.ao` if it truly has no reader. Ring bytes must be **bit-identical** to the three-launch path (same float
  order: reuse the exact arithmetic sequence of the existing kernel; read which kernel `kvNormRope` launches and whether
  it is ours or a kit kernel — if it is a kit cubin, stop and write that as the report: we never replace kit kernels).
- Served shape: absorb runs each decode round for the accepted rows (1–7 rows), 512 channels a stage. Report launch
  counts per round before/after.
- Harness: CPU emulation of the new kernel vs an independent reference of norm + RoPE + ring write, covering ring
  wraparound, `absorbMany` items into different slots, 1/5/7 rows, and both entry points.
- Under graph capture: absorb runs inside captured graphs; no host syncs.
