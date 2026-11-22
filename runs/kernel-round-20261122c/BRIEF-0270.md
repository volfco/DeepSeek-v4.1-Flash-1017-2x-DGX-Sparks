# BRIEF-0270 — prompt attention pads its picks once per selection

**Switch:** `TF_DS_PROMPT_PICKS_ONCE` (off by default). **Module:** `zig/src/families/dsv41/prompt_picks_once.zig` (pure).
**Commit subject:** `0270 dsv41: TF_DS_PROMPT_PICKS_ONCE (off by default): prompt attention pads its picks once per selection`

## The item (todo.md, "TensorFold upstream performance candidates / DeepSeek V4.1 on CUDA")

**Pad prompt attention picks once per selection.** Reuse the padded list until an indexer changes the picks or a replay cut changes the rows. For a cold 127-token prompt, layers 2–7 share the same 63 picks per row. Padding to 64 once saves five buffer fills and five copy launches per rank. Preserve the compact row stride used by sharded fetch and check the −1 padding at the attention input. See `zig/src/families/dsv41/prompt.zig:1122` and `:1251`. [Layer map](docs/fused-c-d-scoping.md#L14)

## Where

`zig/src/families/dsv41/prompt.zig`: the per-layer pick padding (buffer fill to -1 + copy into the padded list) ahead of
prompt sparse attention, and the indexer / replay-cut points that change the picks. Find them by name. Also
`docs/fused-c-d-scoping.md` (the layer map: which prompt layers share a selection) in
`/home/colum/Workspace/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/docs/`.

## Contract

- Off: launch- and byte-identical.
- On: a layer whose picks are the same selection as the previous padded list reuses that padded list (no fill, no copy).
  Any event that changes the picks (a new indexer output, a compressed-layer boundary, a replay cut, a new chunk)
  re-pads. The padded buffer the attention reads must be byte-identical to the off path at every layer.
- Validity must be decided from host-known structure (which layers share an indexer selection, chunk/cut boundaries),
  never from comparing device buffers. Prefill may or may not run under graph capture — check and say which.
- Preserve the compact row stride used by sharded fetch (`TF_DS_KV_SHARD=1` is on in production) and the -1 padding at
  the attention input.
- Banner: log once when a reuse first happens.

## Harness (`h0270/`)

Host emulation of the padding and of the reuse decision over a layer sequence: cold 127-token prompt, multi-chunk prompts
(`TF_DS_PREFILL_SLICE=5`), a replay cut mid-prompt, pick counts 1..64 and >64. Compare the padded list each layer's
attention reads under on vs an independently written off reference, byte for byte.

Expected saving: five fills + five copy launches per rank per prompt chunk for layers 2–7 (prefill only; small). Build it
only if the reuse rule is simple and provably safe.
