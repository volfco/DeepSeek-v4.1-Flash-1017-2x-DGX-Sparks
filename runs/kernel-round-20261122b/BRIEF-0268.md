# BRIEF-0268 — drafter attention uses the decoder's joint query/KV normalization

**Switch:** `TF_DS_DRAFT_QKV_NORM` (off by default). **Module:** `zig/src/families/dsv41/draft_qkv_norm.zig` (pure).
**Commit subject:** `0268 dsv41: TF_DS_DRAFT_QKV_NORM (off by default): the drafter's query and KV normalization run as the decoder's one qKvNorm launch`

## The item (todo.md, "Drafter projections, prompt attention and replay")

After the input projections and any LoRA updates, query normalization and rotation run separately from KV normalization
and RoPE. The decoder's `qKvNorm` already accepts both inputs. The launch fixture records the matching query width
1,280, KV width 512, RoPE width 64 and one input rotation. This could save one launch per draft block, three per pass.
Verify kit dispatch at actual draft row counts, byte-compare both outputs and the rotated query input, and confirm the
drafter's −1 slot sentinel leaves its ring untouched (`draft.zig:729–749`, `tri_norm.zig:43–55`,
`round.zig:1182–1183`, `fixtures/launches.json`).

## Where

`zig/src/families/dsv41/draft.zig` (the block loop, the `tri_norm.kvNormRope` calls around lines 506/532 and the query
norm/rotation next to them), `tri_norm.zig` (`qKvNorm`), the decoder's call at `round.zig` ~1182.

## Contract

- Off: launch- and byte-identical.
- On: the query output, the rotated query input (if a later reader uses it), the KV ring rows and every other output
  bit-identical. `qKvNorm` dispatches a kit kernel: establish from the kit wrapper's code whether its arithmetic is the
  same as the two separate kernels' (same kit kernels fused, or a different kernel). If the joint kernel is a different
  kit kernel with possibly different rounding, you cannot prove bit-identity locally: say so and list exactly what the
  GPU gate must compare. The −1 slot sentinel must leave the ring untouched.
- Report launches per pass before and after at N=1..4, n=5..7.

## Harness (`h0268/`)

If the change is purely a call-site swap to an existing kit kernel, the harness is a host check of the arguments
(widths, strides, slot lists, sentinel handling) the joint call receives vs the two separate calls, over N=1..4,
n=5..7, with and without LoRA. Say clearly in the report that kernel arithmetic parity is GPU-gate only.
