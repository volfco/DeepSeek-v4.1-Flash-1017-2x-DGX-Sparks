# BRIEF-0271 — exact-resume prepend assembles rows and positions in one kernel

**Switch:** `TF_DS_RESUME_ASSEMBLE` (off by default). **Module:** `zig/src/families/dsv41/resume_assemble.zig` (pure).
**Commit subject:** `0271 dsv41: TF_DS_RESUME_ASSEMBLE (off by default): an exact-resume prepend assembles its rows and positions in one kernel`

## The item (todo.md, "Drafter projections, prompt attention and replay")

**Assemble resumed decoder rows and positions in one kernel.** On an exact-resume prepend, the current path copies the fresh residual and its four mixing weights separately, restores each from one or two ring segments, and uploads a new position vector. One kernel can choose the tail or fresh row by output position, copy both fields and write aligned positions. This takes four copy launches, or six when the restored prefix wraps, plus a host upload down to one launch. Keep the subsequent tail stash ordered after the restore so it cannot overwrite an unread slot. Host row layouts matched in 150 wraparound and short-append cases (`prompt.zig:183–189, 2498–2512, 2544–2576`).

## Where

`zig/src/families/dsv41/prompt.zig`: the exact-resume prepend (fresh residual + four mixing weights copies, ring-segment
restores, the position-vector upload, then the tail stash). Find by name (the todo's line refs were 183–189, 2498–2512,
2544–2576 on an older tree).

## Contract

- Off: launch- and byte-identical.
- On: one kernel selects, per output row, the restored tail row (from one or two ring segments, handling the wrap) or the
  fresh row, copies both the residual and its four mixing weights, and writes the aligned positions on device (no host
  upload). Outputs bit-identical to the off path (pure data movement + integer positions).
- The subsequent tail stash must stay ordered after the restore on the same stream so it cannot overwrite an unread slot;
  say how ordering is guaranteed.
- If the position upload is also consumed elsewhere from the host copy, keep that consumer correct.
- Banner: log once when the assembled arm first runs.

## Harness (`h0271/`)

Per-thread CPU emulation of the kernel against an independently written reference of today's copy sequence, over ≥150
layouts: wrapped and unwrapped restored prefixes, short appends (1..7 fresh rows), long appends, ring sizes from the
served config. Compare residual rows, mixing weights and positions byte for byte.

Expected saving: four copy launches (six when the prefix wraps) + a host upload per exact resume. This path matters for
multi-turn chat resume; measure-only on hardware. Build it only if it is clean.
