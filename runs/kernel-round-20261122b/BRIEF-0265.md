# BRIEF-0265 — drafter head projects only the scored rows

**Switch:** `TF_DS_DRAFT_HEAD_SCORED` (off by default). **Module:** `zig/src/families/dsv41/draft_head_scored.zig` (pure).
**Commit subject:** `0265 dsv41: TF_DS_DRAFT_HEAD_SCORED (off by default): the drafter's vocab head projects only the rows a pass scores`

## The item (todo.md, "Drafter projections, prompt attention and replay")

The three attention blocks need every row. After the final block, however, the TP2 head still projects
`N * blockRows(steps)` rows while Markov and confidence read only the first `steps` per stream. Pack those rows while
collapsing the hidden state, retaining the pre-normalization values for confidence. With four streams, five block rows
and three requested steps, the head would project 12 rows instead of 20, avoiding 517,120 FP32 outputs (1.97 MiB) per
rank. A host row-selection check covered 114 layouts, including 64 shallow passes. Account for the distinct strides used
by the noise and speculative-score hooks (`draft.zig:151–175, 558–562, 669–703`, `tri_markov.zig:124–135, 160–165`,
`lanes.zig:1779–1788`). Narrow the final collapse, normalization and head projection together.

## Where

`Drafter.body` in `zig/src/families/dsv41/draft.zig` (the head on every row: `prompt.grouped(e, ch, R, &.{e.w.head}, …
&.{dr.local} …)` near line 679; `dr.xc` / `dr.xn` / `dr.local` layouts at lines ~291–316). Every reader of `dr.local`,
`dr.xc` and `dr.xn` downstream (Markov `localBest*`, confidence packer 0260 `conf_pack.zig`, the noise hook, the
speculative-score hook, `Probe` points `.local`) must see identical values at the rows it reads. Find each reader with
grep and list them in the report with the row index each uses.

## Contract

- Off: launch- and byte-identical.
- On: every value a downstream reader consumes is bit-identical to the off path. The head is a GEMM over fewer rows:
  the kit/cuBLAS dispatch may pick a different kernel at a different M. **Find out whether the head's GEMM result per row
  is row-count invariant** for the dispatch `prompt.grouped` uses (read how it chooses its kernel by M). If it is not
  invariant across the M values involved, either keep M on the same dispatch bucket (say how) or stop and report.
- Graph capture: the pass is captured per (N, steps) shape (`BatchDraftGraph`); the packed row count is a function of
  that same shape, so no new host interaction.
- `lanes.zig`'s round may ask for fewer steps than block rows; record in the report how the served policy picks steps
  vs block rows under `TF_DS_DRAFT_BLOCK=6 TF_DS_DRAFT_BLOCK_KEPT=2` (from reading `passDrafts`, `lanes.zig:2203`), so the
  benchmark knows which passes engage.

## Harness (`h0265/`)

Host emulation of the row selection/packing (gather of `steps` rows per stream out of `n` with the strides used),
against an independent reference, over ≥114 layouts (N 1..4, n 5..7, steps 1..n) including shallow passes. If you add a
packing kernel, emulate it per-thread. If the packing is done with an existing row-copy op, emulate that op's semantics.

Prior measurement (operator side): the target head costs ~9 µs a row, so the expected saving is ~0.07 ms a pass
(0.1–0.2% of a round). That is small; build it only if it is clean and the bit-identity argument holds.
