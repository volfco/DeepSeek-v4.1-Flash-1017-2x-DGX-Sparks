# BRIEF-0269 — the drafter reads contiguous verify-round taps in place

**Switch:** `TF_DS_DRAFT_TAPS_DIRECT` (off by default). **Module:** `zig/src/families/dsv41/draft_taps_direct.zig` (pure).
**Commit subject:** `0269 dsv41: TF_DS_DRAFT_TAPS_DIRECT (off by default): the drafter reads contiguous verify-round taps in place`

## The item (todo.md, "TensorFold upstream performance candidates / DeepSeek V4.1 on CUDA")

**Read contiguous verify-round taps directly in the drafter.** When participating windows occupy consecutive rows of `rd.taps`, pass the first row and total count to `mainKeys`. Keep packing windows that have gaps. Seven rows with three 5,120-channel BF16 taps avoid a 210 KiB copy per rank. Also skip the three tap launches in ordinary verify rounds when the drafter is disabled, retaining taps for explicit probes. Compare drafter ring contents for a single window, adjacent windows and windows with gaps. See `dsv41/round.zig:1976`, `dsv41/model.zig:1760` and `dsv41/draft.zig:512` under `zig/src/families/`.

## Where

The verify round's tap writes (`round.zig`, the three `rd.taps` launches), the model's tap plumbing (`model.zig`), and
`Drafter`'s `mainKeys` input packing in `draft.zig` (the copy that gathers each participating window's tap rows into
the drafter's contiguous input). Find them by name.

## Contract

- Off: launch- and byte-identical.
- On: when the participating windows' tap rows are consecutive in `rd.taps`, `mainKeys` reads them in place (first row +
  total count, with the source row stride) and the gather copy is skipped; windows with gaps keep today's packing. The
  values `mainKeys` consumes must be bit-identical to the off path (this is a pure addressing change: no arithmetic
  changes, so bit-identity is required, not approximate).
- The second half of the item (skip the three tap launches when the drafter is disabled) is **out of scope**: production
  always drafts. Mention it in the report only.
- Graph capture: the contiguity decision must be a function of what the captured graph already keys on, or be made on
  device. If it depends on host state that varies across replays of one captured graph, stop and report (that would make
  the captured graph wrong on replay). Say exactly what decides contiguity and when.
- Record in the report how often windows are contiguous at the served config (1–4 streams, `TF_DS_DRAFT_BLOCK=6`,
  `KEPT=2`, verify windows up to 7 rows): by reading how rows are laid out in `rd.taps` per round.
- Banner: log once when the in-place arm first engages.

## Harness (`h0269/`)

Host emulation of the row addressing: for single window, adjacent windows, windows with gaps, 1–4 streams, window
lengths 1–7, compare the rows the drafter reads under the in-place arm against an independently written packed-copy
reference, byte for byte. If a kernel's argument changes (stride / base pointer), emulate the kernel's indexing per thread.

Expected saving: one ~210 KiB copy launch per rank per round (small; ~10 µs class). Build it only if it is clean.
