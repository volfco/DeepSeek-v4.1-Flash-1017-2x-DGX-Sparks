# BRIEF-0278 — TF_DS_TAPS_PACKED: prompt taps written directly in DSpark's input layout

**Switch:** `TF_DS_TAPS_PACKED` (off by default). **Module:** `zig/src/families/dsv41/taps_packed.zig`.
**Commit subject:** `0278 dsv41: TF_DS_TAPS_PACKED (off by default): prompt taps written in the drafter's row-major layout`
**todo.md item:** "Write prompt taps directly in DSpark's input layout."

- Today each of the three tapped layers writes its own plane of `ch.taps` (`bf16 [taps, cap, D]`, `prompt.zig`
  `hcMean4` at the tap, `ch.taps + j * cap * D`), and at chunk completion the serving path copies the planes into the
  row-major `[rows, taps * D]` layout `absorb` consumes. Decode already writes taps side by side, and `hcMean4` takes an
  output stride.
- With the switch on, write each tap directly at stride `taps * D` into the packed layout and drop the completion copies
  (todo: three copy launches and 3.75 MiB a 128-row chunk per rank). Bytes consumed by `absorb` must be identical.
- **Find every reader of the planar layout** (todo lists `round.zig` ~1978, `exact.zig` ~355, `prompt.zig` ~2582–2590,
  `model.zig` ~1700–1704, `layer_main.zig` ~892 — re-find by name): probes, exact tooling, layer_main. Each must either
  read the packed layout under the switch or stay correct. If any reader cannot be adapted without changing the off path,
  say so in the report.
- No new kernel is needed if `hcMean4`'s stride suffices; then the "harness" is a host emulation of the write/copy
  layouts (planar + copy vs packed) over full and partial chunks (rows 1, 127, 128, 2048, a partial last chunk).
- Engagement banner on first packed write. Both ranks alike.
