# BRIEF-0283 — the L2 prefetch's pacing and budget as knobs

**Switches:** `TF_DS_L2_DELAY_NS` (unset: 3000), `TF_DS_L2_RATE_GBPS` (unset: 230), `TF_DS_L2_BUDGET_MIB` (unset: 16).
**Module:** `zig/src/families/dsv41/l2_knobs.zig`.

**Why.** `ops.zig:410–420` hard-codes the kit's served settings (3 µs before the first wave, waves of 128 chunks of
8 KiB paced at 230 GB/s) and `round.zig:1787` its 16 MiB budget (half behind `wo_b`, a quarter before an Engram
layer). They were copied from the kit and never swept on this pair. The prefetch is worth 5.3% in total, and round B2
traced the rank imbalance at one stream to it (it saves rank 0 about 4 µs a gather more than rank 1).

**Build.** Read the three values once at load (rank-local: a pair may run different values on each rank without
moving a bit, since the kernel writes nothing). Unset reproduces today's arguments exactly: `delay_ns = 3000`,
`wave_ns = @intFromFloat(128.0 * 8192.0 / 230.0)`, budget `16 << 20` with today's `/2` and `/4` splits. Validate:
delay 0..1,000,000 ns, rate 1..2,000 GB/s (a float is fine), budget 1..256 MiB. The values are launch arguments, so
a captured graph bakes them: read them before any capture, never per round. One banner line at load when any is set,
printing all three effective values. The drafter's prefetch (0282) uses the same values.

**Parity.** Writes nothing. Unset: launch arguments identical to `2bb052f` (add a host test that pins the three
default numbers and the `wave_ns` integer).
