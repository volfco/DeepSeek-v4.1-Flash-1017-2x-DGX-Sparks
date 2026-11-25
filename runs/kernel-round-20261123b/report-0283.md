# report-0283 — TF_DS_L2_DELAY_NS / TF_DS_L2_RATE_GBPS / TF_DS_L2_BUDGET_MIB

**Commit:** `1421187` on `kernel-stack-0280` (`~/Workspace/tfref-kstack5`, parent `92dc42f`). It is the branch tip.
**Subject:** `0283 dsv41: TF_DS_L2_DELAY_NS / TF_DS_L2_RATE_GBPS / TF_DS_L2_BUDGET_MIB (unset: 3000 / 230 / 16): the L2 prefetch's pacing and budget`
**Nothing here ran on a GPU.**

## Switches

| Variable | Unset | Accepted | Becomes |
|---|---|---|---|
| `TF_DS_L2_DELAY_NS` | 3000 | whole ns, 0..1,000,000 | the launch's `delay_ns` (0: no wait before the first wave) |
| `TF_DS_L2_RATE_GBPS` | 230 | a number, 1..2000 (a fraction is fine) | the launch's `wave_ns` = `@intFromFloat(128 × 8192 / rate)` |
| `TF_DS_L2_BUDGET_MIB` | 16 | whole MiB, 1..256 | a launch's most bytes; the sites keep their `/2` and `/4` splits |

`""` is unset. Anything else prints `<VARIABLE>: expected …` to stderr for each bad one and refuses startup
(`error.BadL2Knobs`, at `Ops` init). `nan` and `inf` are refused. A wave is never 0 ns (the kernel reads 0 as
unpaced): the fastest accepted rate gives 524 ns.

Values the screen uses: rate 180 → `wave_ns` 5825; 270 → 3883; delay 0 and 10000; budget 8 and 32 MiB.

**Banner**, once a rank at load, only when at least one of the three is set:
`dsv41: TF_DS_L2_DELAY_NS / TF_DS_L2_RATE_GBPS / TF_DS_L2_BUDGET_MIB: the L2 prefetch waits <d> ns before its first wave, issues a wave of 128 chunks of 8 KiB every <w> ns (<r> GB/s) and takes <m> MiB a launch at most`.
The GB/s it prints is recomputed from the integer `wave_ns`, so it is the effective rate.

## What changed

- `l2_knobs.zig` (new, pure): the three parsers, `Knobs { delay_ns, wave_ns, budget }` whose defaults are today's
  numbers, `waveNs`, `fromText`, `anySet`, 3 tests.
- `ops.zig`: `Ops.l2: l2_knobs.Knobs = .{}` read once in `Ops.load`; `l2Prefetch` passes `o.l2.delay_ns` and
  `o.l2.wave_ns` (still `c_longlong`). An `Ops` not built through `load` keeps the defaults.
- `round.zig`: the constant `prefetch_budget` is gone; its six uses read `e.ops.l2.budget` (behind q's linears,
  `/2` behind `wo_b`, `/4` before an Engram layer, the next layer's inputs, the hot-set `/2`, the head).
- `draft.zig`, `draft_prefetch.zig`: 0282's sites read `e.ops.l2.budget` too. No use of either old constant remains
  (`grep -rn 'prefetch_budget\|draft_prefetch\.budget' zig/src` is empty).

The values are launch arguments and a captured graph bakes them. They are read before any capture and nothing reads
the environment a round.

## Launches at the served shape

None added or removed. Unset, every prefetch launch has the arguments `2bb052f` passed. Set, the same launches carry
different `delay_ns` / `wave_ns` and cut their ranges at a different byte count (a larger budget can add ranges to a
launch, up to the kernel's 16).

## Why the bits cannot move

The kernel writes nothing. Unset is argument-identical: the host test `unset is today's launch arguments and budget,
to the integer` pins `delay_ns` 3000, `wave_ns` 4559 (computed in the test by `2bb052f`'s own expression,
`@intFromFloat(128.0 * 8192.0 / 230.0)`) and 16,777,216 bytes with 8 MiB and 4 MiB splits, for the struct default,
for all three unset, for all three empty and for `3000` / `230` / `16` spelled out. Rank-local: a pair may run
different values on each rank.

## Gate

- `zig build test`: exit 0. `zig build`: exit 0. The test log's last line is a `failed command: …/test …` label,
  which Zig 0.17's build runner prints with any test step that wrote to stderr (base `2bb052f` prints it too); the
  family's test binary run directly ends `All 523 tests passed.` with exit 0, the three `l2_knobs` tests among them.
  The log holds the 40 refusal lines of the bounds test.
- `cuda-syntax-stubs/check.sh`: the same 9 errors as base `2bb052f`, output identical after path normalization.
  `git diff 2bb052f --name-only` lists no `.cu` file for the whole round.

## Deviations from the brief

None in behaviour. The module's default for the rate is carried as `wave_ns`, not as the rate, because the launch
argument is what must stay identical.

## What only hardware can settle

- Everything about the values: whether 230 GB/s, 3 µs and 16 MiB are near the best for this pair, and whether a
  per-rank setting removes the roughly 0.4% one-stream rank imbalance that round B2 traced to the prefetch.
- The screen is one knob at a time. An interaction (a larger budget needing a faster pace to finish in time) would
  need a second pass.
- With `TF_DS_DRAFT_PREFETCH` off the knobs touch only the forward. In a combined arm they move both.
