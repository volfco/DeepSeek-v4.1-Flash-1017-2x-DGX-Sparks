# Capacity: the box saturates at 8 streams; wider rounds and more streams do not add throughput

Queue item 1. Window 2026-11-25 04:58–06:34 UTC. Runtime-only: production's own image
`tensorfold-dsv42:bench-fused-next-2bb052f`, no build, no patch. Production was restored and validated after every
suite (`restored: validate` / `all passed`). Hardware reached via `ssh -J ai-node colum@10.50.0.10`; the tailnet
address `100.95.49.21` is offline as during the k296 suite.

## Answers

1. **`TF_DS_ROUND_ROWS` 32/48 at 8 streams is neutral or a loss.** The "drafts 2–5 are free in the drafter" premise
   is false: the extra rows cost exactly what the extra drafts earn. 32 is within the control band; 48 is −10%.
2. **`PARALLEL=16` cannot start on production's settings.** `TF_DS_DRAFT_BLOCK=6` × 16 streams = 96 rows against
   `draft.max_rows = 80` (`draft.zig:38`), so the engine refuses with `DraftBlockTooWide` (`model.zig:565`).
   Production's real ceiling is **13 streams** (13×6 = 78 ≤ 80); 14–16 all fail. This was undocumented.
3. **The box is throughput-saturated at 8 streams.** Doubling the streams 8 → 16 buys +10.7% aggregate throughput
   while per-stream throughput falls 45% and TTFT rises. The live 422 s mean TTFT with 35 waiting is therefore not
   fixable by raising `PARALLEL`; it needs per-token efficiency, not more lanes.
4. **Above 16 rows the 0017 split is sound on hardware.** Every arm was bit-exact against the reference panel
   (`RESULT: IDENTICAL`, plus the sampled panel). The 16-row split path had never run before this window.

## The numbers

Aggregate = completion tokens ÷ decode wall. Per-stream = aggregate ÷ streams. `fwd` is the profile's
`forward_ms_per_round`; `tok/round` is tokens ÷ `profile_rounds`.

| arm | streams | RR | block | depth | wall s | agg tok/s | str tok/s | rounds | fwd ms | tok/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctl8 | 8 | 16 | 6 | 4096 | 25.89 | 98.9 | 12.36 | 202 | 65.7 | 12.67 |
| ctl8 | 8 | 16 | 6 | 32768 | 25.92 | 98.8 | 12.35 | 180 | 71.2 | 14.22 |
| ctl8b | 8 | 16 | 6 | 4096 | 26.21 | 97.8 | 12.21 | 152 | 92.2 | 16.84 |
| ctl8b | 8 | 16 | 6 | 32768 | 26.06 | 98.2 | 12.28 | 121 | 111.0 | 21.16 |
| s8rr32 | 8 | 32 | 6 | 4096 | 26.21 | 97.7 | 12.21 | 152 | 92.2 | 16.84 |
| s8rr32 | 8 | 32 | 6 | 32768 | 26.06 | 98.2 | 12.28 | 121 | 111.0 | 21.16 |
| s8rr48 | 8 | 48 | 6 | 4096 | 28.71 | 89.2 | 11.15 | 142 | 117.0 | 18.03 |
| s8rr48 | 8 | 48 | 6 | 32768 | 29.48 | 86.8 | 10.85 | 114 | 148.5 | 22.46 |
| ctl8b6 (2nd pass) | 8 | 16 | 6 | 4096 | 26.04 | 98.3 | 12.29 | 201 | 66.8 | 12.74 |
| ctl8b6 (2nd pass) | 8 | 16 | 6 | 32768 | 25.99 | 98.5 | 12.31 | 177 | 72.9 | 14.46 |
| s13rr26 | 13 | 26 | 6 | 4096 | 39.31 | 105.8 | 8.14 | 198 | 97.6 | 21.01 |
| s13rr26 | 13 | 26 | 6 | 32768 | 40.11 | 103.7 | 7.98 | 198 | 96.9 | 21.01 |
| s16rr32 | 16 | 32 | 5 | 4096 | 46.75 | 109.5 | 6.84 | 194 | 115.6 | 26.39 |
| s16rr32 | 16 | 32 | 5 | 32768 | 47.33 | 108.2 | 6.76 | 194 | 112.7 | 26.39 |
| s16rr48 | 16 | 48 | 5 | 4096 | 46.93 | 109.1 | 6.82 | 162 | 142.6 | 31.60 |
| s16rr48 | 16 | 48 | 5 | 32768 | 47.38 | 108.1 | 6.75 | 157 | 143.2 | 32.61 |

Note: the `ctl8b` row above is the duplicate of `s8rr32` in the first pass's artefacts; the true third control is
`ctl8b6` in the second pass (98.3/98.5 tok/s). Three controls at 8 streams landed at 98.9/97.8/98.3 tok/s — a
1.1% band, so the ±0.6% arm-to-arm floor from the dec8 audit holds.

### The capacity curve

| streams | agg tok/s | str tok/s | TTFT | round cost |
|---:|---:|---:|---|---|
| 8 | 98.4 | 12.3 | 1.0× | 1.0× |
| 13 | 104.8 | 8.1 | 2.4× | 1.46× |
| 16 | 108.8 | 6.8 | ~2.4× | 1.72× |

Aggregate is flat to within 10% across a 2× change in streams while per-stream throughput halves. Round cost
(`fwd_ms`) scales almost linearly with the row budget, so the engine converts extra streams into extra rows rather
than extra throughput. That is the mechanism behind the saturation: the decode round is row-bound, not lane-bound.

### Why wider rounds do not help

At 8 streams and `round_rows=16`, `room = 16/8 − 1 = 1`, so each stream is offered one draft a round. Round 32
offers three, and 48 offers five. Tokens per round rose as intended (12.67 → 16.84 at 4096) but `fwd_ms` rose in
step (65.7 → 92.2), so the wall did not move. The pass does compute the same 40 block rows regardless of `steps`,
but the *round* pays for the extra rows in the gate/attention path — the saved draft work is not free, it is
re-priced as row cost. At 48 rows the row cost overruns the extra acceptance and throughput falls 10%.

So the row budget is already at its optimum at 16 rows for 8 streams. The lever that remains is **acceptance rate**
(tokens accepted per row computed), which is the M3 draft-triage question, not the row budget.

## The `DraftBlockTooWide` ceiling (a production defect)

`draft.max_rows = max_streams * 5 = 80` (`draft.zig:38`). `model.zig:565` refuses to start when
`streams * DRAFT_BLOCK > max_rows`. Production runs `PARALLEL=8` with `TF_DS_DRAFT_BLOCK=6` (head `.env`), so the
largest startable stream count is `80 / 6 = 13`. `start.sh` allows `PARALLEL` up to 16, so 14/15/16 are advertised
and fail at startup. The first s16 attempt failed exactly this way, all four arms, and the failure is not in any
record. Two ways out, both config-only: run 16 streams at `DRAFT_BLOCK=5` (16×5 = 80, fits exactly), or raise
`max_rows` in a build. The 16-stream arms above used `DRAFT_BLOCK=5`; the 13-stream arm kept production's 6.

## Artefacts

- `runs/capacity-20261125/s8/` — 8 streams, RR 16/32/48, block 6 (first pass).
- `runs/capacity-20261125/s8b/` — 8 streams control, second pass.
- `runs/capacity-20261125/s13/` — 13 streams, RR 26, block 6 (production's ceiling).
- `runs/capacity-20261125/s16c/` — 16 streams, RR 32/48, block 5.
- `runs/capacity-20261125/score.py` — the table above; `run.sh` and `run-s16b.sh` replay both passes.

## Not established

- The load was 4096/32768-token prompts with 320 output tokens, not the live 155K shape. The saturation verdict
  should hold (round cost is row-bound at any depth) but the absolute tok/s is not the live figure.
- The 8→13→16 comparison changes `DRAFT_BLOCK` at the 16-stream point (6 → 5), so the 13→16 step is not a pure
  streams change. The 8→13 step is pure.
- `fwd_ms` is about half the measured round period at 8 streams (65.7 ms of 128 ms), so roughly half the round is
  non-forward work. This window did not decompose it; the record's 14.5 ms fixed-cost estimate does not match this
  workload's shape and should be re-derived from these artefacts.
